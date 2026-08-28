import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import gc

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 7 -- SHAP EXPLAINABILITY
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
SHAP (SHapley Additive exPlanations) quantifies each feature's
contribution to every individual prediction.

Game-theory foundation:
  phi_i = Σ_{S subset F\{i}} [|S|!(|F|-|S|-1)!/|F|!] - [f(S∪{i})-f(S)]

Plots generated:
  A. Global -- Beeswarm Summary Plot
  B. Global -- Bar Plot (mean |SHAP|)
  C. Local  -- Waterfall Plot (single sample)
  D. Local  -- Force Plot (saved as HTML)
  E. Dependence Plot (top feature)
  F. Decision Plot

SHAP approach used: KernelExplainer (model-agnostic)
  -> Works with any PyTorch model
  -> Background: KMeans(50) summary of training data
================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM  # noqa

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
MODELS_DIR  = "models"
PLOTS_DIR   = "results/plots"
RESULTS_DIR = "results"
DEVICE      = "cpu"       # SHAP works best on CPU
N_EXPLAIN   = 200         # number of test samples to explain
BACKGROUND  = 50          # KMeans background size

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# FEATURE LABELS
# -------------------------------------------------------------
feature_names_map = {
    "SC_AGE_YEARS" : "Age (Years)",
    "SC_SEX"       : "Sex",
    "ALLERGIES"    : "Allergies",
    "DIABETES"     : "Diabetes",
    "HEART"        : "Heart Disease",
    "DOWNSYN"      : "Down Syndrome",
    "CYSTFIB"      : "Cystic Fibrosis",
}

print("=" * 60)
print("  SHAP EXPLAINABILITY")
print("=" * 60)

# -------------------------------------------------------------
# LOAD MODEL & DATA
# -------------------------------------------------------------
with open(f"{RESULTS_DIR}/model_config.json") as f:
    cfg = json.load(f)

selected_features = np.load("dataset/selected_features.npy", allow_pickle=True).tolist()
feature_labels    = [feature_names_map.get(f, f) for f in selected_features]

model = TabM.build(
    n_features = cfg["n_features"],
    hidden_dim = cfg["hidden_dim"],
    n_layers   = cfg["n_layers"],
    k          = cfg["k"],
    dropout    = 0.0,   # no dropout at inference
).to(DEVICE)
model.load_state_dict(
    torch.load(f"{MODELS_DIR}/tabm_best.pth", map_location=DEVICE))
model.eval()
print(f"  Model loaded  ({cfg['n_features']} features)")

X_train_bg = np.load("dataset/X_train_final.npy").astype(np.float32)
X_test_ex  = np.load("dataset/X_test_final.npy").astype(np.float32)
y_test_ex  = np.load("dataset/y_test_final.npy").astype(int)

# Subsample test for SHAP (expensive)
rng    = np.random.default_rng(42)
idx    = rng.choice(len(X_test_ex), size=min(N_EXPLAIN, len(X_test_ex)), replace=False)
X_ex   = X_test_ex[idx]
y_ex   = y_test_ex[idx]

print(f"  Explaining {len(X_ex)} test samples  "
      f"(background={BACKGROUND} KMeans centres)")

# -------------------------------------------------------------
# WRAPPER -- PyTorch model -> numpy predict function
# -------------------------------------------------------------
def predict_proba(x_np: np.ndarray) -> np.ndarray:
    """Returns P(ASD=1) for each sample."""
    x_t = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_t)
        probs  = torch.sigmoid(logits).cpu().numpy()
    return probs

# -------------------------------------------------------------
# BUILD EXPLAINER  (KernelExplainer with KMeans background)
# -------------------------------------------------------------
print("\n  Building KernelExplainer (background KMeans)…")
background = shap.kmeans(X_train_bg, BACKGROUND)
explainer  = shap.KernelExplainer(predict_proba, background)

print("  Computing SHAP values (this takes ~1-3 minutes)…")
shap_values = explainer.shap_values(X_ex, nsamples=100)
# shap_values shape: (n_samples, n_features)

# Save SHAP values
np.save(f"{RESULTS_DIR}/shap_values.npy",    shap_values)
np.save(f"{RESULTS_DIR}/shap_X_explain.npy", X_ex)
np.save(f"{RESULTS_DIR}/shap_y_explain.npy", y_ex)
print(f"  [OK]  SHAP values saved -> {RESULTS_DIR}/shap_values.npy")

# -------------------------------------------------------------
# PLOT A -- BEESWARM SUMMARY
# -------------------------------------------------------------
print("\n  Generating SHAP plots…")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_ex, feature_names=feature_labels,
                  plot_type="dot", show=False, max_display=15,
                  color_bar_label="Feature Value")
plt.title("SHAP Beeswarm Summary -- ASD Prediction", fontsize=14,
          fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/19_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/19_shap_beeswarm.png")

# -------------------------------------------------------------
# PLOT B -- BAR PLOT (mean |SHAP|)
# -------------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx    = np.argsort(mean_abs_shap)[::-1]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_ex, feature_names=feature_labels,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14,
          fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/20_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/20_shap_bar.png")

# -------------------------------------------------------------
# PLOT C -- WATERFALL PLOT  (one ASD-positive + one negative)
# -------------------------------------------------------------
base_val = explainer.expected_value
if np.isscalar(base_val):
    base_val = float(base_val)

def make_waterfall(idx_sample, tag):
    sv = shap_values[idx_sample]
    contrib = dict(zip(feature_labels, sv))
    
    # Sort by absolute contribution
    items = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    names  = [i[0] for i in items]
    values = [i[1] for i in items]
    
    cum = base_val
    cumulative = [cum]
    for v in values:
        cum += v
        cumulative.append(cum)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2EB872" if v >= 0 else "#E84855" for v in values]
    ax.barh(names, values, color=colors, edgecolor="white", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(base_val, color="gray", linestyle="--", linewidth=1.2,
               label=f"Base value = {base_val:.3f}")
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(f"SHAP Waterfall -- {tag}", fontsize=13, fontweight="bold")
    ax.legend(); ax.invert_yaxis()
    plt.tight_layout()
    fname = f"{PLOTS_DIR}/21_shap_waterfall_{tag.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK]  Saved -> {fname}")

# Pick one ASD and one non-ASD sample
asd_idx  = np.where(y_ex == 1)[0]
noasd_idx= np.where(y_ex == 0)[0]
if len(asd_idx)  > 0: make_waterfall(asd_idx[0],   "ASD_Sample")
if len(noasd_idx)> 0: make_waterfall(noasd_idx[0], "NonASD_Sample")

# -------------------------------------------------------------
# PLOT D -- DEPENDENCE PLOT (top 2 features)
# -------------------------------------------------------------
top2 = sorted_idx[:2]
for feat_idx in top2:
    feat_name  = feature_labels[feat_idx]
    shap.dependence_plot(feat_idx, shap_values, X_ex,
                         feature_names=feature_labels, show=False)
    plt.title(f"SHAP Dependence Plot -- {feat_name}", fontsize=13,
              fontweight="bold")
    plt.tight_layout()
    safe_name = feat_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{PLOTS_DIR}/22_shap_dependence_{safe_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [OK]  Saved -> {PLOTS_DIR}/22_shap_dependence_{safe_name}.png")

# -------------------------------------------------------------
# GLOBAL IMPORTANCE RANKING TABLE
# -------------------------------------------------------------
import pandas as pd
imp_df = pd.DataFrame({
    "Feature"   : [feature_labels[i] for i in sorted_idx],
    "Mean|SHAP|": [round(mean_abs_shap[i], 5) for i in sorted_idx],
    "Rank"      : range(1, len(feature_labels) + 1),
})
print("\n------------------------------------------")
print("  GLOBAL FEATURE IMPORTANCE (SHAP)")
print("------------------------------------------")
print(imp_df.to_string(index=False))
imp_df.to_csv(f"{RESULTS_DIR}/shap_feature_importance.csv", index=False)
print(f"\n  [OK]  Saved -> {RESULTS_DIR}/shap_feature_importance.csv")

# -------------------------------------------------------------
# COMBINED SHAP IMPORTANCE BAR (custom, publication-ready)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(feature_labels)))
ordered_labels = [feature_labels[i] for i in sorted_idx]
ordered_values = [mean_abs_shap[i] for i in sorted_idx]
bars = ax.barh(ordered_labels[::-1], ordered_values[::-1],
               color=colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, ordered_values[::-1]):
    ax.text(bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("Global Feature Importance via SHAP", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/23_shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/23_shap_global_importance.png")

print("\n" + "=" * 60)
print("  SHAP ANALYSIS COMPLETE")
print("=" * 60)



