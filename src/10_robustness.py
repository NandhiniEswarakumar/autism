import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import gc

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 10 -- ROBUSTNESS TESTING (Laptop-Safe)
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
A. 5-Fold Stratified Cross-Validation
B. Noise Robustness   (Gaussian noise sigma = 0, 0.05, 0.10, 0.20, 0.30)
C. Missing Value Robustness (masking 0%, 5%, 10%, 20%, 30%)
================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
DEVICE      = torch.device("cpu")
EPOCHS      = 30          # fast but enough
BATCH_SIZE  = 2048
PATIENCE    = 6
N_FOLDS     = 5
RANDOM_SEED = 42

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  ROBUSTNESS TESTING")
print(f"  Device : {DEVICE}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD CONFIG & PRE-PROCESSED DATA
# ─────────────────────────────────────────────────────────────
with open(f"{RESULTS_DIR}/model_config.json") as f:
    cfg = json.load(f)
with open(f"{RESULTS_DIR}/best_hyperparams.json") as f:
    hp = json.load(f)

X_te_np = np.load("dataset/X_test_final.npy").astype(np.float32)
y_te_np = np.load("dataset/y_test_final.npy").astype(int)
n_features = X_te_np.shape[1]

# ─────────────────────────────────────────────────────────────
# SHARED TRAINING HELPER
# ─────────────────────────────────────────────────────────────
def quick_train_eval(X_tr, y_tr, X_te, y_te_labels,
                     hidden_dim=64, n_layers=2, k=8, dropout=0.2,
                     lr=3e-3, wd=1e-4):
    """Train a fresh TabM for EPOCHS epochs, return best test AUC + probs."""
    n_pos = max((y_tr == 1).sum(), 1)
    n_neg = max((y_tr == 0).sum(), 1)
    pw    = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    model = TabM.build(n_features, hidden_dim=hidden_dim,
                       n_layers=n_layers, k=k, dropout=dropout)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    ldr   = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    best_auc = 0.5; best_p = np.zeros(len(y_te_labels)); pat = 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in ldr:
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(X_te_t)).numpy()
        try:
            auc = roc_auc_score(y_te_labels, p)
        except Exception:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc = auc; best_p = p; pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    del model; gc.collect()
    return best_auc, best_p

# ─────────────────────────────────────────────────────────────
# A. 5-FOLD STRATIFIED CV
# ─────────────────────────────────────────────────────────────
print("\n-------------------------------------")
print("  A. 5-Fold Stratified CV")
print("-------------------------------------")

df = pd.read_csv("dataset/cleaned_asd_dataset.csv")
TARGET = "K2Q35A"
feat_cols = [c for c in df.columns if c != TARGET]
sel_feats = np.load("dataset/selected_features.npy",
                     allow_pickle=True).tolist()
sel_idx   = [feat_cols.index(f) for f in sel_feats]

X_cv = df[feat_cols].values[:, sel_idx].astype(np.float32)
y_cv = df[TARGET].values.astype(int)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    X_tr_f, X_te_f = X_cv[tr_idx], X_cv[te_idx]
    y_tr_f, y_te_f = y_cv[tr_idx], y_cv[te_idx]

    scaler   = StandardScaler()
    X_tr_f   = scaler.fit_transform(X_tr_f).astype(np.float32)
    X_te_f   = scaler.transform(X_te_f).astype(np.float32)

    # SMOTE only if minority is genuinely small
    ratio = (y_tr_f == 1).sum() / max((y_tr_f == 0).sum(), 1)
    if ratio < 0.4:
        sm = SMOTE(sampling_strategy=0.5, random_state=RANDOM_SEED)
        X_tr_f, y_tr_f = sm.fit_resample(X_tr_f, y_tr_f)

    auc_f, best_p_f = quick_train_eval(
        X_tr_f, y_tr_f.astype(np.float32),
        X_te_f, y_te_f)

    y_pred_f = (best_p_f >= 0.5).astype(int)
    m = {
        "Fold"    : fold,
        "AUC"     : round(auc_f, 4),
        "F1"      : round(f1_score(y_te_f, y_pred_f, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_te_f, y_pred_f), 4),
    }
    fold_metrics.append(m)
    print(f"  Fold {fold}  ->  AUC={m['AUC']:.4f}  "
          f"F1={m['F1']:.4f}  Acc={m['Accuracy']:.4f}")

cv_df   = pd.DataFrame(fold_metrics)
cv_mean = cv_df[["AUC", "F1", "Accuracy"]].mean().round(4)
cv_std  = cv_df[["AUC", "F1", "Accuracy"]].std().round(4)
print(f"\n  Mean  ->  AUC={cv_mean['AUC']}  F1={cv_mean['F1']}  "
      f"Acc={cv_mean['Accuracy']}")
print(f"  Std   ->  AUC={cv_std['AUC']}  F1={cv_std['F1']}  "
      f"Acc={cv_std['Accuracy']}")

# CV Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, ["AUC", "F1", "Accuracy"]):
    vals = cv_df[metric].tolist()
    ax.bar(range(1, N_FOLDS + 1), vals, color="#2E86AB",
           edgecolor="white", linewidth=1.2)
    ax.axhline(cv_mean[metric], color="#E84855", linestyle="--",
               linewidth=2, label=f"Mean={cv_mean[metric]:.4f}")
    ax.fill_between(range(0, N_FOLDS + 2),
                    cv_mean[metric] - cv_std[metric],
                    cv_mean[metric] + cv_std[metric],
                    alpha=0.15, color="#E84855")
    ax.set_xlabel("Fold"); ax.set_ylabel(metric)
    ax.set_xticks(range(1, N_FOLDS + 1))
    ax.set_title(f"{metric} per Fold", fontsize=12, fontweight="bold")
    ax.legend()
plt.suptitle("5-Fold Stratified CV -- TabM ASD Detection",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/27_cv_results.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved -> {PLOTS_DIR}/27_cv_results.png")

# ─────────────────────────────────────────────────────────────
# B. NOISE ROBUSTNESS  (uses saved model — no re-training)
# ─────────────────────────────────────────────────────────────
print("\n-------------------------------------")
print("  B. Noise Robustness")
print("-------------------------------------")

model_noise = TabM.build(n_features,
                          hidden_dim=cfg["hidden_dim"],
                          n_layers=cfg["n_layers"],
                          k=cfg["k"], dropout=0.0)
model_noise.load_state_dict(
    torch.load("models/tabm_best.pth", map_location="cpu"))
model_noise.eval()

noise_levels  = [0.0, 0.05, 0.10, 0.20, 0.30]
noise_metrics = []

for sigma in noise_levels:
    X_noisy = X_te_np + np.random.normal(0, sigma, X_te_np.shape).astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model_noise(
            torch.tensor(X_noisy, dtype=torch.float32))).numpy()
    try:
        auc = roc_auc_score(y_te_np, p)
    except Exception:
        auc = 0.5
    y_pred = (p >= 0.5).astype(int)
    noise_metrics.append({
        "Noise_sigma": sigma,
        "AUC"        : round(auc, 4),
        "F1"         : round(f1_score(y_te_np, y_pred, zero_division=0), 4),
        "Accuracy"   : round(accuracy_score(y_te_np, y_pred), 4),
    })
    print(f"  sigma={sigma:.2f}  ->  AUC={auc:.4f}  "
          f"F1={noise_metrics[-1]['F1']:.4f}")

noise_df = pd.DataFrame(noise_metrics)
fig, ax  = plt.subplots(figsize=(9, 5))
for metric, color in zip(["AUC", "F1", "Accuracy"],
                          ["#2E86AB", "#E84855", "#2EB872"]):
    ax.plot(noise_df["Noise_sigma"], noise_df[metric],
            marker="o", ms=7, lw=2.5, color=color, label=metric)
ax.set_xlabel("Gaussian Noise Sigma"); ax.set_ylabel("Score")
ax.set_title("Noise Robustness Analysis", fontsize=13, fontweight="bold")
ax.legend(); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/28_noise_robustness.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved -> {PLOTS_DIR}/28_noise_robustness.png")

# ─────────────────────────────────────────────────────────────
# C. MISSING VALUE ROBUSTNESS  (uses saved model)
# ─────────────────────────────────────────────────────────────
print("\n-------------------------------------")
print("  C. Missing Value Robustness")
print("-------------------------------------")

miss_rates   = [0.0, 0.05, 0.10, 0.20, 0.30]
miss_metrics = []

for rate in miss_rates:
    X_miss = X_te_np.copy()
    mask   = np.random.rand(*X_miss.shape) < rate
    X_miss[mask] = 0.0          # replace with mean (=0 after scaling)
    with torch.no_grad():
        p = torch.sigmoid(model_noise(
            torch.tensor(X_miss, dtype=torch.float32))).numpy()
    try:
        auc = roc_auc_score(y_te_np, p)
    except Exception:
        auc = 0.5
    y_pred = (p >= 0.5).astype(int)
    miss_metrics.append({
        "Miss_Rate": rate,
        "AUC"      : round(auc, 4),
        "F1"       : round(f1_score(y_te_np, y_pred, zero_division=0), 4),
        "Accuracy" : round(accuracy_score(y_te_np, y_pred), 4),
    })
    print(f"  miss={rate:.2f}  ->  AUC={auc:.4f}  "
          f"F1={miss_metrics[-1]['F1']:.4f}")

miss_df = pd.DataFrame(miss_metrics)
fig, ax = plt.subplots(figsize=(9, 5))
for metric, color in zip(["AUC", "F1", "Accuracy"],
                          ["#2E86AB", "#E84855", "#2EB872"]):
    ax.plot(miss_df["Miss_Rate"], miss_df[metric],
            marker="s", ms=7, lw=2.5, color=color, label=metric)
ax.set_xlabel("Missing Value Rate"); ax.set_ylabel("Score")
ax.set_title("Missing Value Robustness Analysis", fontsize=13, fontweight="bold")
ax.legend(); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/29_missing_robustness.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved -> {PLOTS_DIR}/29_missing_robustness.png")

# ─────────────────────────────────────────────────────────────
# SAVE ALL RESULTS
# ─────────────────────────────────────────────────────────────
robustness_results = {
    "cross_validation": {
        "fold_results": fold_metrics,
        "mean"        : cv_mean.to_dict(),
        "std"         : cv_std.to_dict(),
    },
    "noise_robustness"  : noise_metrics,
    "missing_robustness": miss_metrics,
}
with open(f"{RESULTS_DIR}/robustness_results.json", "w") as f:
    json.dump(robustness_results, f, indent=2)
print(f"\n  [OK] Saved -> {RESULTS_DIR}/robustness_results.json")

print("\n" + "=" * 60)
print("  ROBUSTNESS TESTING COMPLETE")
print(f"  CV  : AUC = {cv_mean['AUC']} +/- {cv_std['AUC']}")
print(f"  Noise  (sigma=0.20) : AUC = {noise_metrics[3]['AUC']}")
print(f"  Missing (rate=0.20) : AUC = {miss_metrics[3]['AUC']}")
print("=" * 60)
