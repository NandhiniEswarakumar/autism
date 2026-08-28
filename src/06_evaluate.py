import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import gc

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 6 -- FULL PERFORMANCE EVALUATION
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
Metrics generated:
  • Accuracy, Precision, Recall, F1, Specificity, Sensitivity
  • ROC-AUC, PR-AUC, MCC, Cohen's Kappa

Plots:
  • Confusion Matrix (normalised + raw)
  • ROC Curve
  • Precision-Recall Curve
  • Calibration Curve
  • Threshold Sensitivity Analysis

Output -> results/metrics.json + results/plots/
================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM   # noqa

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
MODELS_DIR  = "models"
PLOTS_DIR   = "results/plots"
RESULTS_DIR = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD   = 0.50

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  PERFORMANCE EVALUATION")
print("=" * 60)

# -------------------------------------------------------------
# LOAD MODEL CONFIG & MODEL
# -------------------------------------------------------------
with open(f"{RESULTS_DIR}/model_config.json") as f:
    cfg = json.load(f)

model = TabM.build(
    n_features = cfg["n_features"],
    hidden_dim = cfg["hidden_dim"],
    n_layers   = cfg["n_layers"],
    k          = cfg["k"],
    dropout    = cfg["dropout"],
).to(DEVICE)
model.load_state_dict(torch.load(f"{MODELS_DIR}/tabm_best.pth",
                                  map_location=DEVICE))
model.eval()
print(f"\n  Model loaded  ->  {cfg['n_features']} features  "
      f"| hidden={cfg['hidden_dim']}  | k={cfg['k']}  | layers={cfg['n_layers']}")

# -------------------------------------------------------------
# LOAD TEST DATA
# -------------------------------------------------------------
X_test = torch.tensor(np.load("dataset/X_test_final.npy"), dtype=torch.float32)
y_test = np.load("dataset/y_test_final.npy").astype(int)

with torch.no_grad():
    logits = model(X_test.to(DEVICE))
    probs  = torch.sigmoid(logits).cpu().numpy()

y_pred = (probs >= THRESHOLD).astype(int)

# -------------------------------------------------------------
# METRIC COMPUTATION
# -------------------------------------------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

acc      = accuracy_score(y_test, y_pred)
prec     = precision_score(y_test, y_pred, zero_division=0)
rec      = recall_score(y_test, y_pred, zero_division=0)
f1       = f1_score(y_test, y_pred, zero_division=0)
spec     = specificity(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, probs)
pr_auc   = average_precision_score(y_test, probs)
mcc      = matthews_corrcoef(y_test, y_pred)
kappa    = cohen_kappa_score(y_test, y_pred)

metrics = {
    "Accuracy"     : round(acc,     4),
    "Precision"    : round(prec,    4),
    "Recall(Sens)": round(rec,     4),
    "Specificity"  : round(spec,    4),
    "F1-Score"     : round(f1,      4),
    "ROC-AUC"      : round(roc_auc, 4),
    "PR-AUC"       : round(pr_auc,  4),
    "MCC"          : round(mcc,     4),
    "Cohen_Kappa"  : round(kappa,   4),
}

print("\n------------------------------------------")
print("  PERFORMANCE METRICS  (Threshold = 0.50)")
print("------------------------------------------")
for k_m, v_m in metrics.items():
    bar_len = int(v_m * 30)
    bar     = "█" * bar_len
    print(f"  {k_m:<18} {v_m:.4f}  {bar}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No ASD", "ASD"]))

# Save metrics
with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  [OK]  Saved -> {RESULTS_DIR}/metrics.json")

# -------------------------------------------------------------
# PLOT 1 -- CONFUSION MATRIX
# -------------------------------------------------------------
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm],
    ["d", ".2%"],
    ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"]
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                xticklabels=["No ASD", "ASD"],
                yticklabels=["No ASD", "ASD"],
                linewidths=1, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/14_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/14_confusion_matrix.png")

# -------------------------------------------------------------
# PLOT 2 -- ROC CURVE
# -------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, probs)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2E86AB", lw=2.5,
        label=f"TabM  (AUC = {roc_auc:.4f})")
ax.fill_between(fpr, tpr, alpha=0.12, color="#2E86AB")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve -- ASD Detection", fontsize=13, fontweight="bold")
ax.legend(loc="lower right"); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/15_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/15_roc_curve.png")

# -------------------------------------------------------------
# PLOT 3 -- PRECISION-RECALL CURVE
# -------------------------------------------------------------
prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
baseline_pr = y_test.sum() / len(y_test)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_curve, prec_curve, color="#E84855", lw=2.5,
        label=f"TabM  (AP = {pr_auc:.4f})")
ax.fill_between(rec_curve, prec_curve, alpha=0.12, color="#E84855")
ax.axhline(baseline_pr, color="gray", linestyle="--", lw=1.5,
           label=f"Baseline  (AP = {baseline_pr:.4f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve -- ASD Detection", fontsize=13, fontweight="bold")
ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/16_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/16_pr_curve.png")

# -------------------------------------------------------------
# PLOT 4 -- CALIBRATION CURVE
# -------------------------------------------------------------
fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(mean_pred, fraction_pos, marker="o", ms=6, lw=2,
        color="#2EB872", label="TabM Calibration")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve", fontsize=13, fontweight="bold")
ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/17_calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/17_calibration_curve.png")

# -------------------------------------------------------------
# PLOT 5 -- THRESHOLD SENSITIVITY
# -------------------------------------------------------------
thresholds = np.arange(0.1, 0.91, 0.05)
th_metrics  = {"accuracy": [], "precision": [], "recall": [],
               "f1": [], "specificity": []}
for th in thresholds:
    yp = (probs >= th).astype(int)
    th_metrics["accuracy"].append(accuracy_score(y_test, yp))
    th_metrics["precision"].append(precision_score(y_test, yp, zero_division=0))
    th_metrics["recall"].append(recall_score(y_test, yp, zero_division=0))
    th_metrics["f1"].append(f1_score(y_test, yp, zero_division=0))
    th_metrics["specificity"].append(specificity(y_test, yp))

fig, ax = plt.subplots(figsize=(10, 5))
colors_th = ["#2E86AB", "#E84855", "#2EB872", "#F4A261", "#9B59B6"]
for (metric_name, vals), color in zip(th_metrics.items(), colors_th):
    ax.plot(thresholds, vals, marker="o", ms=4, lw=2,
            color=color, label=metric_name.capitalize())
ax.axvline(THRESHOLD, color="gray", linestyle="--", lw=1.5,
           label=f"Default threshold ({THRESHOLD})")
ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score")
ax.set_title("Metric Sensitivity to Threshold", fontsize=13, fontweight="bold")
ax.legend(loc="center right"); ax.set_xlim([0.1, 0.9]); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/18_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/18_threshold_sensitivity.png")

# -------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("  EVALUATION COMPLETE")
print("=" * 60)
print(f"  Accuracy   : {acc:.4f}")
print(f"  F1-Score   : {f1:.4f}")
print(f"  ROC-AUC    : {roc_auc:.4f}")
print(f"  PR-AUC     : {pr_auc:.4f}")
print(f"  MCC        : {mcc:.4f}")
print(f"  Kappa      : {kappa:.4f}")
print("=" * 60)



