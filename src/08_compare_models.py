import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import gc

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 8 -- BASELINE MODEL COMPARISON (Laptop-Safe)
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
Models compared:
  1. TabM        (our model)
  2. MLP         (PyTorch, 2 layers)
  3. TabNet      (pytorch-tabnet)
  4. XGBoost     (gradient boosting)
  5. LightGBM    (gradient boosting)
  6. CatBoost    (gradient boosting)
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, roc_curve,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM

warnings.filterwarnings("ignore")

MODELS_DIR  = "models"
RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
RANDOM_SEED = 42
THRESHOLD   = 0.50
DEVICE      = torch.device("cpu")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  BASELINE MODEL COMPARISON")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
X_tr = np.load("dataset/X_train_final.npy").astype(np.float32)
y_tr = np.load("dataset/y_train_final.npy").astype(int)
X_va = np.load("dataset/X_val_final.npy").astype(np.float32)
y_va = np.load("dataset/y_val_final.npy").astype(int)
X_te = np.load("dataset/X_test_final.npy").astype(np.float32)
y_te = np.load("dataset/y_test_final.npy").astype(int)

X_tr_all = np.concatenate([X_tr, X_va], axis=0)
y_tr_all  = np.concatenate([y_tr, y_va], axis=0)
n_features = X_tr.shape[1]

print(f"  Train: {X_tr_all.shape}  |  Test: {X_te.shape}")

# ─────────────────────────────────────────────────────────────
# EVALUATION HELPER  (fixed: takes y_true, y_prob)
# ─────────────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_prob, threshold=THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC"   : round(average_precision_score(y_true, y_prob), 4),
        "MCC"      : round(matthews_corrcoef(y_true, y_pred), 4),
    }

results   = []
all_probs = {}

# ─────────────────────────────────────────────────────────────
# 1. TABM  (load saved model)
# ─────────────────────────────────────────────────────────────
print("\n  [1/6] TabM (our model) ...")
with open(f"{RESULTS_DIR}/model_config.json") as f:
    cfg = json.load(f)

tabm = TabM.build(n_features=cfg["n_features"], hidden_dim=cfg["hidden_dim"],
                   n_layers=cfg["n_layers"], k=cfg["k"], dropout=0.0)
tabm.load_state_dict(torch.load(f"{MODELS_DIR}/tabm_best.pth", map_location="cpu"))
tabm.eval()
with torch.no_grad():
    tabm_probs = torch.sigmoid(
        tabm(torch.tensor(X_te, dtype=torch.float32))).numpy()

results.append(evaluate_model("TabM (Ours)", y_te, tabm_probs))
all_probs["TabM (Ours)"] = tabm_probs
print(f"    ROC-AUC = {results[-1]['ROC-AUC']:.4f}")
gc.collect()

# ─────────────────────────────────────────────────────────────
# 2. MLP  (simple PyTorch, laptop-safe)
# ─────────────────────────────────────────────────────────────
print("\n  [2/6] MLP (PyTorch) ...")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

mlp     = MLP(n_features)
opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=3e-3)
crit    = nn.BCEWithLogitsLoss()
ldr_mlp = DataLoader(
    TensorDataset(torch.tensor(X_tr_all), torch.tensor(y_tr_all.astype(np.float32))),
    batch_size=2048, shuffle=True, num_workers=0)

best_auc_mlp = 0.0; best_p_mlp = None
for ep in range(30):                             # 30 epochs only
    mlp.train()
    for xb, yb in ldr_mlp:
        opt_mlp.zero_grad(set_to_none=True)
        crit(mlp(xb), yb).backward()
        opt_mlp.step()
    mlp.eval()
    with torch.no_grad():
        p = torch.sigmoid(mlp(torch.tensor(X_te))).numpy()
    auc = roc_auc_score(y_te, p)
    if auc > best_auc_mlp:
        best_auc_mlp = auc; best_p_mlp = p

results.append(evaluate_model("MLP", y_te, best_p_mlp))
all_probs["MLP"] = best_p_mlp
print(f"    ROC-AUC = {results[-1]['ROC-AUC']:.4f}")
del mlp; gc.collect()

# ─────────────────────────────────────────────────────────────
# 3. TABNET  (laptop-safe settings)
# ─────────────────────────────────────────────────────────────
print("\n  [3/6] TabNet ...")
tabnet = TabNetClassifier(
    n_d=8, n_a=8, n_steps=3, gamma=1.5,
    seed=RANDOM_SEED, verbose=0, device_name="cpu",
)
tabnet.fit(
    X_train=X_tr_all, y_train=y_tr_all,
    eval_set=[(X_te, y_te)], eval_metric=["auc"],
    max_epochs=50, patience=10, batch_size=2048,
)
tabnet_probs = tabnet.predict_proba(X_te)[:, 1]
results.append(evaluate_model("TabNet", y_te, tabnet_probs))
all_probs["TabNet"] = tabnet_probs
print(f"    ROC-AUC = {results[-1]['ROC-AUC']:.4f}")
del tabnet; gc.collect()

# ─────────────────────────────────────────────────────────────
# 4. XGBOOST
# ─────────────────────────────────────────────────────────────
print("\n  [4/6] XGBoost ...")
scale_pos = float((y_tr_all == 0).sum()) / float((y_tr_all == 1).sum())
xgb_model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    scale_pos_weight=scale_pos, eval_metric="auc",
    random_state=RANDOM_SEED, verbosity=0, n_jobs=2,
)
xgb_model.fit(X_tr_all, y_tr_all, eval_set=[(X_te, y_te)], verbose=False)
xgb_probs = xgb_model.predict_proba(X_te)[:, 1]
results.append(evaluate_model("XGBoost", y_te, xgb_probs))
all_probs["XGBoost"] = xgb_probs
print(f"    ROC-AUC = {results[-1]['ROC-AUC']:.4f}")
del xgb_model; gc.collect()

# ─────────────────────────────────────────────────────────────
# 5. LIGHTGBM
# ─────────────────────────────────────────────────────────────
print("\n  [5/6] LightGBM ...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1, num_leaves=31,
    scale_pos_weight=scale_pos, random_state=RANDOM_SEED,
    verbose=-1, n_jobs=2,
)
lgb_model.fit(X_tr_all, y_tr_all, eval_set=[(X_te, y_te)])
lgb_probs = lgb_model.predict_proba(X_te)[:, 1]
results.append(evaluate_model("LightGBM", y_te, lgb_probs))
all_probs["LightGBM"] = lgb_probs
print(f"    ROC-AUC = {results[-1]['ROC-AUC']:.4f}")
del lgb_model; gc.collect()

# ─────────────────────────────────────────────────────────────
# 6. CATBOOST
# ─────────────────────────────────────────────────────────────
print("\n  [6/6] CatBoost ...")
cat_model = CatBoostClassifier(
    iterations=200, learning_rate=0.1, depth=4,
    scale_pos_weight=scale_pos, random_seed=RANDOM_SEED,
    eval_metric="AUC", verbose=False, thread_count=2,
)
cat_model.fit(X_tr_all, y_tr_all, eval_set=(X_te, y_te))
cat_probs = cat_model.predict_proba(X_te)[:, 1]
results.append(evaluate_model("CatBoost", y_te, cat_probs))
all_probs["CatBoost"] = cat_probs
print(f"    ROC-AUC = {results[-1]['ROC-AUC']:.4f}")
del cat_model; gc.collect()

# ─────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n" + "=" * 60)
print("  COMPARISON TABLE")
print("=" * 60)
print(df_results.to_string(index=False))
df_results.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)
print(f"\n  [OK] Saved -> {RESULTS_DIR}/comparison_table.csv")

# ─────────────────────────────────────────────────────────────
# PLOT 1 -- GROUPED BAR CHART
# ─────────────────────────────────────────────────────────────
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
model_names = df_results["Model"].tolist()
x      = np.arange(len(metric_cols))
width  = 0.13
palette= plt.cm.tab10(np.linspace(0, 1, len(model_names)))

fig, ax = plt.subplots(figsize=(15, 6))
for i, (_, row) in enumerate(df_results.iterrows()):
    ax.bar(x + i * width, [row[m] for m in metric_cols],
           width=width, label=row["Model"],
           color=palette[i], edgecolor="white", linewidth=0.8)
ax.set_xticks(x + width * (len(model_names) - 1) / 2)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylabel("Score"); ax.set_ylim([0, 1.15])
ax.set_title("Model Comparison -- ASD Detection", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/24_model_comparison.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved -> {PLOTS_DIR}/24_model_comparison.png")

# ─────────────────────────────────────────────────────────────
# PLOT 2 -- ROC CURVES
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for (name, prob), color in zip(all_probs.items(), palette):
    fpr, tpr, _ = roc_curve(y_te, prob)
    auc_val = roc_auc_score(y_te, prob)
    lw = 3.0 if "TabM" in name else 1.8
    ls = "-"  if "TabM" in name else "--"
    ax.plot(fpr, tpr, lw=lw, linestyle=ls, color=color,
            label=f"{name}  (AUC={auc_val:.4f})")
ax.plot([0, 1], [0, 1], "k:", lw=1.5)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves -- All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/25_roc_all_models.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  [OK] Saved -> {PLOTS_DIR}/25_roc_all_models.png")

print("\n" + "=" * 60)
print("  COMPARISON COMPLETE")
print("=" * 60)
