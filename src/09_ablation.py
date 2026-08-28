import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import gc

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 9 -- ABLATION STUDY
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
Systematically remove one component at a time to quantify
its contribution to the final model's performance.

Configurations tested:
  1. Full Model          -- TabM + Feature Selection + HPO + SMOTE
  2. No SMOTE            -- Same model, no oversampling
  3. No Feature Sel.     -- Use all 7 raw features
  4. No HPO              -- Default hyperparameters
  5. Shallow TabM        -- Only 1 hidden layer
  6. Small Ensemble      -- k=4 instead of best k

Output:
  -> results/ablation_results.csv
  -> results/plots/26_ablation_study.png
================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM  # noqa

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 30
BATCH_SIZE  = 2048
PATIENCE    = 5
RANDOM_SEED = 42

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  ABLATION STUDY")
print("=" * 60)

# -------------------------------------------------------------
# LOAD BEST HYPERPARAMS
# -------------------------------------------------------------
with open(f"{RESULTS_DIR}/best_hyperparams.json") as f:
    best_hp = json.load(f)
with open(f"{RESULTS_DIR}/model_config.json") as f:
    cfg = json.load(f)

# -------------------------------------------------------------
# LOAD ORIGINAL CLEANED DATA (before feature selection)
# -------------------------------------------------------------
df = pd.read_csv("dataset/cleaned_asd_dataset.csv")

TARGET_COL = "K2Q35A"
X_all = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
y_all = df[TARGET_COL].values.astype(int)

selected_features = np.load("dataset/selected_features.npy",
                             allow_pickle=True).tolist()
all_features      = df.drop(columns=[TARGET_COL]).columns.tolist()
sel_idx           = [all_features.index(f) for f in selected_features]

# -------------------------------------------------------------
# TRAIN/EVALUATE HELPER
# -------------------------------------------------------------
def run_experiment(X_train_sc, y_train_np, X_test_sc, y_test_np,
                   hidden_dim=128, n_layers=3, k=32, dropout=0.1,
                   lr=1e-3, weight_decay=1e-4) -> dict:
    """Trains TabM and returns test metrics."""
    n_feat   = X_train_sc.shape[1]
    X_tr_t   = torch.tensor(X_train_sc, dtype=torch.float32)
    y_tr_t   = torch.tensor(y_train_np, dtype=torch.float32)
    X_te_t   = torch.tensor(X_test_sc,  dtype=torch.float32)

    n_pos    = (y_tr_t == 1).sum().float()
    n_neg    = (y_tr_t == 0).sum().float()
    pw       = (n_neg / n_pos).to(DEVICE)

    model    = TabM.build(n_feat, hidden_dim=hidden_dim,
                          n_layers=n_layers, k=k, dropout=dropout).to(DEVICE)
    optimizer= torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion= nn.BCEWithLogitsLoss(pos_weight=pw)
    loader   = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=BATCH_SIZE, shuffle=True)

    best_auc  = 0.0
    patience_ = 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(X_te_t.to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(y_test_np, p)
        if auc > best_auc + 1e-4:
            best_auc  = auc
            best_p    = p
            patience_ = 0
        else:
            patience_ += 1
            if patience_ >= PATIENCE:
                break

    y_pred = (best_p >= 0.5).astype(int)
    return {
        "AUC"     : round(best_auc, 4),
        "F1"      : round(f1_score(y_test_np, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_test_np, y_pred), 4),
    }


# -------------------------------------------------------------
# PREPARE SPLITS  (reusable for all configs)
# -------------------------------------------------------------
def prepare_data(X, y, use_smote=True, feature_idx=None):
    """Returns (X_train_sc, y_train, X_test_sc, y_test)."""
    if feature_idx is not None:
        X = X[:, feature_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)

    if use_smote:
        current_ratio = (y_tr == 1).sum() / max((y_tr == 0).sum(), 1)
        if current_ratio < 0.4:
            smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_SEED)
            X_tr_sc, y_tr = smote.fit_resample(X_tr_sc, y_tr)

    return X_tr_sc, y_tr.astype(np.float32), X_te_sc, y_te.astype(int)


# -------------------------------------------------------------
# ABLATION CONFIGURATIONS
# -------------------------------------------------------------
configs = [
    {
        "name"       : "Full Model\n(TabM + FS + HPO + SMOTE)",
        "use_smote"  : True,
        "feat_idx"   : sel_idx,
        "hidden_dim" : best_hp["hidden_dim"],
        "n_layers"   : best_hp["n_layers"],
        "k"          : best_hp["k"],
        "dropout"    : best_hp["dropout"],
        "lr"         : best_hp["lr"],
        "weight_decay": best_hp["weight_decay"],
    },
    {
        "name"       : "No SMOTE\n(without oversampling)",
        "use_smote"  : False,
        "feat_idx"   : sel_idx,
        "hidden_dim" : best_hp["hidden_dim"],
        "n_layers"   : best_hp["n_layers"],
        "k"          : best_hp["k"],
        "dropout"    : best_hp["dropout"],
        "lr"         : best_hp["lr"],
        "weight_decay": best_hp["weight_decay"],
    },
    {
        "name"       : "No Feature Selection\n(all 7 features)",
        "use_smote"  : True,
        "feat_idx"   : None,           # use all features
        "hidden_dim" : best_hp["hidden_dim"],
        "n_layers"   : best_hp["n_layers"],
        "k"          : best_hp["k"],
        "dropout"    : best_hp["dropout"],
        "lr"         : best_hp["lr"],
        "weight_decay": best_hp["weight_decay"],
    },
    {
        "name"       : "No HPO\n(default hyperparams)",
        "use_smote"  : True,
        "feat_idx"   : sel_idx,
        "hidden_dim" : 128,           # defaults
        "n_layers"   : 2,
        "k"          : 8,
        "dropout"    : 0.1,
        "lr"         : 1e-3,
        "weight_decay": 1e-4,
    },
    {
        "name"       : "Shallow TabM\n(1 hidden layer)",
        "use_smote"  : True,
        "feat_idx"   : sel_idx,
        "hidden_dim" : best_hp["hidden_dim"],
        "n_layers"   : 1,             # only 1 layer
        "k"          : best_hp["k"],
        "dropout"    : best_hp["dropout"],
        "lr"         : best_hp["lr"],
        "weight_decay": best_hp["weight_decay"],
    },
    {
        "name"       : "Small Ensemble\n(k=4 heads)",
        "use_smote"  : True,
        "feat_idx"   : sel_idx,
        "hidden_dim" : best_hp["hidden_dim"],
        "n_layers"   : best_hp["n_layers"],
        "k"          : 4,             # tiny ensemble
        "dropout"    : best_hp["dropout"],
        "lr"         : best_hp["lr"],
        "weight_decay": best_hp["weight_decay"],
    },
]

# -------------------------------------------------------------
# RUN ALL CONFIGURATIONS
# -------------------------------------------------------------
ablation_results = []
for i, cfg_exp in enumerate(configs):
    name = cfg_exp["name"].replace("\n", " ")
    print(f"\n  [{i+1}/{len(configs)}] {name}")
    X_tr_sc, y_tr, X_te_sc, y_te = prepare_data(
        X_all, y_all,
        use_smote  = cfg_exp["use_smote"],
        feature_idx= cfg_exp["feat_idx"],
    )
    metrics = run_experiment(
        X_tr_sc, y_tr, X_te_sc, y_te,
        hidden_dim   = cfg_exp["hidden_dim"],
        n_layers     = cfg_exp["n_layers"],
        k            = cfg_exp["k"],
        dropout      = cfg_exp["dropout"],
        lr           = cfg_exp["lr"],
        weight_decay = cfg_exp["weight_decay"],
    )
    ablation_results.append({"Configuration": name, **metrics})
    print(f"    AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  "
          f"Acc={metrics['Accuracy']:.4f}")

# -------------------------------------------------------------
# SAVE TABLE
# -------------------------------------------------------------
ablation_df = pd.DataFrame(ablation_results)
print("\n" + "=" * 60)
print("  ABLATION STUDY RESULTS")
print("=" * 60)
print(ablation_df.to_string(index=False))
ablation_df.to_csv(f"{RESULTS_DIR}/ablation_results.csv", index=False)
print(f"\n  [OK]  Saved -> {RESULTS_DIR}/ablation_results.csv")

# -------------------------------------------------------------
# PLOT -- SIDE-BY-SIDE BARS
# -------------------------------------------------------------
short_names = [c["name"] for c in configs]
metrics_plot= ["AUC", "F1", "Accuracy"]
palette     = ["#2E86AB", "#E84855", "#2EB872"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, color in zip(axes, metrics_plot, palette):
    vals = [r[metric] for r in ablation_results]
    bars = ax.bar(range(len(vals)), vals, color=color,
                  edgecolor="white", linewidth=1.2, width=0.6)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.05))
    ax.set_title(f"{metric}", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.axhline(vals[0], color="navy", linestyle="--",
               linewidth=1.5, label="Full Model")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend()

plt.suptitle("Ablation Study -- TabM ASD Detection",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/26_ablation_study.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/26_ablation_study.png")

print("\n" + "=" * 60)
print("  ABLATION STUDY COMPLETE")
print("=" * 60)





