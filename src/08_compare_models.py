import gc
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

DATASET_DIR = "dataset"
RESULTS_DIR = "results"
SEED = 42
FIXED_THRESHOLD = 0.50

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    arrays = {
        name: np.load(os.path.join(DATASET_DIR, f"{name}_enh.npy"))
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    }
    arrays["X_train"] = arrays["X_train"].astype(np.float32)
    arrays["X_val"] = arrays["X_val"].astype(np.float32)
    arrays["X_test"] = arrays["X_test"].astype(np.float32)
    arrays["y_train"] = arrays["y_train"].astype(int)
    arrays["y_val"] = arrays["y_val"].astype(int)
    arrays["y_test"] = arrays["y_test"].astype(int)

    feature_names = np.load(
        os.path.join(DATASET_DIR, "feature_names_enhanced.npy"), allow_pickle=True
    )
    num_idx = np.load(os.path.join(DATASET_DIR, "num_feature_idx.npy")).astype(int)
    bin_idx = np.load(os.path.join(DATASET_DIR, "bin_feature_idx.npy")).astype(int)
    expected = arrays["X_train"].shape[1]
    if len(feature_names) != expected or len(num_idx) + len(bin_idx) != expected:
        raise ValueError("Enhanced feature metadata does not match enhanced arrays")
    if arrays["X_val"].shape[1] != expected or arrays["X_test"].shape[1] != expected:
        raise ValueError("Enhanced train/validation/test arrays have different widths")

    print(f"Features: {expected} ({len(num_idx)} numerical + {len(bin_idx)} binary)")
    print(
        f"Splits: train={len(arrays['y_train'])}, val={len(arrays['y_val'])}, "
        f"test={len(arrays['y_test'])}"
    )
    print(
        f"Train positives after SMOTE: {arrays['y_train'].sum()} / "
        f"{len(arrays['y_train'])} ({arrays['y_train'].mean():.3f})"
    )
    print(
        f"Validation positives: {arrays['y_val'].sum()} / {len(arrays['y_val'])}; "
        f"test positives: {arrays['y_test'].sum()} / {len(arrays['y_test'])}"
    )
    return arrays, num_idx, bin_idx


def best_f1_threshold(y_true, probabilities):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 981), 0.50])
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def metrics_row(model, threshold_name, threshold, y_true, probabilities):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Model": model,
        "Threshold": threshold_name,
        "ThresholdValue": round(float(threshold), 4),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, predictions),
        "CohenKappa": cohen_kappa_score(y_true, predictions),
    }


def train_mlp(X_train, y_train, X_val, y_val):
    class MLP(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(width, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLP(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=2048, shuffle=True, num_workers=0,
    )
    val_tensor = torch.tensor(X_val)
    best_auc, best_probs, stale = -1.0, None, 0
    for _ in range(60):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(features), labels).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(val_tensor)).numpy()
        auc = roc_auc_score(y_val, probs)
        if auc > best_auc + 1e-4:
            best_auc, best_probs, stale = auc, probs, 0
        else:
            stale += 1
        if stale >= 10:
            break
    return model, best_probs


def train_tabm(X_train, y_train, X_val, y_val):
    model = TabM.build(
        n_features=X_train.shape[1], hidden_dim=128, n_layers=3, k=16, dropout=0.2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=1024, shuffle=True, num_workers=0,
    )
    val_tensor = torch.tensor(X_val)
    best_auc, best_state, best_probs, stale = -1.0, None, None, 0
    for _ in range(60):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(features), labels).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(val_tensor)).numpy()
        auc = roc_auc_score(y_val, probs)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_probs = probs
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    return model, best_probs


def main():
    data, num_idx, bin_idx = load_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    models = {}

    print("\nTraining models on the enhanced training split only...")
    print("[1/6] CatBoost")
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6, random_seed=SEED,
        eval_metric="AUC", verbose=False, thread_count=2, allow_writing_files=False,
    )
    models["CatBoost"].fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
    print("[2/6] XGBoost")
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9,
        colsample_bytree=0.9, eval_metric="auc", random_state=SEED, n_jobs=2,
    )
    models["XGBoost"].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("[3/6] LightGBM")
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=SEED, verbose=-1, n_jobs=2,
    )
    models["LightGBM"].fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    print("[4/6] MLP")
    mlp, mlp_val_probs = train_mlp(X_train, y_train, X_val, y_val)
    print("[5/6] TabNet")
    tabnet = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, seed=SEED,
        verbose=0, device_name="cpu",
    )
    tabnet.fit(
        X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)],
        eval_name=["val"], eval_metric=["auc"], max_epochs=80,
        patience=12, batch_size=2048, virtual_batch_size=256,
    )
    print("[6/6] TabM")
    tabm, tabm_val_probs = train_tabm(X_train, y_train, X_val, y_val)

    val_probs = {
        "CatBoost": models["CatBoost"].predict_proba(X_val)[:, 1],
        "XGBoost": models["XGBoost"].predict_proba(X_val)[:, 1],
        "LightGBM": models["LightGBM"].predict_proba(X_val)[:, 1],
        "MLP": mlp_val_probs,
        "TabNet": tabnet.predict_proba(X_val)[:, 1],
        "TabM": tabm_val_probs,
    }
    test_probs = {
        "CatBoost": models["CatBoost"].predict_proba(X_test)[:, 1],
        "XGBoost": models["XGBoost"].predict_proba(X_test)[:, 1],
        "LightGBM": models["LightGBM"].predict_proba(X_test)[:, 1],
        "MLP": torch.sigmoid(mlp(torch.tensor(X_test))).detach().numpy(),
        "TabNet": tabnet.predict_proba(X_test)[:, 1],
        "TabM": torch.sigmoid(tabm(torch.tensor(X_test))).detach().numpy(),
    }

    rows, matrices = [], []
    for name in ("CatBoost", "XGBoost", "LightGBM", "MLP", "TabNet", "TabM"):
        threshold = best_f1_threshold(y_val, val_probs[name])
        rows.append(metrics_row(name, "0.50", FIXED_THRESHOLD, y_test, test_probs[name]))
        rows.append(metrics_row(name, "Validation-best-F1", threshold, y_test, test_probs[name]))
        predictions = (test_probs[name] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        matrices.append({"Model": name, "Threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp})

    table = pd.DataFrame(rows)
    metric_columns = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa"]
    auc_consistency = table.groupby("Model")["ROC-AUC"].nunique()
    if (auc_consistency > 1).any():
        raise ValueError("ROC-AUC must be identical across classification thresholds")
    table[metric_columns] = table[metric_columns].round(4)
    table = table.sort_values(["Threshold", "ROC-AUC"], ascending=[True, False])
    print("\n" + "=" * 120)
    print("FAIR BASELINE COMPARISON (test metrics; threshold selected on validation only)")
    print("=" * 120)
    print(table.to_string(index=False))
    print("\nCONFUSION MATRICES AT VALIDATION-BEST-F1 THRESHOLD")
    print(pd.DataFrame(matrices).to_string(index=False))
    table.to_csv(os.path.join(RESULTS_DIR, "comparison_table_enhanced_fair.csv"), index=False)
    pd.DataFrame(matrices).to_csv(os.path.join(RESULTS_DIR, "confusion_matrices_enhanced_fair.csv"), index=False)
    print("\nSaved results/comparison_table_enhanced_fair.csv")
    print("Saved results/confusion_matrices_enhanced_fair.csv")

    del models, mlp, tabnet, tabm
    gc.collect()


if __name__ == "__main__":
    main()
    raise SystemExit(0)
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
