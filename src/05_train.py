import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
Training script (laptop-safe defaults).

This script sets conservative defaults to allow training on CPU without
excessive memory or time usage. It uses a small TabM configuration and
fixed hyperparameters so the script can be run on typical student laptops.
"""

import os
import gc
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from tabm_model import TabM

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# LAPTOP-SAFE SETTINGS
# ─────────────────────────────────────────────────────────────
torch.set_num_threads(2)          # only use 2 CPU cores — prevents overheating
torch.set_num_interop_threads(1)

MODELS_DIR   = "models"
PLOTS_DIR    = "results/plots"
RESULTS_DIR  = "results"

# Fixed good hyperparameters (no Optuna needed — saves time + CPU)
BEST_PARAMS = {
    "hidden_dim"   : 64,     # small model = less RAM
    "n_layers"     : 2,
    "k"            : 8,      # 8 ensemble heads = good balance
    "dropout"      : 0.2,
    "lr"           : 3e-3,
    "weight_decay" : 1e-4,
    "batch_size"   : 1024,   # large batch = fewer gradient steps
}

TRAIN_EPOCHS = 50            # enough for convergence
PATIENCE     = 8
RANDOM_SEED  = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR,exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  TabM TRAINING  (Laptop-Safe Mode)")
print("  CPU Threads : 2   |   RAM target : < 500 MB")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA  (efficient)
# ─────────────────────────────────────────────────────────────
print("\n  Loading data ...")
X_train = torch.tensor(np.load("dataset/X_train_final.npy"), dtype=torch.float32)
y_train = torch.tensor(np.load("dataset/y_train_final.npy"), dtype=torch.float32)
X_val   = torch.tensor(np.load("dataset/X_val_final.npy"),   dtype=torch.float32)
y_val   = torch.tensor(np.load("dataset/y_val_final.npy"),   dtype=torch.float32)
X_test  = torch.tensor(np.load("dataset/X_test_final.npy"),  dtype=torch.float32)
y_test  = torch.tensor(np.load("dataset/y_test_final.npy"),  dtype=torch.float32)

n_features = X_train.shape[1]
print(f"  Train  : {X_train.shape}")
print(f"  Val    : {X_val.shape}")
print(f"  Test   : {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────
n_pos    = (y_train == 1).sum().float()
n_neg    = (y_train == 0).sum().float()
pw       = n_neg / n_pos

model    = TabM.build(n_features,
                      hidden_dim = BEST_PARAMS["hidden_dim"],
                      n_layers   = BEST_PARAMS["n_layers"],
                      k          = BEST_PARAMS["k"],
                      dropout    = BEST_PARAMS["dropout"])

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model parameters : {total_params:,}")
print(f"  Hyperparameters  : {BEST_PARAMS}")

optimizer = torch.optim.AdamW(model.parameters(),
                               lr           = BEST_PARAMS["lr"],
                               weight_decay = BEST_PARAMS["weight_decay"])
criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-5)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size = BEST_PARAMS["batch_size"],
    shuffle    = True,
    num_workers= 0,        # 0 workers = no extra processes
    pin_memory = False,    # no pinned memory = less RAM
)

# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
print(f"\n  Training for {TRAIN_EPOCHS} epochs ...")
print(f"  {'Epoch':<8} {'Loss':<12} {'Val AUC':<12} {'Test AUC':<12} {'Status'}")
print("  " + "-" * 55)

train_losses = []
val_aucs     = []
test_aucs    = []
best_val_auc = 0.0
best_test_auc= 0.0
patience_cnt = 0

for epoch in range(1, TRAIN_EPOCHS + 1):
    # --- Train ---
    model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)   # more memory efficient
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(yb)
    epoch_loss /= len(loader.dataset)
    scheduler.step()

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        val_probs  = torch.sigmoid(model(X_val)).numpy()
        test_probs = torch.sigmoid(model(X_test)).numpy()

    try:
        val_auc  = roc_auc_score(y_val.numpy(),  val_probs)
        test_auc = roc_auc_score(y_test.numpy(), test_probs)
    except:
        val_auc = test_auc = 0.5

    train_losses.append(epoch_loss)
    val_aucs.append(val_auc)
    test_aucs.append(test_auc)

    # Early stopping
    if val_auc > best_val_auc + 1e-4:
        best_val_auc  = val_auc
        best_test_auc = test_auc
        patience_cnt  = 0
        torch.save(model.state_dict(), f"{MODELS_DIR}/tabm_best.pth")
        status = "[SAVED]"
    else:
        patience_cnt += 1
        status = f"(patience {patience_cnt}/{PATIENCE})"

    # Print every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        print(f"  {epoch:<8} {epoch_loss:<12.4f} {val_auc:<12.4f} "
              f"{test_auc:<12.4f} {status}")

    # Early stop
    if patience_cnt >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}")
        break

    # Free memory every epoch
    gc.collect()

print(f"\n  Best Val AUC  : {best_val_auc:.4f}")
print(f"  Best Test AUC : {best_test_auc:.4f}")
print(f"  [OK] Model saved -> {MODELS_DIR}/tabm_best.pth")

# ─────────────────────────────────────────────────────────────
# SAVE CONFIG & PARAMS
# ─────────────────────────────────────────────────────────────
model_config = {
    "n_features"   : n_features,
    "hidden_dim"   : BEST_PARAMS["hidden_dim"],
    "n_layers"     : BEST_PARAMS["n_layers"],
    "k"            : BEST_PARAMS["k"],
    "dropout"      : BEST_PARAMS["dropout"],
    "best_val_auc" : round(best_val_auc,  4),
    "best_test_auc": round(best_test_auc, 4),
}
with open(f"{RESULTS_DIR}/model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

with open(f"{RESULTS_DIR}/best_hyperparams.json", "w") as f:
    json.dump({**BEST_PARAMS, "best_val_auc": best_val_auc}, f, indent=2)

print(f"  [OK] Saved -> {RESULTS_DIR}/model_config.json")
print(f"  [OK] Saved -> {RESULTS_DIR}/best_hyperparams.json")

# ─────────────────────────────────────────────────────────────
# LEARNING CURVE PLOTS
# ─────────────────────────────────────────────────────────────
ep_range = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(ep_range, train_losses, color="#2E86AB", lw=2, label="Train Loss")
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCEWithLogits Loss")
axes[0].legend()

axes[1].plot(ep_range, val_aucs,  color="#2EB872", lw=2, label="Val AUC")
axes[1].plot(ep_range, test_aucs, color="#E84855", lw=2, label="Test AUC",
             linestyle="--")
axes[1].axhline(best_test_auc, color="#F4A261", linestyle=":",
                lw=1.5, label=f"Best Test={best_test_auc:.4f}")
axes[1].set_title("AUC Curves")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
axes[1].legend()

plt.suptitle("TabM Training Curves (Laptop-Safe)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/13_training_curves.png", dpi=120, bbox_inches="tight")
plt.close()
gc.collect()
print(f"  [OK] Saved -> {PLOTS_DIR}/13_training_curves.png")

# Optuna history placeholder (simple flat line since we used fixed params)
fig, ax = plt.subplots(figsize=(8, 3))
ax.axhline(best_val_auc, color="#2E86AB", lw=2,
           label=f"Best Val AUC = {best_val_auc:.4f}")
ax.set_xlabel("Trial (fixed params used)"); ax.set_ylabel("Val AUC")
ax.set_title("HPO: Fixed Best Parameters (No Optuna Search)")
ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/11_optuna_history.png", dpi=120, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE")
print(f"  Best Test AUC : {best_test_auc:.4f}")
print(f"  Parameters    : {total_params:,}")
print("=" * 60)
