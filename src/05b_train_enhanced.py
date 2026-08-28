import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import gc

"""
================================================================
STEP 05b -- TRAIN TabM-PLE (Enhanced 70-Feature Model)
Autism Spectrum Disorder Detection
================================================================
Model: TabM-PLE (Piecewise Linear Encoding + BatchEnsemble)
  - PLE: 50 bins per numerical feature
  - Hidden: 256 units, 3 layers, k=16 heads
  - Training: 60 epochs, early stop patience=10
  - Loss: BCEWithLogitsLoss + class weights
================================================================
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys = __import__('sys')
sys.path.insert(0, os.path.dirname(__file__))
from tabm_ple_model import TabMPLE

warnings.filterwarnings('ignore')

MODELS_DIR  = 'models'
RESULTS_DIR = 'results'
PLOTS_DIR   = 'results/plots'
DATASET_DIR = 'dataset'
RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print('=' * 60)
print('  TabM-PLE ENHANCED TRAINING')
print('  CPU Threads: 2  |  PLE Bins: 50  |  Heads: 16')
print('=' * 60)

# Load data
print('\n  Loading data...')
X_train = np.load(f'{DATASET_DIR}/X_train_enh.npy').astype(np.float32)
y_train = np.load(f'{DATASET_DIR}/y_train_enh.npy').astype(np.float32)
X_val   = np.load(f'{DATASET_DIR}/X_val_enh.npy').astype(np.float32)
y_val   = np.load(f'{DATASET_DIR}/y_val_enh.npy').astype(np.float32)
X_test  = np.load(f'{DATASET_DIR}/X_test_enh.npy').astype(np.float32)
y_test  = np.load(f'{DATASET_DIR}/y_test_enh.npy').astype(np.float32)

num_idx = np.load(f'{DATASET_DIR}/num_feature_idx.npy').tolist()
bin_idx = np.load(f'{DATASET_DIR}/bin_feature_idx.npy').tolist()

n_num = len(num_idx)
n_bin = len(bin_idx)
print(f'  Train  : {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}')
print(f'  Num features: {n_num}  |  Bin features: {n_bin}')

# Fit PLE on training numerical data
print('\n  Fitting PLE encoding on training data...')
model = TabMPLE.build(
    n_num=n_num, n_bin=n_bin,
    n_bins=50, hidden_dim=256,
    n_layers=3, k=16, dropout=0.2
)
model.ple.fit(X_train[:, num_idx])
print(f'  PLE fitted. Model params: {model.count_params():,}')

# Class weight
n_pos = max((y_train == 1).sum(), 1)
n_neg = max((y_train == 0).sum(), 1)
pw    = torch.tensor(n_neg / n_pos, dtype=torch.float32)
print(f'  Class weight (pos): {pw.item():.2f}')

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

# Prepare tensors
def split_tensor(X, num_idx, bin_idx):
    return (torch.tensor(X[:, num_idx], dtype=torch.float32),
            torch.tensor(X[:, bin_idx], dtype=torch.float32))

X_tr_num, X_tr_bin = split_tensor(X_train, num_idx, bin_idx)
X_va_num, X_va_bin = split_tensor(X_val,   num_idx, bin_idx)
X_te_num, X_te_bin = split_tensor(X_test,  num_idx, bin_idx)
y_tr_t = torch.tensor(y_train)
y_va_t = torch.tensor(y_val)
y_te_t = torch.tensor(y_test)

ldr = DataLoader(
    TensorDataset(X_tr_num, X_tr_bin, y_tr_t),
    batch_size=1024, shuffle=True, num_workers=0
)

# Training loop
print(f'\n  Training for 60 epochs (early stop patience=10)...')
print(f'  {"Epoch":<8} {"Loss":<12} {"Val AUC":<12} {"Test AUC":<12} {"Status"}')
print('  ' + '-' * 55)

train_losses, val_aucs, test_aucs = [], [], []
best_val_auc  = 0.0
best_test_auc = 0.0
patience_cnt  = 0
PATIENCE      = 10

for epoch in range(1, 61):
    model.train()
    ep_loss = 0.0
    for xn, xb, yb in ldr:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xn, xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item() * len(yb)
    ep_loss /= len(ldr.dataset)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(model(X_va_num, X_va_bin)).numpy()
        tp = torch.sigmoid(model(X_te_num, X_te_bin)).numpy()
    try:
        va_auc = roc_auc_score(y_val, vp)
        te_auc = roc_auc_score(y_test, tp)
    except:
        va_auc = te_auc = 0.5

    train_losses.append(ep_loss)
    val_aucs.append(va_auc)
    test_aucs.append(te_auc)

    if va_auc > best_val_auc + 1e-4:
        best_val_auc  = va_auc
        best_test_auc = te_auc
        patience_cnt  = 0
        torch.save(model.state_dict(), f'{MODELS_DIR}/tabm_ple_best.pth')
        status = '[SAVED]'
    else:
        patience_cnt += 1
        status = f'(patience {patience_cnt}/{PATIENCE})'

    if epoch % 5 == 0 or epoch == 1:
        print(f'  {epoch:<8} {ep_loss:<12.4f} {va_auc:<12.4f} {te_auc:<12.4f} {status}')
    if patience_cnt >= PATIENCE:
        print(f'\n  Early stopping at epoch {epoch}')
        break
    gc.collect()

print(f'\n  Best Val AUC  : {best_val_auc:.4f}')
print(f'  Best Test AUC : {best_test_auc:.4f}')

# Save config
cfg = {
    'n_num': n_num, 'n_bin': n_bin,
    'n_bins': 50, 'hidden_dim': 256,
    'n_layers': 3, 'k': 16, 'dropout': 0.2,
    'best_val_auc': round(best_val_auc, 4),
    'best_test_auc': round(best_test_auc, 4),
    'num_idx': num_idx, 'bin_idx': bin_idx,
}
with open(f'{RESULTS_DIR}/model_config_enhanced.json', 'w') as f:
    json.dump(cfg, f, indent=2)

# Plot
ep_range = range(1, len(train_losses)+1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(ep_range, train_losses, color='#2E86AB', lw=2)
axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
axes[1].plot(ep_range, val_aucs,  color='#2EB872', lw=2, label='Val AUC')
axes[1].plot(ep_range, test_aucs, color='#E84855', lw=2, linestyle='--', label='Test AUC')
axes[1].axhline(best_test_auc, color='#F4A261', linestyle=':', lw=1.5,
                label=f'Best={best_test_auc:.4f}')
axes[1].set_title('AUC Curves'); axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.suptitle('TabM-PLE Enhanced Training Curves', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/30_tabmple_training_curves.png', dpi=130, bbox_inches='tight')
plt.close()
gc.collect()

print(f'  [OK] Plot saved -> {PLOTS_DIR}/30_tabmple_training_curves.png')
print(f'  [OK] Config saved -> {RESULTS_DIR}/model_config_enhanced.json')
print('\n' + '=' * 60)
print(f'  TRAINING COMPLETE  |  Best AUC = {best_test_auc:.4f}')
print('=' * 60)
