import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
torch.set_num_threads(2)
import gc

"""
================================================================
STEP 06b -- EVALUATE ENHANCED TabM-PLE MODEL
================================================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, matthews_corrcoef,
    cohen_kappa_score, brier_score_loss,
)
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from tabm_ple_model import TabMPLE

warnings.filterwarnings('ignore')

RESULTS_DIR = 'results'
PLOTS_DIR   = 'results/plots'
MODELS_DIR  = 'models'
DATASET_DIR = 'dataset'
os.makedirs(PLOTS_DIR, exist_ok=True)

print('=' * 60)
print('  ENHANCED MODEL EVALUATION (TabM-PLE, 70 Features)')
print('=' * 60)

# Load config & model
with open(f'{RESULTS_DIR}/model_config_enhanced.json') as f:
    cfg = json.load(f)

model = TabMPLE.build(
    n_num=cfg['n_num'], n_bin=cfg['n_bin'],
    n_bins=cfg['n_bins'], hidden_dim=cfg['hidden_dim'],
    n_layers=cfg['n_layers'], k=cfg['k'], dropout=0.0,
)
model.load_state_dict(torch.load(f'{MODELS_DIR}/tabm_ple_best.pth', map_location='cpu'))
model.eval()
print(f'  Model loaded. Params: {model.count_params():,}')

# Load data
num_idx = cfg['num_idx']
bin_idx = cfg['bin_idx']
X_test  = np.load(f'{DATASET_DIR}/X_test_enh.npy').astype(np.float32)
y_test  = np.load(f'{DATASET_DIR}/y_test_enh.npy').astype(int)

X_te_num = torch.tensor(X_test[:, num_idx])
X_te_bin = torch.tensor(X_test[:, bin_idx])

with torch.no_grad():
    probs = torch.sigmoid(model(X_te_num, X_te_bin)).numpy()

# Metrics
threshold = 0.5
y_pred = (probs >= threshold).astype(int)
auc    = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)

metrics = {
    'Accuracy'    : round(accuracy_score(y_test, y_pred), 4),
    'Precision'   : round(precision_score(y_test, y_pred, zero_division=0), 4),
    'Recall(Sens)': round(recall_score(y_test, y_pred, zero_division=0), 4),
    'Specificity' : round(recall_score(y_test, y_pred, pos_label=0, zero_division=0), 4),
    'F1-Score'    : round(f1_score(y_test, y_pred, zero_division=0), 4),
    'ROC-AUC'     : round(auc, 4),
    'PR-AUC'      : round(pr_auc, 4),
    'MCC'         : round(matthews_corrcoef(y_test, y_pred), 4),
    'Cohen_Kappa' : round(cohen_kappa_score(y_test, y_pred), 4),
    'Brier_Score' : round(brier_score_loss(y_test, probs), 4),
}

print(f'\n{"─"*50}')
print(f'  PERFORMANCE METRICS (70 Features + PLE)')
print(f'{"─"*50}')
for k, v in metrics.items():
    bar = '█' * int(v * 20)
    print(f'  {k:<18} {v:<8} {bar}')

print(f'\n  Classification Report:')
print(classification_report(y_test, y_pred,
      target_names=['No ASD', 'ASD'], zero_division=0))

with open(f'{RESULTS_DIR}/metrics_enhanced.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  [OK] Saved -> {RESULTS_DIR}/metrics_enhanced.json')

# Plot 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No ASD','ASD'], yticklabels=['No ASD','ASD'],
            linewidths=1, linecolor='#1E293B')
ax.set_title(f'Confusion Matrix\nTabM-PLE (70 Features)  AUC={auc:.4f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/31_enh_confusion_matrix.png', dpi=130, bbox_inches='tight')
plt.close()

# Plot 2 — ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, lw=3, color='#3B82F6', label=f'TabM-PLE (AUC={auc:.4f})')
ax.plot([0,1],[0,1],'k:',lw=1.5)
ax.fill_between(fpr, tpr, alpha=0.1, color='#3B82F6')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curve — Enhanced TabM-PLE (70 Features)', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/32_enh_roc_curve.png', dpi=130, bbox_inches='tight')
plt.close()

# Plot 3 — PR Curve
prec, rec, _ = precision_recall_curve(y_test, probs)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, lw=3, color='#10B981', label=f'PR-AUC={pr_auc:.4f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — TabM-PLE', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/33_enh_pr_curve.png', dpi=130, bbox_inches='tight')
plt.close()

gc.collect()
print(f'  [OK] Plots saved to {PLOTS_DIR}')
print('\n' + '=' * 60)
print(f'  EVALUATION COMPLETE')
print(f'  ROC-AUC : {auc:.4f}')
print(f'  Recall  : {metrics["Recall(Sens)"]:.4f}')
print(f'  F1      : {metrics["F1-Score"]:.4f}')
print('=' * 60)
