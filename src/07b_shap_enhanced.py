import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
torch.set_num_threads(2)
import gc

"""
================================================================
STEP 07b -- SHAP EXPLAINABILITY (Enhanced 70-Feature Model)
================================================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from tabm_ple_model import TabMPLE

warnings.filterwarnings('ignore')

RESULTS_DIR = 'results'
PLOTS_DIR   = 'results/plots'
MODELS_DIR  = 'models'
DATASET_DIR = 'dataset'
os.makedirs(PLOTS_DIR, exist_ok=True)

print('=' * 60)
print('  STEP 07b -- SHAP ANALYSIS (70 Features + PLE)')
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

# Load data & feature names
num_idx  = cfg['num_idx']
bin_idx  = cfg['bin_idx']
X_test   = np.load(f'{DATASET_DIR}/X_test_enh.npy').astype(np.float32)
y_test   = np.load(f'{DATASET_DIR}/y_test_enh.npy').astype(int)
feat_names = np.load(f'{DATASET_DIR}/feature_names_enhanced.npy', allow_pickle=True).tolist()

# Load feature metadata for nice labels
try:
    with open(f'{DATASET_DIR}/feature_metadata.json') as f:
        feat_meta = json.load(f)
    feat_labels = [feat_meta.get(n, n) for n in feat_names]
except:
    feat_labels = feat_names

# SHAP wrapper function
def predict_proba(X_flat):
    X_flat = X_flat.astype(np.float32)
    xn = torch.tensor(X_flat[:, num_idx])
    xb = torch.tensor(X_flat[:, bin_idx])
    with torch.no_grad():
        p = torch.sigmoid(model(xn, xb)).numpy()
    return p

# Use background + explain samples (keep small for speed)
bg_size  = 150
ex_size  = 200
bg_idx   = np.random.choice(len(X_test), bg_size, replace=False)
ex_idx   = np.random.choice(len(X_test), ex_size, replace=False)

X_bg = X_test[bg_idx]
X_ex = X_test[ex_idx]
y_ex = y_test[ex_idx]

print(f'  Background: {X_bg.shape}  |  Explain: {X_ex.shape}')
print('  Running SHAP KernelExplainer (may take 3-5 minutes)...')

explainer   = shap.KernelExplainer(predict_proba, X_bg)
shap_values = explainer.shap_values(X_ex, nsamples=150)

# Save
np.save(f'{RESULTS_DIR}/shap_values_enhanced.npy', shap_values)
np.save(f'{RESULTS_DIR}/shap_X_enhanced.npy', X_ex)
print(f'  [OK] SHAP values saved')

# ── Global feature importance ──
mean_abs = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature' : feat_names,
    'Label'   : feat_labels,
    'Mean|SHAP|': mean_abs,
    'Rank'    : pd.Series(mean_abs).rank(ascending=False).astype(int),
}).sort_values('Mean|SHAP|', ascending=False).reset_index(drop=True)
importance_df['Rank'] = range(1, len(importance_df)+1)

print(f'\n{"─"*55}')
print(f'  TOP 20 FEATURES BY SHAP IMPORTANCE')
print(f'{"─"*55}')
for _, row in importance_df.head(20).iterrows():
    bar = '█' * int(row['Mean|SHAP|'] / importance_df['Mean|SHAP|'].max() * 20)
    print(f'  {int(row["Rank"]):>2}. {row["Feature"]:<20} {row["Mean|SHAP|"]:.4f}  {bar}')

importance_df.to_csv(f'{RESULTS_DIR}/shap_feature_importance_enhanced.csv', index=False)

# ── Plot 1: Beeswarm (top 20 features) ──
top20_idx = importance_df.head(20)['Feature'].apply(
    lambda f: feat_names.index(f)).tolist()
top20_labels = [feat_labels[i] for i in top20_idx]

fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(
    shap_values[:, top20_idx],
    X_ex[:, top20_idx],
    feature_names=top20_labels,
    show=False, plot_type='dot', max_display=20,
)
plt.title('SHAP Beeswarm — Top 20 Features\nTabM-PLE (70 Features)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/34_enh_shap_beeswarm.png', dpi=130, bbox_inches='tight')
plt.close(); gc.collect()
print(f'  [OK] Saved -> {PLOTS_DIR}/34_enh_shap_beeswarm.png')

# ── Plot 2: Bar (top 20) ──
fig, ax = plt.subplots(figsize=(10, 8))
top20_imp = importance_df.head(20)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 20))
bars = ax.barh(range(20), top20_imp['Mean|SHAP|'].values[::-1],
               color=colors, edgecolor='white', linewidth=0.8)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_imp['Label'].values[::-1], fontsize=9)
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('Top 20 Features — SHAP Global Importance\nTabM-PLE (70 Features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/35_enh_shap_bar.png', dpi=130, bbox_inches='tight')
plt.close(); gc.collect()
print(f'  [OK] Saved -> {PLOTS_DIR}/35_enh_shap_bar.png')

# ── Plot 3: Waterfall for an ASD sample ──
asd_indices = np.where(y_ex == 1)[0]
if len(asd_indices) > 0:
    idx = asd_indices[0]
    exp = shap.Explanation(
        values=shap_values[idx, top20_idx],
        base_values=explainer.expected_value,
        data=X_ex[idx, top20_idx],
        feature_names=top20_labels,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(exp, show=False, max_display=15)
    plt.title('SHAP Waterfall — ASD Sample\nTabM-PLE (70 Features)',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/36_enh_shap_waterfall_asd.png',
                dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Saved -> {PLOTS_DIR}/36_enh_shap_waterfall_asd.png')

gc.collect()
print('\n' + '=' * 60)
print('  SHAP ANALYSIS COMPLETE')
print(f'  Top Feature: {importance_df.iloc[0]["Label"]}')
print(f'  2nd Feature: {importance_df.iloc[1]["Label"]}')
print(f'  3rd Feature: {importance_df.iloc[2]["Label"]}')
print('=' * 60)
