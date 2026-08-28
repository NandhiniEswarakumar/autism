import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
torch.set_num_threads(2)

"""
================================================================
STEP 02b -- ENHANCED PREPROCESSING (70 Features)
Autism Spectrum Disorder Detection Using TabM-PLE & SHAP
================================================================
  - Splits into Train/Val/Test (70/10/20)
  - Identifies numerical vs binary features
  - StandardScaler on numerical features only
  - SMOTE on training set (with ratio guard)
  - PLE boundaries fitted on training numerical features
  - Saves all arrays and metadata
================================================================
"""

import os, json, warnings, gc
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

DATASET_DIR = 'dataset'
INPUT_CSV   = os.path.join(DATASET_DIR, 'asd_enhanced_dataset.csv')
RESULTS_DIR = 'results'
PLOTS_DIR   = 'results/plots'
RANDOM_SEED = 42
TARGET      = 'K2Q35A'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

print('=' * 60)
print('  STEP 02b -- ENHANCED PREPROCESSING')
print('=' * 60)

# Load
df = pd.read_csv(INPUT_CSV)
print(f'  Dataset: {df.shape}')

feat_cols = [c for c in df.columns if c != TARGET]
X = df[feat_cols].values.astype(np.float32)
y = df[TARGET].values.astype(int)

print(f'  Features : {len(feat_cols)}')
print(f'  Samples  : {len(X)}')
print(f'  ASD      : {y.sum()} ({y.mean()*100:.2f}%)')

# Identify numerical vs binary features
NUMERICAL_NAMES = ['SC_AGE_YEARS', 'A1_GRADE', 'HHCOUNT', 'FAMCOUNT',
                    'FPL_I1', 'BIRTHWT', 'SC_RACE_R', 'SC_HISPANIC_R', 'HHLANGUAGE']
num_idx = [i for i, c in enumerate(feat_cols) if c in NUMERICAL_NAMES]
bin_idx = [i for i in range(len(feat_cols)) if i not in num_idx]

print(f'  Numerical features : {len(num_idx)}')
print(f'  Binary features    : {len(bin_idx)}')

# Save indices
np.save(os.path.join(DATASET_DIR, 'num_feature_idx.npy'), np.array(num_idx))
np.save(os.path.join(DATASET_DIR, 'bin_feature_idx.npy'), np.array(bin_idx))
np.save(os.path.join(DATASET_DIR, 'feature_names_enhanced.npy'), np.array(feat_cols))

# Train/Val/Test split
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.125, random_state=RANDOM_SEED, stratify=y_tv)

print(f'\n  Train : {X_train.shape}')
print(f'  Val   : {X_val.shape}')
print(f'  Test  : {X_test.shape}')

# Scale numerical features only
scaler = StandardScaler()
X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
X_val[:, num_idx]   = scaler.transform(X_val[:, num_idx])
X_test[:, num_idx]  = scaler.transform(X_test[:, num_idx])

# SMOTE
ratio = (y_train == 1).sum() / max((y_train == 0).sum(), 1)
print(f'\n  Class ratio before SMOTE: {ratio:.4f}')
if ratio < 0.4:
    sm = SMOTE(sampling_strategy=0.5, random_state=RANDOM_SEED)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f'  After SMOTE: {X_train.shape}, ASD={y_train.sum()} ({y_train.mean()*100:.1f}%)')
else:
    print('  SMOTE skipped (ratio already balanced)')

# Save arrays
for name, arr in [('X_train_enh', X_train), ('y_train_enh', y_train),
                  ('X_val_enh', X_val),   ('y_val_enh', y_val),
                  ('X_test_enh', X_test), ('y_test_enh', y_test)]:
    np.save(os.path.join(DATASET_DIR, f'{name}.npy'), arr)

# Save preprocessor
prep_data = {
    'scaler'   : scaler,
    'features' : feat_cols,
    'num_idx'  : num_idx,
    'bin_idx'  : bin_idx,
    'numerical_names': NUMERICAL_NAMES,
}
joblib.dump(prep_data, 'models/enhanced_preprocessor.pkl')

print('\n  [OK] All arrays saved to dataset/')
print('  [OK] Enhanced preprocessor saved to models/')
print('=' * 60)
print('  PREPROCESSING COMPLETE')
print('=' * 60)
