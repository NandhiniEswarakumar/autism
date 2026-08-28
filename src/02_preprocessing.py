import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 2 -- FULL DATA PREPROCESSING PIPELINE
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
Steps:
  1. Load cleaned dataset
  2. Remove duplicates
  3. Handle missing values (median/mode imputation)
  4. Binary encoding of Yes/No columns
  5. StandardScaler for numerical features
  6. SMOTE oversampling for class imbalance
  7. Save preprocessor -> models/preprocessor.pkl
  8. Save final arrays -> dataset/
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DATASET_PATH   = "dataset/cleaned_asd_dataset.csv"
MODELS_DIR     = "models"
PLOTS_DIR      = "results/plots"
TARGET_COL     = "K2Q35A"
TEST_SIZE      = 0.20
VAL_SIZE       = 0.10      # fraction of training set used as validation
RANDOM_STATE   = 42
SMOTE_RATIO    = 0.5       # after SMOTE, minority / majority = 0.5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

print("=" * 60)
print("  PREPROCESSING PIPELINE")
print("=" * 60)

# -------------------------------------------------------------
# 1. LOAD
# -------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
print(f"\n[1] Loaded  ->  {df.shape[0]:,} rows  x  {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# 2. NOTE ON DUPLICATES
# ─────────────────────────────────────────────────────────────
# NSCH survey data has only 7 mostly-binary features, so most
# rows share the same feature combination. These are NOT data
# entry errors — they represent different children who happen
# to have the same clinical profile. We keep all rows.
before = len(df)
after  = len(df)
print(f"[2] Duplicate check  ->  {df.duplicated().sum():,} rows share feature patterns"
      f" (kept — valid survey respondents, not data errors)")

# -------------------------------------------------------------
# 3. MISSING VALUES
# -------------------------------------------------------------
missing_before = df.isnull().sum().sum()
numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols       = df.select_dtypes(exclude=[np.number]).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   Filled {col} NaN  ->  median = {median_val:.4f}")

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"   Filled {col} NaN  ->  mode = {mode_val}")

missing_after = df.isnull().sum().sum()
print(f"[3] Missing values  ->  before={missing_before}, after={missing_after}")

# -------------------------------------------------------------
# 4. FEATURE / TARGET SPLIT
# -------------------------------------------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

feature_names = X.columns.tolist()
print(f"[4] Features  : {feature_names}")
print(f"    Target    : {TARGET_COL}  ->  classes {y.unique()}")

# -------------------------------------------------------------
# 5. IDENTIFY COLUMN TYPES
# -------------------------------------------------------------
binary_cols  = [c for c in feature_names if X[c].nunique() == 2]
numeric_feat = [c for c in feature_names if X[c].nunique() >  2]
print(f"[5] Numeric features : {numeric_feat}")
print(f"    Binary features  : {binary_cols}")

# -------------------------------------------------------------
# 6. TRAIN / VAL / TEST SPLIT  (before scaling to avoid leakage)
# -------------------------------------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=RANDOM_STATE, stratify=y_train_full)

print(f"\n[6] Split:")
print(f"    Train  : {X_train.shape[0]:,} rows  |  ASD={y_train.sum():,}")
print(f"    Val    : {X_val.shape[0]:,} rows  |  ASD={y_val.sum():,}")
print(f"    Test   : {X_test.shape[0]:,} rows  |  ASD={y_test.sum():,}")

# -------------------------------------------------------------
# 7. STANDARD SCALING  (fit on train only)
# -------------------------------------------------------------
scaler = StandardScaler()

def scale_split(X_df, fit=False):
    X_np = X_df.copy()
    if numeric_feat:
        if fit:
            X_np[numeric_feat] = scaler.fit_transform(X_np[numeric_feat])
        else:
            X_np[numeric_feat] = scaler.transform(X_np[numeric_feat])
    return X_np.values.astype(np.float32)

X_train_sc = scale_split(X_train, fit=True)
X_val_sc   = scale_split(X_val)
X_test_sc  = scale_split(X_test)

print(f"[7] Scaling done  ->  scaler fitted on training set only")

# -------------------------------------------------------------
# 8. SMOTE  (applied only on training data)
# -------------------------------------------------------------
class_before = np.bincount(y_train.values.astype(int))
current_ratio = class_before[1] / class_before[0]
print(f"\n[8] Before SMOTE  ->  NoASD={class_before[0]:,}  ASD={class_before[1]:,}"
      f"  ratio={current_ratio:.4f}")

# Only apply SMOTE if minority class is genuinely rare (< 40%)
if current_ratio < 0.4:
    smote = SMOTE(sampling_strategy=min(SMOTE_RATIO, 0.8), random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train.values)
    class_after = np.bincount(y_train_res.astype(int))
    print(f"    After SMOTE   ->  NoASD={class_after[0]:,}  ASD={class_after[1]:,}"
          f"  ratio={class_after[1]/class_after[0]:.4f}")
else:
    X_train_res, y_train_res = X_train_sc, y_train.values
    class_after = class_before
    print(f"    SMOTE skipped  ->  ratio already {current_ratio:.4f} (>= 0.40)")

# -------------------------------------------------------------
# 9. VISUALIZE CLASS BALANCE (before vs after SMOTE)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
PALETTE = ["#2E86AB", "#E84855"]

for ax, counts, title in zip(
    axes,
    [class_before, class_after],
    ["Before SMOTE (Train)", "After SMOTE (Train)"]):
    bars = ax.bar(["No ASD", "ASD"], counts, color=PALETTE,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{cnt:,}", ha="center", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13); ax.set_ylabel("Count")
    total = counts.sum()
    ax.set_ylim(0, counts.max() * 1.15)

plt.suptitle("Class Balance: Before vs After SMOTE", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_smote_balance.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"    [OK]  Saved -> {PLOTS_DIR}/08_smote_balance.png")

# -------------------------------------------------------------
# 10. SAVE PREPROCESSOR & ARRAYS
# -------------------------------------------------------------
preprocessor = {
    "scaler"        : scaler,
    "numeric_feat"  : numeric_feat,
    "binary_cols"   : binary_cols,
    "feature_names" : feature_names,
    "target_col"    : TARGET_COL,
}
joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")
print(f"\n[9]  Saved preprocessor -> {MODELS_DIR}/preprocessor.pkl")

# Save numpy arrays
np.save("dataset/X_train_scaled.npy",  X_train_res)
np.save("dataset/y_train_smote.npy",   y_train_res)
np.save("dataset/X_val_scaled.npy",    X_val_sc)
np.save("dataset/y_val.npy",           y_val.values)
np.save("dataset/X_test_scaled.npy",   X_test_sc)
np.save("dataset/y_test_arr.npy",      y_test.values)
np.save("dataset/feature_names.npy",   np.array(feature_names))

# Also save original (unscaled) test for interpretability
X_test.to_csv("dataset/X_test_original.csv", index=False)
y_test.to_csv("dataset/y_test_original.csv",  index=False)

print("[10] Saved numpy arrays -> dataset/")
print("\n" + "=" * 60)
print(f"  PREPROCESSING COMPLETE")
print(f"  Final training set  : {X_train_res.shape[0]:,} x {X_train_res.shape[1]}")
print(f"  Validation set      : {X_val_sc.shape[0]:,} x {X_val_sc.shape[1]}")
print(f"  Test set            : {X_test_sc.shape[0]:,} x {X_test_sc.shape[1]}")
print("=" * 60)


