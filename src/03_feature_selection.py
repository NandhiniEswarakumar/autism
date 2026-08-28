import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 3 -- FEATURE SELECTION
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
Methods:
  A. Mutual Information (MI)
  B. SelectKBest (Chi-squared for binary, f_classif for numeric)
  C. Correlation filter (remove highly correlated features)
  D. Fisher Score ranking
Output:
  -> results/plots/09_feature_importance_mi.png
  -> results/plots/10_feature_selection_comparison.png
  -> dataset/selected_features.npy   (list of chosen feature names)
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, chi2, f_classif
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
PLOTS_DIR      = "results/plots"
TARGET_COL     = "K2Q35A"
CORR_THRESHOLD = 0.90      # remove one of a highly-correlated pair
TOP_K          = 7         # final number of features to keep

os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_NAMES_FULL = {
    "SC_AGE_YEARS" : "Age (Years)",
    "SC_SEX"       : "Sex",
    "ALLERGIES"    : "Allergies",
    "DIABETES"     : "Diabetes",
    "HEART"        : "Heart Disease",
    "DOWNSYN"      : "Down Syndrome",
    "CYSTFIB"      : "Cystic Fibrosis",
}

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
print("=" * 60)
print("  FEATURE SELECTION")
print("=" * 60)

df = pd.read_csv("dataset/cleaned_asd_dataset.csv")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)
feature_cols = X.columns.tolist()

print(f"\n  Input features : {feature_cols}")
print(f"  Samples        : {len(df):,}")

# -------------------------------------------------------------
# METHOD A -- MUTUAL INFORMATION
# -------------------------------------------------------------
mi_scores = mutual_info_classif(X, y, discrete_features="auto",
                                 random_state=42)
mi_df = pd.DataFrame({
    "Feature"     : feature_cols,
    "Label"       : [FEATURE_NAMES_FULL.get(c, c) for c in feature_cols],
    "MI_Score"    : mi_scores
}).sort_values("MI_Score", ascending=False)

print("\n------------------------------------------")
print("  MUTUAL INFORMATION SCORES")
print("------------------------------------------")
print(mi_df[["Label", "MI_Score"]].to_string(index=False))

# MI bar plot
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(mi_df)))
bars = ax.barh(mi_df["Label"], mi_df["MI_Score"], color=colors,
               edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, mi_df["MI_Score"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Importance -- Mutual Information", fontsize=13, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_feature_importance_mi.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [OK]  Saved -> {PLOTS_DIR}/09_feature_importance_mi.png")

# -------------------------------------------------------------
# METHOD B -- SelectKBest with f_classif (ANOVA F-test)
# -------------------------------------------------------------
selector_f = SelectKBest(f_classif, k="all")
selector_f.fit(X, y)
f_scores = selector_f.scores_
f_pvals  = selector_f.pvalues_

f_df = pd.DataFrame({
    "Feature"  : feature_cols,
    "Label"    : [FEATURE_NAMES_FULL.get(c, c) for c in feature_cols],
    "F_Score"  : f_scores,
    "P_Value"  : f_pvals,
}).sort_values("F_Score", ascending=False)

print("\n------------------------------------------")
print("  ANOVA F-SCORES")
print("------------------------------------------")
print(f_df[["Label", "F_Score", "P_Value"]].round(4).to_string(index=False))

# -------------------------------------------------------------
# METHOD C -- CORRELATION FILTER
# -------------------------------------------------------------
corr_matrix = X.corr().abs()
upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop     = [col for col in upper_tri.columns
               if any(upper_tri[col] > CORR_THRESHOLD)]

print(f"\n------------------------------------------")
print(f"  CORRELATION FILTER  (threshold={CORR_THRESHOLD})")
print(f"------------------------------------------")
if to_drop:
    print(f"  Features to drop due to high correlation: {to_drop}")
else:
    print("  No highly correlated features found -- all kept")

kept_after_corr = [c for c in feature_cols if c not in to_drop]

# -------------------------------------------------------------
# COMBINED RANKING -- normalise + average scores
# -------------------------------------------------------------
scaler_norm = MinMaxScaler()
mi_norm = scaler_norm.fit_transform(mi_df.set_index("Feature")[["MI_Score"]])
f_norm  = scaler_norm.fit_transform(f_df.set_index("Feature")[["F_Score"]])

combined = pd.DataFrame({
    "Feature"    : feature_cols,
    "Label"      : [FEATURE_NAMES_FULL.get(c, c) for c in feature_cols],
}).set_index("Feature")

for feat, val in zip(mi_df["Feature"], mi_norm.ravel()):
    combined.loc[feat, "MI_norm"] = val
for feat, val in zip(f_df["Feature"], f_norm.ravel()):
    combined.loc[feat, "F_norm"]  = val

combined["Combined_Score"] = (combined["MI_norm"] + combined["F_norm"]) / 2
combined = combined.sort_values("Combined_Score", ascending=False).reset_index()

print("\n------------------------------------------")
print("  COMBINED RANKING (MI + F-Score, normalised)")
print("------------------------------------------")
print(combined[["Feature", "Label", "MI_norm", "F_norm", "Combined_Score"]].round(4).to_string(index=False))

# -------------------------------------------------------------
# FINAL SELECTION
# -------------------------------------------------------------
# Keep top-K from combined ranking, but honour correlation filter
eligible = [f for f in combined["Feature"].tolist() if f in kept_after_corr]
selected_features = eligible[:TOP_K]

print(f"\n------------------------------------------")
print(f"  SELECTED FEATURES  (top {TOP_K})")
print(f"------------------------------------------")
for i, f in enumerate(selected_features, 1):
    score = combined.loc[combined["Feature"] == f, "Combined_Score"].values[0]
    print(f"  {i}. {FEATURE_NAMES_FULL.get(f, f):<25}  score={score:.4f}")

# -------------------------------------------------------------
# COMPARISON PLOT -- MI vs F-Score (normalised)
# -------------------------------------------------------------
plot_df = combined.copy()
plot_df["Label"] = [FEATURE_NAMES_FULL.get(f, f) for f in plot_df["Feature"]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MI
axes[0].barh(plot_df["Label"], plot_df["MI_norm"],
             color="#2E86AB", edgecolor="white"); axes[0].invert_yaxis()
axes[0].set_title("Mutual Information (norm.)"); axes[0].set_xlabel("Score (0-1)")

# F-Score
axes[1].barh(plot_df["Label"], plot_df["F_norm"],
             color="#E84855", edgecolor="white"); axes[1].invert_yaxis()
axes[1].set_title("ANOVA F-Score (norm.)"); axes[1].set_xlabel("Score (0-1)")

# Combined
cmap_colors = ["#2EB872" if f in selected_features else "#AAAAAA"
               for f in plot_df["Feature"]]
axes[2].barh(plot_df["Label"], plot_df["Combined_Score"],
             color=cmap_colors, edgecolor="white"); axes[2].invert_yaxis()
axes[2].set_title("Combined Score (selected = green)"); axes[2].set_xlabel("Score (0-1)")

plt.suptitle("Feature Selection Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_feature_selection_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [OK]  Saved -> {PLOTS_DIR}/10_feature_selection_comparison.png")

# -------------------------------------------------------------
# SAVE SELECTED FEATURE LIST
# -------------------------------------------------------------
np.save("dataset/selected_features.npy", np.array(selected_features))
print(f"  [OK]  Saved -> dataset/selected_features.npy")

# -------------------------------------------------------------
# Re-run preprocessing on selected features and save
# -------------------------------------------------------------
print(f"\n  Re-building train/val/test sets with selected features…")
X_sel = X[selected_features]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sel, y, test_size=0.20, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=0.125, random_state=42, stratify=y_tr)

scaler2 = StandardScaler()
X_tr_sc = scaler2.fit_transform(X_tr).astype(np.float32)
X_va_sc = scaler2.transform(X_va).astype(np.float32)
X_te_sc = scaler2.transform(X_te).astype(np.float32)

smote   = SMOTE(sampling_strategy=0.5, random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_tr_sc, y_tr.values)

print(f"  After SMOTE  ->  Train {X_tr_res.shape}  Val {X_va_sc.shape}  Test {X_te_sc.shape}")

np.save("dataset/X_train_final.npy",  X_tr_res.astype(np.float32))
np.save("dataset/y_train_final.npy",  y_tr_res.astype(np.float32))
np.save("dataset/X_val_final.npy",    X_va_sc)
np.save("dataset/y_val_final.npy",    y_va.values.astype(np.float32))
np.save("dataset/X_test_final.npy",   X_te_sc)
np.save("dataset/y_test_final.npy",   y_te.values.astype(np.float32))
X_te.to_csv("dataset/X_test_selected_original.csv", index=False)

joblib.dump({"scaler": scaler2, "features": selected_features},
            "models/selected_preprocessor.pkl")

print("\n" + "=" * 60)
print("  FEATURE SELECTION COMPLETE")
print(f"  Selected features : {selected_features}")
print("=" * 60)



