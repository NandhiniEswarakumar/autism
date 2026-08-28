import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 1 -- EXPLORATORY DATA ANALYSIS (EDA)
Autism Spectrum Disorder Detection Using TabM & SHAP
================================================================
Dataset  : NSCH 2023 (cleaned_asd_dataset.csv)
Target   : K2Q35A  (1 = ASD, 0 = No ASD)
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DATASET_PATH  = "dataset/cleaned_asd_dataset.csv"
PLOTS_DIR     = "results/plots"
TARGET_COL    = "K2Q35A"

os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# STYLE SETTINGS
# -------------------------------------------------------------
PALETTE_MAIN  = ["#2E86AB", "#E84855"]   # Blue  = No ASD | Red = ASD
PALETTE_FEATS = "viridis"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                      "axes.titleweight": "bold"})

# -------------------------------------------------------------
# FEATURE DESCRIPTIONS  (for readable axis labels)
# -------------------------------------------------------------
FEATURE_NAMES = {
    "SC_AGE_YEARS" : "Age (Years)",
    "SC_SEX"       : "Sex (1=Male, 0=Female)",
    "ALLERGIES"    : "Allergies",
    "DIABETES"     : "Diabetes",
    "HEART"        : "Heart Disease",
    "DOWNSYN"      : "Down Syndrome",
    "CYSTFIB"      : "Cystic Fibrosis",
    "K2Q35A"       : "ASD Diagnosis",
}

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
print("=" * 60)
print("  ASD PROJECT -- EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n[OK]  Dataset loaded  ->  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

# -------------------------------------------------------------
# 2. BASIC INFO
# -------------------------------------------------------------
print("\n------------------------------------------")
print("  COLUMN SUMMARY")
print("------------------------------------------")
info = pd.DataFrame({
    "Feature"    : [FEATURE_NAMES.get(c, c) for c in df.columns],
    "Column"     : df.columns,
    "Dtype"      : df.dtypes.values,
    "Non-Null"   : df.notnull().sum().values,
    "Missing"    : df.isnull().sum().values,
    "Miss%"      : (df.isnull().mean() * 100).round(2).values,
    "Unique"     : df.nunique().values,
})
print(info.to_string(index=False))

# -------------------------------------------------------------
# 3. STATISTICAL SUMMARY
# -------------------------------------------------------------
print("\n------------------------------------------")
print("  STATISTICAL SUMMARY")
print("------------------------------------------")
print(df.describe().round(4).to_string())

# -------------------------------------------------------------
# 4. CLASS DISTRIBUTION
# -------------------------------------------------------------
class_counts  = df[TARGET_COL].value_counts().sort_index()
class_pct     = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
labels        = ["No ASD (0)", "ASD (1)"]

print("\n------------------------------------------")
print("  CLASS DISTRIBUTION")
print("------------------------------------------")
for idx, (cnt, pct) in enumerate(zip(class_counts, class_pct)):
    print(f"  {labels[idx]}  :  {cnt:,}  ({pct:.2f}%)")
print(f"  Imbalance Ratio (ASD/NoASD) = {class_counts[1]/class_counts[0]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ASD Class Distribution", fontsize=16, fontweight="bold", y=1.02)

# Bar chart
bars = axes[0].bar(labels, class_counts.values, color=PALETTE_MAIN,
                   edgecolor="white", linewidth=1.5, width=0.5)
axes[0].set_title("Sample Count per Class")
axes[0].set_ylabel("Number of Records")
for bar, cnt, pct in zip(bars, class_counts.values, class_pct.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50, f"{cnt:,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    class_counts.values, labels=labels, colors=PALETTE_MAIN,
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=2))
for at in autotexts:
    at.set_fontsize(13); at.set_fontweight("bold")
axes[1].set_title("Class Proportion")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [OK]  Saved -> {PLOTS_DIR}/01_class_distribution.png")

# -------------------------------------------------------------
# 5. FEATURE DISTRIBUTIONS
# -------------------------------------------------------------
features = [c for c in df.columns if c != TARGET_COL]
binary_feats  = [c for c in features if df[c].nunique() <= 2]
numeric_feats = [c for c in features if df[c].nunique() >  2]

# Numerical: histogram + KDE split by class
for feat in numeric_feats:
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, color, lbl in zip([0, 1], PALETTE_MAIN, labels):
        sub = df[df[TARGET_COL] == cls][feat].dropna()
        ax.hist(sub, bins=30, alpha=0.55, color=color,
                density=True, label=lbl, edgecolor="white")
        # KDE overlay
        kde_x = np.linspace(sub.min(), sub.max(), 300)
        kde   = stats.gaussian_kde(sub)
        ax.plot(kde_x, kde(kde_x), color=color, lw=2.5)
    ax.set_title(f"Distribution of {FEATURE_NAMES.get(feat, feat)} by ASD Status")
    ax.set_xlabel(FEATURE_NAMES.get(feat, feat))
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_dist_{feat}.png", dpi=130, bbox_inches="tight")
    plt.close()

# Binary: grouped bar counts
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for i, feat in enumerate(binary_feats):
    ct = df.groupby([feat, TARGET_COL]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[i], color=PALETTE_MAIN,
            edgecolor="white", linewidth=1.2, width=0.65)
    axes[i].set_title(f"{FEATURE_NAMES.get(feat, feat)} vs ASD")
    axes[i].set_xlabel(FEATURE_NAMES.get(feat, feat))
    axes[i].set_ylabel("Count")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
    axes[i].legend(["No ASD", "ASD"])
# Remove unused subplots
for j in range(len(binary_feats), len(axes)):
    fig.delaxes(axes[j])
plt.suptitle("Binary Feature vs ASD Status", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_binary_features.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/02_dist_*.png & 03_binary_features.png")

# -------------------------------------------------------------
# 6. MISSING VALUE HEATMAP
# -------------------------------------------------------------
missing = df.isnull().sum()
if missing.sum() > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_missing_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [OK]  Saved -> {PLOTS_DIR}/04_missing_heatmap.png")
else:
    print("  [OK]  No missing values detected -- skipping heatmap")

# -------------------------------------------------------------
# 7. CORRELATION HEATMAP
# -------------------------------------------------------------
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            xticklabels=[FEATURE_NAMES.get(c, c) for c in corr.columns],
            yticklabels=[FEATURE_NAMES.get(c, c) for c in corr.index])
ax.set_title("Pearson Correlation Matrix (Lower Triangle)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK]  Saved -> {PLOTS_DIR}/05_correlation_heatmap.png")

# Print top correlations with target
target_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
print("\n------------------------------------------")
print("  FEATURE CORRELATION WITH TARGET (ABS)")
print("------------------------------------------")
for feat, val in target_corr.items():
    print(f"  {FEATURE_NAMES.get(feat, feat):<30}  r = {val:.4f}")

# -------------------------------------------------------------
# 8. OUTLIER ANALYSIS -- IQR BOXPLOTS
# -------------------------------------------------------------
if numeric_feats:
    fig, axes = plt.subplots(1, len(numeric_feats), figsize=(5 * len(numeric_feats), 5))
    if len(numeric_feats) == 1:
        axes = [axes]
    for ax, feat in zip(axes, numeric_feats):
        data_per_class = [df[df[TARGET_COL] == cls][feat].dropna().values
                          for cls in [0, 1]]
        bp = ax.boxplot(data_per_class, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5))
        for patch, color in zip(bp["boxes"], PALETTE_MAIN):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticklabels(labels)
        ax.set_title(f"{FEATURE_NAMES.get(feat, feat)}")
        ax.set_ylabel("Value")
        # IQR outlier count
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[feat] < Q1 - 1.5 * IQR) | (df[feat] > Q3 + 1.5 * IQR)).sum()
        ax.set_xlabel(f"IQR Outliers: {outliers}", fontsize=10, color="gray")
    plt.suptitle("Boxplot: Numerical Features by ASD Status", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_boxplots.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [OK]  Saved -> {PLOTS_DIR}/06_boxplots.png")

# -------------------------------------------------------------
# 9. AGE DISTRIBUTION -- DETAILED
# -------------------------------------------------------------
if "SC_AGE_YEARS" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Violin plot
    parts = axes[0].violinplot(
        [df[df[TARGET_COL] == cls]["SC_AGE_YEARS"].dropna() for cls in [0, 1]],
        positions=[0, 1], showmedians=True)
    for i, (body, color) in enumerate(zip(parts["bodies"], PALETTE_MAIN)):
        body.set_facecolor(color); body.set_alpha(0.75)
    parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2.5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Age (Years)"); axes[0].set_title("Age Distribution by Class (Violin)")

    # Age group bar
    df["age_group"] = pd.cut(df["SC_AGE_YEARS"],
                              bins=[0, 5, 10, 14, 17, 100],
                              labels=["0-5", "6-10", "11-14", "15-17", "18+"])
    age_ct = df.groupby(["age_group", TARGET_COL]).size().unstack(fill_value=0)
    age_ct.plot(kind="bar", ax=axes[1], color=PALETTE_MAIN,
                edgecolor="white", linewidth=1.2)
    axes[1].set_title("ASD Prevalence by Age Group")
    axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(["No ASD", "ASD"])
    df.drop(columns=["age_group"], inplace=True)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_age_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [OK]  Saved -> {PLOTS_DIR}/07_age_analysis.png")

# -------------------------------------------------------------
# 10. DUPLICATE RECORDS
# -------------------------------------------------------------
dup_count = df.duplicated().sum()
print(f"\n------------------------------------------")
print(f"  DUPLICATE RECORDS : {dup_count:,}")
print(f"------------------------------------------")

# -------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("  EDA COMPLETE -- All plots saved to results/plots/")
print("=" * 60)
print(f"  Total records     : {df.shape[0]:,}")
print(f"  Total features    : {df.shape[1] - 1}")
print(f"  ASD cases         : {class_counts[1]:,} ({class_pct[1]:.2f}%)")
print(f"  Non-ASD cases     : {class_counts[0]:,} ({class_pct[0]:.2f}%)")
print(f"  Duplicate rows    : {dup_count:,}")
print(f"  Missing values    : {df.isnull().sum().sum():,}")
print("=" * 60)


