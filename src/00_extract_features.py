import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 00 -- ENHANCED FEATURE EXTRACTION (70 Features)
Extracts clinically relevant features from NSCH 2023 SAS file
================================================================
Feature Groups:
  A. Demographics          (5 features)
  B. Medical Conditions   (15 features)
  C. Comorbidities         (6 features)
  D. Developmental Milestones (12 features)
  E. Behavioral / Social  (12 features)
  F. School / Learning     (6 features)
  G. Socioeconomic         (7 features)
  H. Birth & Early Life    (4 features)
  I. Adverse Childhood Exp (3 features)
  ─────────────────────────
  TOTAL: ~70 features  →  TARGET: K2Q35A (ASD)
================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DATASET_DIR = "dataset"
SAS_FILE    = os.path.join(DATASET_DIR, "nsch_2023e_topical.sas7bdat")
OUT_CSV     = os.path.join(DATASET_DIR, "asd_enhanced_dataset.csv")

print("=" * 60)
print("  STEP 00 -- ENHANCED FEATURE EXTRACTION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────
TARGET = "K2Q35A"

FEATURES = {
    # ── A. Demographics ──────────────────────────────────────
    "SC_AGE_YEARS"  : "Age (Years)",
    "SC_SEX"        : "Sex (1=Male)",
    "SC_RACE_R"     : "Race",
    "SC_HISPANIC_R" : "Hispanic",
    "A1_GRADE"      : "Parent Education Level",

    # ── B. Medical Conditions ─────────────────────────────────
    "ALLERGIES"     : "Allergies",
    "DIABETES"      : "Diabetes",
    "HEART"         : "Heart Condition",
    "DOWNSYN"       : "Down Syndrome",
    "CYSTFIB"       : "Cystic Fibrosis",
    "BLINDNESS"     : "Blindness",
    "K2Q40A"        : "Epilepsy / Seizures",
    "HEADACHE"      : "Headaches / Migraines",
    "AUTOIMMUNE"    : "Autoimmune Condition",
    "BLOOD"         : "Blood Disorder",
    "FASD"          : "Fetal Alcohol Spectrum",
    "OVERWEIGHT"    : "Overweight",
    "K2Q42A"        : "Brain Injury",
    "BREATHING"     : "Breathing Problem",
    "STOMACH"       : "Stomach / Digestive Problem",

    # ── C. Comorbidities ──────────────────────────────────────
    "K2Q31A"        : "ADHD",
    "K2Q32A"        : "Depression",
    "K2Q33A"        : "Anxiety",
    "K2Q36A"        : "Intellectual Disability",
    "K2Q37A"        : "Tourette Syndrome",
    "K2Q60A"        : "Behavioral / Conduct Problems",

    # ── D. Developmental Milestones ───────────────────────────
    "ONEWORD"       : "Said One Word (by 12 mo)",
    "TWOWORDS"      : "Said Two Words (by 16 mo)",
    "THREEWORDS"    : "Said 3 Word Phrases (by 24 mo)",
    "ASKQUESTION"   : "Asks Questions",
    "ASKQUESTION2"  : "Asks Complex Questions",
    "TELLSTORY"     : "Tells Stories",
    "UNDERSTAND"    : "Understands Instructions",
    "DIRECTIONS"    : "Follows Directions",
    "POINT"         : "Points at Objects",
    "BOUNCEABALL"   : "Can Bounce a Ball",
    "DRAWACIRCLE"   : "Can Draw a Circle",
    "DRAWAPERSON"   : "Can Draw a Person",

    # ── E. Behavioral / Social ────────────────────────────────
    "TEMPER_R"      : "Temper Tantrums",
    "PLAYWELL"      : "Plays Well with Others",
    "DISTRACTED"    : "Easily Distracted",
    "HURTSAD"       : "Feels Hurt or Sad",
    "CALMDOWN_R"    : "Calms Down When Upset",
    "WAITFORTURN"   : "Waits for Turn",
    "HARDWORK"      : "Works Hard on Tasks",
    "SHARETOYS"     : "Shares Toys",
    "MAKEFRIEND"    : "Makes Friends Easily",
    "TALKABOUT"     : "Talks About Thoughts & Feelings",
    "K7Q30"         : "Feels Safe in Neighborhood",
    "K7Q31"         : "People in Neighborhood Help Each Other",

    # ── F. School / Learning ──────────────────────────────────
    "RECOGBEGIN"    : "Recognizes Beginning Sounds",
    "SAMESOUND"     : "Identifies Same Sounds",
    "WRITENAME"     : "Writes Own Name",
    "FOCUSON"       : "Can Focus on Task",
    "READONEDIGIT"  : "Reads One-Digit Numbers",
    "SIMPLEADDITION": "Does Simple Addition",

    # ── G. Socioeconomic ──────────────────────────────────────
    "HHCOUNT"       : "Household Count",
    "FAMCOUNT"      : "Family Count",
    "CURRCOV"       : "Currently Insured",
    "TENURE"        : "Home Ownership",
    "EVERHOMELESS"  : "Ever Homeless",
    "MISSMORTGAGE"  : "Missed Mortgage / Rent",
    "FPL_I1"        : "Federal Poverty Level",

    # ── H. Birth & Early Life ─────────────────────────────────
    "BIRTHWT"       : "Birth Weight Category",
    "BIRTHWT_VL"    : "Very Low Birth Weight",
    "BORNUSA"       : "Born in USA",
    "HHLANGUAGE"    : "Household Language",

    # ── I. Adverse Childhood Experiences ─────────────────────
    "ACE1"          : "ACE: Parent Divorce / Separation",
    "ACE3"          : "ACE: Parent Died",
    "ACE4"          : "ACE: Parent in Jail",
}

ALL_COLS = list(FEATURES.keys()) + [TARGET]

# ─────────────────────────────────────────────────────────────
# READ SAS FILE
# ─────────────────────────────────────────────────────────────
print(f"\n  Reading SAS file: {SAS_FILE}")
print("  (This may take 1-2 minutes...)")

try:
    df_full = pd.read_sas(SAS_FILE, encoding='latin1')
    print(f"  Full dataset: {df_full.shape}")
except Exception as e:
    print(f"  Error reading SAS: {e}")
    raise

# ─────────────────────────────────────────────────────────────
# SELECT & FILTER AVAILABLE COLUMNS
# ─────────────────────────────────────────────────────────────
available = [c for c in ALL_COLS if c in df_full.columns]
missing   = [c for c in ALL_COLS if c not in df_full.columns]
print(f"\n  Requested : {len(ALL_COLS)} columns")
print(f"  Available : {len(available)} columns")
if missing:
    print(f"  Missing   : {missing}")

df = df_full[available].copy()

# ─────────────────────────────────────────────────────────────
# FILTER: Keep only rows where TARGET is 1 or 2 (Yes / No ASD)
# ─────────────────────────────────────────────────────────────
if TARGET in df.columns:
    df = df[df[TARGET].isin([1, 2])].copy()
    df[TARGET] = (df[TARGET] == 1).astype(int)  # 1=ASD, 0=No ASD
    print(f"\n  After target filter: {df.shape}")
    print(f"  ASD cases: {df[TARGET].sum()} ({df[TARGET].mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# NUMERICAL FEATURE TYPES
# ─────────────────────────────────────────────────────────────
NUMERICAL = ["SC_AGE_YEARS", "A1_GRADE", "HHCOUNT", "FAMCOUNT", "FPL_I1",
             "BIRTHWT", "SC_RACE_R", "SC_HISPANIC_R", "HHLANGUAGE"]

for col in df.columns:
    if col == TARGET: continue
    if col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # Binary: recode any value > 1 as 1 (Yes), 0 as 0 (No), NaN as 0
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Most NSCH binary cols: 1=Yes, 2=No -> recode to 1/0
        df[col] = df[col].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

# ─────────────────────────────────────────────────────────────
# HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n  Handling missing values...")
for col in df.columns:
    pct_miss = df[col].isnull().mean()
    if col in NUMERICAL:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(0)  # Missing = No for binary

# ─────────────────────────────────────────────────────────────
# DROP COLUMNS WITH NEAR-ZERO VARIANCE
# ─────────────────────────────────────────────────────────────
drop_cols = []
for col in df.columns:
    if col == TARGET: continue
    if df[col].std() < 0.001:
        drop_cols.append(col)

if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"  Dropped zero-variance cols: {drop_cols}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
df = df.astype(np.float32)
df[TARGET] = df[TARGET].astype(int)
df.to_csv(OUT_CSV, index=False)

feat_cols = [c for c in df.columns if c != TARGET]
print(f"\n  Final feature count : {len(feat_cols)}")
print(f"  Final row count     : {len(df)}")
print(f"  ASD cases           : {df[TARGET].sum()} ({df[TARGET].mean()*100:.2f}%)")
print(f"\n  Features selected:")
for i, c in enumerate(feat_cols, 1):
    label = FEATURES.get(c, c)
    print(f"    {i:2d}. {c:<20} {label}")

print(f"\n  [OK] Saved -> {OUT_CSV}")
print("\n" + "=" * 60)
print("  FEATURE EXTRACTION COMPLETE")
print("=" * 60)

# Save feature metadata
import json
feature_meta = {c: FEATURES.get(c, c) for c in feat_cols}
with open("dataset/feature_metadata.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
print(f"  [OK] Saved -> dataset/feature_metadata.json")
