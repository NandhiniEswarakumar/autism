import io
import subprocess
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATASET_BLOB = "HEAD:dataset/asd_enhanced_dataset.csv"
TARGET = "K2Q35A"
SEED = 42
NUMERICAL_NAMES = {
    "SC_AGE_YEARS", "A1_GRADE", "HHCOUNT", "FAMCOUNT", "FPL_I1",
    "BIRTHWT", "SC_RACE_R", "SC_HISPANIC_R", "HHLANGUAGE",
}


def load_committed_dataset():
    raw = subprocess.run(
        ["git", "show", DATASET_BLOB], check=True, capture_output=True
    ).stdout
    df = pd.read_csv(io.BytesIO(raw))
    if TARGET not in df or len(df.columns) - 1 != 70:
        raise ValueError(
            f"Committed dataset must contain 70 predictors; got {len(df.columns) - 1}"
        )
    return df


def prepare_data(df):
    feature_names = [column for column in df.columns if column != TARGET]
    num_idx = [i for i, name in enumerate(feature_names) if name in NUMERICAL_NAMES]
    bin_idx = [i for i in range(len(feature_names)) if i not in num_idx]
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=int)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.125, random_state=SEED, stratify=y_tv
    )

    scaler = StandardScaler()
    X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_val[:, num_idx] = scaler.transform(X_val[:, num_idx])
    X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

    before_smote = len(y_train)
    smote = SMOTE(sampling_strategy=0.5, random_state=SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Dataset: {len(feature_names)} features ({len(num_idx)} numerical + {len(bin_idx)} binary)")
    print(f"Rows: {len(df)} | train={before_smote} -> {len(y_train)} after SMOTE | val={len(y_val)} | test={len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def best_f1_threshold(y_true, probabilities):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 981), 0.5])
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def metrics_row(model, label, threshold, y_true, probabilities):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Model": model,
        "Threshold": label,
        "ThresholdValue": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, predictions),
        "CohenKappa": cohen_kappa_score(y_true, predictions),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    df = load_committed_dataset()
    data = prepare_data(df)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    models = {
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, random_seed=SEED,
            eval_metric="AUC", verbose=False, thread_count=2,
            allow_writing_files=False,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
            random_state=SEED, n_jobs=2,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=SEED, verbose=-1, n_jobs=2,
        ),
    }
    validation_probabilities = {}
    test_probabilities = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "CatBoost":
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40)
        elif name == "LightGBM":
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        validation_probabilities[name] = model.predict_proba(X_val)[:, 1]
        test_probabilities[name] = model.predict_proba(X_test)[:, 1]

    rows = []
    for name in models:
        threshold = best_f1_threshold(y_val, validation_probabilities[name])
        rows.append(metrics_row(name, "0.50", 0.50, y_test, test_probabilities[name]))
        rows.append(metrics_row(name, "Validation-best-F1", threshold, y_test, test_probabilities[name]))

    table = pd.DataFrame(rows)
    metric_columns = [
        "ThresholdValue", "Accuracy", "Precision", "Recall", "Specificity",
        "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa",
    ]
    table[metric_columns] = table[metric_columns].round(4)
    if table.groupby("Model")["ROC-AUC"].nunique().max() != 1:
        raise ValueError("ROC-AUC changed between thresholds")

    print("\nFAIR 70-FEATURE BOOSTING BASELINES")
    print(table.drop(columns=["TN", "FP", "FN", "TP"]).to_string(index=False))
    print("\nCONFUSION MATRICES AT VALIDATION-BEST-F1 THRESHOLD")
    print(table[table["Threshold"] == "Validation-best-F1"][
        ["Model", "ThresholdValue", "TN", "FP", "FN", "TP"]
    ].to_string(index=False))
    table.to_csv("results/comparison_boosting_70_features.csv", index=False)
    print("\nSaved results/comparison_boosting_70_features.csv")


if __name__ == "__main__":
    main()