import io
import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from tabm_ple_model import TabMPLE

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

TARGET = "K2Q35A"
SEED = 42
DATASET_BLOB = "HEAD:dataset/asd_enhanced_dataset.csv"
NUMERICAL_NAMES = {
    "SC_AGE_YEARS", "A1_GRADE", "HHCOUNT", "FAMCOUNT", "FPL_I1",
    "BIRTHWT", "SC_RACE_R", "SC_HISPANIC_R", "HHLANGUAGE",
}


def load_historical_splits():
    raw = subprocess.run(
        ["git", "show", DATASET_BLOB], check=True, capture_output=True
    ).stdout
    df = pd.read_csv(io.BytesIO(raw))
    features = [column for column in df.columns if column != TARGET]
    if len(features) != 70:
        raise ValueError(f"Expected 70 historical features, found {len(features)}")
    num_idx = [i for i, name in enumerate(features) if name in NUMERICAL_NAMES]
    bin_idx = [i for i in range(70) if i not in num_idx]
    X = df[features].to_numpy(dtype=np.float32)
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
    X_train, y_train = SMOTE(
        sampling_strategy=0.5, random_state=SEED
    ).fit_resample(X_train, y_train)
    print(
        f"Historical 70-feature data: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, num_idx, bin_idx


def best_f1_threshold(y_true, probabilities):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 981), 0.50])
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(model_name, threshold, y_true, probabilities):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Model": model_name,
        "Threshold": "Validation-best-F1",
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


def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def main():
    _, _, X_val, y_val, X_test, y_test, num_idx, bin_idx = load_historical_splits()
    with open("results/model_config_enhanced.json") as handle:
        config = json.load(handle)
    model = TabMPLE.build(
        n_num=config["n_num"], n_bin=config["n_bin"],
        n_bins=config["n_bins"], hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"], k=config["k"], dropout=0.0,
    )
    model.load_state_dict(torch.load("models/tabm_ple_best.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(
            model(torch.tensor(X_val[:, num_idx]), torch.tensor(X_val[:, bin_idx]))
        ).numpy()
        test_probs = torch.sigmoid(
            model(torch.tensor(X_test[:, num_idx]), torch.tensor(X_test[:, bin_idx]))
        ).numpy()

    threshold = best_f1_threshold(y_val, val_probs)
    result = pd.DataFrame([
        evaluate("TabM-PLE", threshold, y_test, test_probs)
    ])
    metric_columns = [
        "ThresholdValue", "Accuracy", "Precision", "Recall", "Specificity",
        "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa",
    ]
    result[metric_columns] = result[metric_columns].round(4)
    result.to_csv("results/tabm_ple_best_f1.csv", index=False)
    print("\nTABM-PLE VALIDATION-BEST-F1 RESULT")
    print(result.to_string(index=False))
    print("\nSaved results/tabm_ple_best_f1.csv")

    baseline_paths = [
        "results/comparison_boosting_70_features.csv",
        "results/comparison_neural_70_features.csv",
        "results/comparison_tabm_70_features.csv",
    ]
    baseline = pd.concat(
        [
            pd.read_csv(path).query("Threshold == 'Validation-best-F1'")
            for path in baseline_paths
        ],
        ignore_index=True,
    )
    ple = result.copy()
    combined_columns = [
        "Model", "Threshold", "ThresholdValue", "Accuracy", "Precision", "Recall",
        "Specificity", "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa",
    ]
    combined = pd.concat(
        [baseline[combined_columns], ple[combined_columns]], ignore_index=True
    )
    print("\nFULL 7-ROW VALIDATION-BEST-F1 COMPARISON")
    print(markdown_table(combined[combined_columns]))


if __name__ == "__main__":
    main()