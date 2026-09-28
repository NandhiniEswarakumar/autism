import io
import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")
from tabm_model import TabM
from tabm_ple_model import TabMPLE

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

TARGET = "K2Q35A"
SEED = 42
NUMERICAL_NAMES = {
    "SC_AGE_YEARS", "A1_GRADE", "HHCOUNT", "FAMCOUNT", "FPL_I1",
    "BIRTHWT", "SC_RACE_R", "SC_HISPANIC_R", "HHLANGUAGE",
}


def historical_splits():
    raw = subprocess.run(
        ["git", "show", "HEAD:dataset/asd_enhanced_dataset.csv"],
        check=True, capture_output=True,
    ).stdout
    df = pd.read_csv(io.BytesIO(raw))
    features = [column for column in df.columns if column != TARGET]
    if len(features) != 70:
        raise ValueError(f"Expected 70 features, found {len(features)}")
    num_idx = [i for i, name in enumerate(features) if name in NUMERICAL_NAMES]
    bin_idx = [i for i in range(70) if i not in num_idx]
    X = df[features].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=int)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.125, random_state=SEED, stratify=y_tv,
    )
    scaler = StandardScaler()
    X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_val[:, num_idx] = scaler.transform(X_val[:, num_idx])
    X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])
    X_train, y_train = SMOTE(sampling_strategy=0.5, random_state=SEED).fit_resample(X_train, y_train)
    print(f"70 features: {len(num_idx)} numerical + {len(bin_idx)} binary")
    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, num_idx, bin_idx


def best_threshold(y_true, probabilities):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 981), 0.50])
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def metrics_row(model, label, threshold, y_true, probabilities):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Model": model, "Threshold": label, "ThresholdValue": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Specificity": tn / (tn + fp), "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "MCC": matthews_corrcoef(y_true, predictions),
        "CohenKappa": cohen_kappa_score(y_true, predictions),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
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


def train_tabm(X_train, y_train, X_val, y_val):
    torch.manual_seed(SEED)
    model = TabM.build(n_features=70, hidden_dim=128, n_layers=3, k=16, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=1024, shuffle=True, num_workers=0,
    )
    best_auc, best_state, stale = -1.0, None, 0
    for _ in range(60):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(features), labels).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.tensor(X_val))).numpy()
        auc = roc_auc_score(y_val, val_probs)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    return model


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, num_idx, bin_idx = historical_splits()
    print("Training plain TabM...")
    model = train_tabm(X_train, y_train, X_val, y_val)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.tensor(X_val))).numpy()
        test_probs = torch.sigmoid(model(torch.tensor(X_test))).numpy()
    threshold = best_threshold(y_val, val_probs)
    rows = [
        metrics_row("TabM", "0.50", 0.50, y_test, test_probs),
        metrics_row("TabM", "Validation-best-F1", threshold, y_test, test_probs),
    ]
    table = pd.DataFrame(rows)
    numeric = [
        "ThresholdValue", "Accuracy", "Precision", "Recall", "Specificity",
        "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa",
    ]
    table[numeric] = table[numeric].round(4)
    table.to_csv("results/comparison_tabm_70_features.csv", index=False)
    print("\nPLAIN TABM 70-FEATURE RESULTS")
    print(table.to_string(index=False))
    print("\nSaved results/comparison_tabm_70_features.csv")

    baseline_files = [
        "results/comparison_boosting_70_features.csv",
        "results/comparison_neural_70_features.csv",
        "results/comparison_tabm_70_features.csv",
    ]
    baseline = pd.concat(
        [pd.read_csv(path).query("Threshold == 'Validation-best-F1'") for path in baseline_files],
        ignore_index=True,
    )
    with open("results/metrics_enhanced.json") as handle:
        paper = json.load(handle)
    paper_row = {
        "Model": "TabM-PLE (paper)", "Threshold": "Paper threshold", "ThresholdValue": paper["Classification_Threshold"],
        "Accuracy": paper["Accuracy"], "Precision": paper["Precision"], "Recall": paper["Recall(Sens)"],
        "Specificity": paper["Specificity"], "F1": paper["F1-Score"], "ROC-AUC": paper["ROC-AUC"],
        "PR-AUC": paper["PR-AUC"], "MCC": paper["MCC"], "CohenKappa": paper["Cohen_Kappa"],
    }
    combined_columns = [
        "Model", "Threshold", "ThresholdValue", "Accuracy", "Precision", "Recall",
        "Specificity", "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa",
    ]
    combined = pd.concat([baseline[combined_columns], pd.DataFrame([paper_row])], ignore_index=True)
    print("\nCOMBINED 70-FEATURE COMPARISON")
    print(markdown_table(combined[combined_columns]))


if __name__ == "__main__":
    main()