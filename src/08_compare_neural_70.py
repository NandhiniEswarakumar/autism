import io
import subprocess
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, "src")
from tabm_model import TabM

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

TARGET = "K2Q35A"
SEED = 42
NUMERICAL_NAMES = {
    "SC_AGE_YEARS", "A1_GRADE", "HHCOUNT", "FAMCOUNT", "FPL_I1",
    "BIRTHWT", "SC_RACE_R", "SC_HISPANIC_R", "HHLANGUAGE",
}


def load_data():
    raw = subprocess.run(
        ["git", "show", "HEAD:dataset/asd_enhanced_dataset.csv"],
        check=True, capture_output=True,
    ).stdout
    df = pd.read_csv(io.BytesIO(raw))
    features = [column for column in df.columns if column != TARGET]
    if len(features) != 70:
        raise ValueError(f"Expected 70 historical features, found {len(features)}")
    num_idx = [i for i, name in enumerate(features) if name in NUMERICAL_NAMES]
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

    from imblearn.over_sampling import SMOTE
    X_train, y_train = SMOTE(sampling_strategy=0.5, random_state=SEED).fit_resample(X_train, y_train)
    print(f"Historical Git dataset: {len(features)} features, {len(df)} rows")
    print(f"Split: train={len(y_train)} after SMOTE, val={len(y_val)}, test={len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def best_threshold(y_true, probabilities):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 981), 0.5])
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def row(model, label, threshold, y_true, probabilities):
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


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val):
    torch.manual_seed(SEED)
    model = MLP(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=2048, shuffle=True, num_workers=0,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    stale = 0
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
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    return model


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print("Training MLP...")
    mlp = train_mlp(X_train, y_train, X_val, y_val)
    print("Training TabNet...")
    tabnet = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, seed=SEED,
        verbose=0, device_name="cpu",
    )
    tabnet.fit(
        X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)],
        eval_name=["val"], eval_metric=["auc"], max_epochs=80,
        patience=12, batch_size=2048, virtual_batch_size=256,
    )
    with torch.no_grad():
        mlp_val = torch.sigmoid(mlp(torch.tensor(X_val))).numpy()
        mlp_test = torch.sigmoid(mlp(torch.tensor(X_test))).numpy()
    val_probs = {
        "MLP": mlp_val,
        "TabNet": tabnet.predict_proba(X_val)[:, 1],
    }
    test_probs = {
        "MLP": mlp_test,
        "TabNet": tabnet.predict_proba(X_test)[:, 1],
    }
    rows = []
    for name in ("MLP", "TabNet"):
        threshold = best_threshold(y_val, val_probs[name])
        rows.append(row(name, "0.50", 0.50, y_test, test_probs[name]))
        rows.append(row(name, "Validation-best-F1", threshold, y_test, test_probs[name]))
    table = pd.DataFrame(rows)
    metric_columns = [
        "ThresholdValue", "Accuracy", "Precision", "Recall", "Specificity",
        "F1", "ROC-AUC", "PR-AUC", "MCC", "CohenKappa",
    ]
    table[metric_columns] = table[metric_columns].round(4)
    print("\n70-FEATURE NEURAL BASELINES")
    print(table.to_string(index=False))
    table.to_csv("results/comparison_neural_70_features.csv", index=False)
    print("\nSaved results/comparison_neural_70_features.csv")


if __name__ == "__main__":
    main()