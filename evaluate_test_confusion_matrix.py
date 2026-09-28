import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tabm_ple_model import TabMPLE


RANDOM_SEED = 42
TEST_SIZE = 0.20
THRESHOLD = 0.30

with open(os.path.join(ROOT, "results", "model_config_enhanced.json")) as config_file:
    config = json.load(config_file)

preprocessor = joblib.load(
    os.path.join(ROOT, "models", "enhanced_preprocessor.pkl")
)

model = TabMPLE.build(
    n_num=config["n_num"],
    n_bin=config["n_bin"],
    n_bins=config["n_bins"],
    hidden_dim=config["hidden_dim"],
    n_layers=config["n_layers"],
    k=config["k"],
    dropout=0.0,
)
model.load_state_dict(
    torch.load(
        os.path.join(ROOT, "models", "tabm_ple_best.pth"),
        map_location="cpu",
    )
)
model.eval()

# Recreate the same held-out test partition used by 02b_preprocessing_enhanced.py.
data = pd.read_csv(os.path.join(ROOT, "dataset", "asd_enhanced_dataset.csv"))
X = data[preprocessor["features"]].values.astype(np.float32)
y = data["K2Q35A"].values.astype(int)
_, X_test_raw, _, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

# Apply the saved preprocessing to the test partition only.
X_test = X_test_raw.copy()
num_idx = preprocessor["num_idx"]
bin_idx = preprocessor["bin_idx"]
X_test[:, num_idx] = preprocessor["scaler"].transform(X_test[:, num_idx])

X_test_num = torch.tensor(X_test[:, num_idx], dtype=torch.float32)
X_test_bin = torch.tensor(X_test[:, bin_idx], dtype=torch.float32)

# Generate fresh predictions from the loaded model; no cached predictions are used.
with torch.no_grad():
    probabilities = torch.sigmoid(model(X_test_num, X_test_bin)).numpy()

y_pred = (probabilities >= THRESHOLD).astype(int)
matrix = confusion_matrix(y_test, y_pred)

print(f"Test samples: {len(y_test)}")
print(f"Threshold: {THRESHOLD:.2f}")
print("Fresh confusion matrix [rows=true, columns=predicted]:")
print(matrix)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "Specificity": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities),
    "PR-AUC": average_precision_score(y_test, probabilities),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "Cohen's kappa": cohen_kappa_score(y_test, y_pred),
    "Brier score": brier_score_loss(y_test, probabilities),
}

print("\nMetrics:")
print(f"{'Metric':<18} {'Value':>10}")
print("-" * 29)
for name, value in metrics.items():
    print(f"{name:<18} {value:>10.6f}")