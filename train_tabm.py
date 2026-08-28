import pandas as pd
import numpy as np
import torch
from tabm import TabM
from sklearn.metrics import roc_auc_score

# Load your saved splits
X_train = pd.read_csv("dataset/X_train.csv").values.astype(np.float32)
X_test = pd.read_csv("dataset/X_test.csv").values.astype(np.float32)
y_train = pd.read_csv("dataset/y_train.csv").values.ravel().astype(np.float32)
y_test = pd.read_csv("dataset/y_test.csv").values.ravel().astype(np.float32)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

# Build TabM model
model = TabM.make(
    n_num_features=X_train.shape[1],
    cat_cardinalities=[],
    d_out=1,
)

# Loss and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t, None).squeeze(-1)
    loss = loss_fn(preds, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    test_preds = torch.sigmoid(model(X_test_t, None).squeeze(-1))
    auc = roc_auc_score(y_test, test_preds.numpy())
    print("Test AUC:", auc)