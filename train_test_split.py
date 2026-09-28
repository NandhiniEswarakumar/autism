import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_asd_dataset.csv")

# Features
X = df.drop("K2Q35A", axis=1)

# Target
y = df["K2Q35A"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)
print("X_test shape :", X_test.shape, "| length:", len(X_test))
print("y_test shape :", y_test.shape, "| length:", len(y_test))

# Save split datasets to CSV files
X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)
y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("\nDataset Split Successfully!")