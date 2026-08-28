import pandas as pd

# Load dataset
df = pd.read_csv("dataset/asd_selected_dataset.csv")

print("Original Shape:", df.shape)

# Remove HEART_BORN (too many missing values)
df = df.drop(columns=["HEART_BORN"])

# Remove rows with missing values
df = df.dropna()

print("Shape after Cleaning:", df.shape)

# Convert Yes/No values
columns = [
    "SC_SEX",
    "ALLERGIES",
    "DIABETES",
    "HEART",
    "DOWNSYN",
    "CYSTFIB",
    "K2Q35A"
]

for col in columns:
    df[col] = df[col].replace({
        1: 1,
        2: 0
    })

print("\nFirst 5 Rows")
print(df.head())

# Save cleaned dataset
df.to_csv("dataset/cleaned_asd_dataset.csv", index=False)

print("\nDataset cleaned and saved successfully!")