import pandas as pd

# Load selected dataset
df = pd.read_csv("dataset/asd_selected_dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\n============================")
print("Missing Values")
print("============================")
print(df.isnull().sum())

print("\n============================")
print("Duplicate Rows")
print("============================")
print(df.duplicated().sum())

print("\n============================")
print("Data Types")
print("============================")
print(df.dtypes)

print("\n============================")
print("Statistical Summary")
print("============================")
print(df.describe())