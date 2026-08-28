import pandas as pd

# Load dataset
df = pd.read_sas("dataset/nsch_2023e_topical.sas7bdat")

# Select important columns
selected_columns = [
    "SC_AGE_YEARS",   # Age
    "SC_SEX",         # Gender
    "ALLERGIES",      # Allergies
    "DIABETES",       # Diabetes
    "HEART",          # Heart Disease
    "HEART_BORN",     # Congenital Heart Disease
    "DOWNSYN",        # Down Syndrome
    "CYSTFIB",        # Cystic Fibrosis
    "K2Q35A"          # Autism (Target)
]

selected_df = df[selected_columns]

print(selected_df.head())

print("\nShape of Selected Dataset:")
print(selected_df.shape)

# Save new dataset
selected_df.to_csv("dataset/asd_selected_dataset.csv", index=False)

print("\nDataset saved successfully!")