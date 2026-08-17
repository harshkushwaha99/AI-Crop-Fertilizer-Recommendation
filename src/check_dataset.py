import pandas as pd

# Dataset load karo
df = pd.read_csv("data/crop_recommendation.csv")

# Dataset ki basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCrop Distribution:")
print(df["label"].value_counts())