import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset load
df = pd.read_csv("data/crop_recommendation.csv")

# Basic information
print("Dataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Crop distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="label")
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.drop(columns=["label"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()