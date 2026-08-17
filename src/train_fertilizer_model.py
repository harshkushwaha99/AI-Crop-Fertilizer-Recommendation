import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data/fertilizer_prediction.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# Encode categorical columns
# -----------------------------

soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = fertilizer_encoder.fit_transform(
    df["Fertilizer Name"]
)

# -----------------------------
# Features
# -----------------------------

X = df[
    [
        "Temparature",
        "Humidity ",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous"
    ]
]

# Target
y = df["Fertilizer Name"]

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# Random Forest Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nFertilizer Model Accuracy:",
      round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=fertilizer_encoder.classes_
    )
)

# -----------------------------
# Save Model + Encoders
# -----------------------------

joblib.dump(model, "models/fertilizer_model.pkl")

joblib.dump(
    soil_encoder,
    "models/soil_encoder.pkl"
)

joblib.dump(
    crop_encoder,
    "models/fertilizer_crop_encoder.pkl"
)

joblib.dump(
    fertilizer_encoder,
    "models/fertilizer_encoder.pkl"
)

print("\nFertilizer model saved successfully!")