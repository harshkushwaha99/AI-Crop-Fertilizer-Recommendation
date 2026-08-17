import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/crop_model.pkl")

# Test input
data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200
}])

# Predict crop
prediction = model.predict(data)

print("Recommended Crop:", prediction[0])