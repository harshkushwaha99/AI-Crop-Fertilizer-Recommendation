import joblib
import pandas as pd

# Load model
model = joblib.load("models/fertilizer_model.pkl")

# Load encoders
soil_encoder = joblib.load("models/soil_encoder.pkl")
crop_encoder = joblib.load("models/fertilizer_crop_encoder.pkl")
fertilizer_encoder = joblib.load("models/fertilizer_encoder.pkl")

# Example farmer input
soil_type = "Clayey"
crop_type = "Paddy"

data = pd.DataFrame([{
    "Temparature": 28,
    "Humidity ": 54,
    "Moisture": 46,
    "Soil Type": soil_encoder.transform([soil_type])[0],
    "Crop Type": crop_encoder.transform([crop_type])[0],
    "Nitrogen": 35,
    "Potassium": 0,
    "Phosphorous": 0
}])

# Predict
prediction = model.predict(data)

# Convert prediction back to fertilizer name
fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

print("Recommended Fertilizer:", fertilizer)