# 🌱 AI-Based Smart Crop & Fertilizer Recommendation System

An AI/ML-powered agricultural decision-support system that recommends suitable crops and fertilizers based on soil and environmental conditions.

## 🚀 Features

- 🌾 Crop Recommendation
- 🧪 Fertilizer Recommendation
- 🤖 Random Forest Machine Learning Models
- 📊 Soil and environmental parameter analysis
- 🌐 Interactive Streamlit Web Application
- 📈 Model confidence display
- 🔐 Saved trained ML models
- 🖥️ User-friendly dashboard

## 🌾 Crop Recommendation

The crop recommendation model uses the following parameters:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The model predicts one of 22 crop classes.

### Crop Model Result

Test-set accuracy:

**99.55%**

> This accuracy is based on the project's 20% held-out test split and should not be interpreted as guaranteed real-world agricultural accuracy.

## 🧪 Fertilizer Recommendation

The fertilizer model uses:

- Temperature
- Humidity
- Soil Moisture
- Soil Type
- Crop Type
- Nitrogen
- Potassium
- Phosphorous

The model predicts the fertilizer class based on the provided conditions.

### Fertilizer Model Result

Test-set accuracy:

**100%**

> The fertilizer dataset contains only 99 records, so this result should be interpreted cautiously and not as real-world accuracy.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Random Forest

## 📁 Project Structure

```text
AI-Crop-Fertilizer-Recommendation/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   ├── crop_recommendation.csv
│   └── fertilizer_prediction.csv
│
├── models/
│   ├── crop_model.pkl
│   ├── fertilizer_model.pkl
│   ├── soil_encoder.pkl
│   ├── fertilizer_crop_encoder.pkl
│   └── fertilizer_encoder.pkl
│
├── src/
│   ├── check_dataset.py
│   ├── eda.py
│   ├── train_crop_model.py
│   ├── test_crop_prediction.py
│   ├── train_fertilizer_model.py
│   └── test_fertilizer_prediction.py
│
└── venv/