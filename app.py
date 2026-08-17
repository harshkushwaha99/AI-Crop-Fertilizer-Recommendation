import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Crop & Fertilizer AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #e8f5e9,
        #f1f8e9
    );
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    color: #555;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    background: #f0fff4;
    border: 1px solid #b7e4c7;
    text-align: center;
    margin-top: 15px;
}

.result-title {
    font-size: 16px;
    color: #555;
}

.result-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}

.section-card {
    padding: 20px;
    border-radius: 15px;
    background: #ffffff;
    border: 1px solid #eeeeee;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 25px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    crop_model = joblib.load(
        "models/crop_model.pkl"
    )

    fertilizer_model = joblib.load(
        "models/fertilizer_model.pkl"
    )

    soil_encoder = joblib.load(
        "models/soil_encoder.pkl"
    )

    fertilizer_crop_encoder = joblib.load(
        "models/fertilizer_crop_encoder.pkl"
    )

    fertilizer_encoder = joblib.load(
        "models/fertilizer_encoder.pkl"
    )

    return (
        crop_model,
        fertilizer_model,
        soil_encoder,
        fertilizer_crop_encoder,
        fertilizer_encoder
    )


(
    crop_model,
    fertilizer_model,
    soil_encoder,
    fertilizer_crop_encoder,
    fertilizer_encoder
) = load_models()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌱 Smart Agriculture")

    st.markdown("---")

    st.subheader("🤖 ML Models")

    st.write("🌾 Crop Recommendation")
    st.write("🧪 Fertilizer Recommendation")

    st.markdown("---")

    st.subheader("🛠️ Technology")

    st.write("Python")
    st.write("Scikit-learn")
    st.write("Random Forest")
    st.write("Pandas")
    st.write("Streamlit")

    st.markdown("---")

    st.info(
        "This system provides ML-based recommendations "
        "for educational and decision-support purposes."
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🌱 AI-Based Smart Crop & Fertilizer Recommendation
</div>

<div class="hero-subtitle">
Machine Learning powered agricultural decision-support system
for crop and fertilizer recommendations.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "🌾 Crop Model",
        "Random Forest"
    )

with m2:
    st.metric(
        "🧪 Fertilizer Model",
        "Random Forest"
    )

with m3:
    st.metric(
        "📊 Crop Classes",
        "22"
    )

with m4:
    st.metric(
        "⚙️ Platform",
        "Streamlit"
    )


st.markdown("---")


# =========================================================
# CROP RECOMMENDATION
# =========================================================

st.header("🌾 Crop Recommendation")

st.write(
    "Enter soil and environmental conditions to predict "
    "the most suitable crop."
)

with st.container(border=True):

    st.subheader("🌱 Soil Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
            step=1.0
        )

    with col2:
        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=42.0,
            step=1.0
        )

    with col3:
        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=200.0,
            value=43.0,
            step=1.0
        )

    st.subheader("🌦️ Environmental Parameters")

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0,
            step=0.5
        )

    with col5:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col6:
        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1
        )

    with col7:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=200.0,
            step=1.0
        )

    crop_button = st.button(
        "🌱 Predict Best Crop",
        type="primary",
        use_container_width=True
    )


# =========================================================
# CROP PREDICTION
# =========================================================

if crop_button:

    crop_input = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    predicted_crop = crop_model.predict(
        crop_input
    )[0]

    crop_probabilities = crop_model.predict_proba(
        crop_input
    )[0]

    crop_confidence = max(
        crop_probabilities
    ) * 100

    st.markdown(f"""
    <div class="result-card">

    <div class="result-title">
    🌾 Recommended Crop
    </div>

    <div class="result-value">
    {predicted_crop.title()}
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.progress(
        int(crop_confidence),
        text=f"Model confidence: {crop_confidence:.2f}%"
    )


st.markdown("---")


# =========================================================
# FERTILIZER RECOMMENDATION
# =========================================================

st.header("🧪 Fertilizer Recommendation")

st.write(
    "Enter soil and crop information to predict a suitable fertilizer."
)

with st.container(border=True):

    st.subheader("🌦️ Environment & Soil")

    col1, col2, col3 = st.columns(3)

    with col1:
        fertilizer_temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=28.0,
            step=0.5,
            key="fert_temperature"
        )

    with col2:
        fertilizer_humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=54.0,
            step=1.0,
            key="fert_humidity"
        )

    with col3:
        moisture = st.number_input(
            "Soil Moisture",
            min_value=0.0,
            max_value=100.0,
            value=46.0,
            step=1.0
        )

    col4, col5 = st.columns(2)

    with col4:
        soil_type = st.selectbox(
            "Soil Type",
            soil_encoder.classes_
        )

    with col5:
        fertilizer_crop = st.selectbox(
            "Crop Type",
            fertilizer_crop_encoder.classes_
        )

    st.subheader("🧪 Nutrient Levels")

    col6, col7, col8 = st.columns(3)

    with col6:
        fertilizer_nitrogen = st.number_input(
            "Nitrogen",
            min_value=0.0,
            max_value=200.0,
            value=35.0,
            step=1.0
        )

    with col7:
        fertilizer_potassium = st.number_input(
            "Potassium",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=1.0
        )

    with col8:
        fertilizer_phosphorous = st.number_input(
            "Phosphorous",
            min_value=0.0,
            max_value=200.0,
            value=0.0,
            step=1.0
        )

    fertilizer_button = st.button(
        "🧪 Predict Fertilizer",
        type="primary",
        use_container_width=True
    )


# =========================================================
# FERTILIZER PREDICTION
# =========================================================

if fertilizer_button:

    encoded_soil = soil_encoder.transform(
        [soil_type]
    )[0]

    encoded_crop = fertilizer_crop_encoder.transform(
        [fertilizer_crop]
    )[0]

    fertilizer_input = pd.DataFrame([{
        "Temparature": fertilizer_temperature,
        "Humidity ": fertilizer_humidity,
        "Moisture": moisture,
        "Soil Type": encoded_soil,
        "Crop Type": encoded_crop,
        "Nitrogen": fertilizer_nitrogen,
        "Potassium": fertilizer_potassium,
        "Phosphorous": fertilizer_phosphorous
    }])

    predicted_fertilizer = fertilizer_model.predict(
        fertilizer_input
    )

    fertilizer_name = fertilizer_encoder.inverse_transform(
        predicted_fertilizer
    )[0]

    fertilizer_probabilities = fertilizer_model.predict_proba(
        fertilizer_input
    )[0]

    fertilizer_confidence = max(
        fertilizer_probabilities
    ) * 100

    st.markdown(f"""
    <div class="result-card">

    <div class="result-title">
    🧪 Recommended Fertilizer
    </div>

    <div class="result-value">
    {fertilizer_name}
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.progress(
        int(fertilizer_confidence),
        text=f"Model confidence: {fertilizer_confidence:.2f}%"
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.markdown("---")

st.header("📖 About This Project")

about1, about2 = st.columns(2)

with about1:

    st.subheader("🌱 Crop Recommendation")

    st.write(
        "The crop model uses soil nutrients and environmental "
        "conditions such as N, P, K, temperature, humidity, "
        "soil pH and rainfall to predict a suitable crop."
    )


with about2:

    st.subheader("🧪 Fertilizer Recommendation")

    st.write(
        "The fertilizer model considers environmental conditions, "
        "soil type, crop type and nutrient levels to recommend "
        "a fertilizer."
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    "⚠️ Disclaimer: Model predictions are based on the training "
    "datasets and should be treated as decision-support information. "
    "For real agricultural use, recommendations should be validated "
    "with local soil tests and agricultural experts."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🌱 AI-Based Smart Crop & Fertilizer Recommendation System

<br>

Built with Python • Pandas • Scikit-learn • Random Forest • Streamlit

</div>
""", unsafe_allow_html=True)