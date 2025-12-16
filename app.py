import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Klasifikasi ISPU Jakarta",
    page_icon="🌫️",
    layout="wide"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_pipeline():
    return joblib.load("ispu_pipeline.pkl")

pipeline = load_pipeline()

# ================================
# HEADER
# ================================
st.title("🌫️ Klasifikasi Kualitas Udara (ISPU) Jakarta")
st.markdown(
    """
    Aplikasi ini mengklasifikasikan **Indeks Standar Pencemar Udara (ISPU) di Jakarta**
    berdasarkan parameter polutan.

    Model dilatih menggunakan **Random Forest** dan di-*deploy* menggunakan **Streamlit**.
    """
)

# ================================
# SIDEBAR INPUT
# ================================
st.sidebar.header("🧪 Input Parameter Polutan")

pm10 = st.sidebar.number_input("PM10", min_value=0.0, step=1.0)
pm25 = st.sidebar.number_input("PM2.5", min_value=0.0, step=1.0)
so2 = st.sidebar.number_input("SO2", min_value=0.0, step=1.0)
co = st.sidebar.number_input("CO", min_value=0.0, step=1.0)
o3 = st.sidebar.number_input("O3", min_value=0.0, step=1.0)
no2 = st.sidebar.number_input("NO2", min_value=0.0, step=1.0)

# ================================
# PREDICTION
# ================================
st.markdown("---")

if st.button("🔍 Prediksi ISPU", use_container_width=True):
    input_data = np.array([[pm10, pm25, so2, co, o3, no2]])
    pred = pipeline.predict(input_data)[0]

    label_map = {
        1: ("TIDAK SEHAT", "🔴"),
        2: ("SEDANG", "🟡"),
        3: ("BAIK", "🟢")
    }

    label, emoji = label_map[pred]

    st.subheader("📌 Hasil Prediksi")
    st.markdown(f"## {emoji} **{label}**")

    # Deskripsi kategori
    desc = {
        "BAIK": "Kualitas udara sangat baik dan tidak menimbulkan risiko kesehatan.",
        "SEDANG": "Kualitas udara masih dapat diterima, namun berpotensi berdampak ringan pada kelompok sensitif.",
        "TIDAK SEHAT": "Kualitas udara dapat membahayakan kesehatan masyarakat."
    }

    st.info(desc[label])

# ================================
# FEATURE IMPORTANCE (OPTIONAL)
# ================================
st.markdown("---")
st.subheader("📊 Feature Importance")

model = pipeline.named_steps['model']
importances = model.feature_importances_

features = ['PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2']
fi_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.dataframe(fi_df, use_container_width=True)

# ================================
# FOOTER
# ================================
st.markdown(
    """
    ---
    **Tech Stack**: Python · Streamlit · Scikit-learn  
    **Model**: Random Forest Classifier  
    """
)
