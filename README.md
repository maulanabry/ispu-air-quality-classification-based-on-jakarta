# ISPU Air Quality Classification App

End-to-end machine learning application for classifying air quality (ISPU) based on Jakarta air pollution data from 2021–2022. The model is trained in Google Colab and deployed as an interactive web application using Streamlit.

---

## 📌 Project Overview

Air quality is an important environmental and public health issue in Jakarta. This project aims to classify air quality status using the **ISPU (Indeks Standar Pencemar Udara)** categories based on several pollutant parameters.

The application allows users to input pollutant values and instantly obtain the predicted air quality category.

---

## 📊 Dataset

- **Source**: Jakarta Air Pollution Dataset
- **Period**: 2021–2022
- **Features**:

  - PM10
  - PM2.5
  - SO2
  - CO
  - O3
  - NO2

- **Target**: ISPU Category

  - 1 = Tidak Sehat
  - 2 = Sedang
  - 3 = Baik

Data cleaning steps include:

- Removing irrelevant columns
- Handling missing values using median imputation
- Encoding categorical labels

---

## 🧠 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Preprocessing**:

  - StandardScaler

- **Pipeline**: Scikit-learn Pipeline (Scaler + Model)
- **Training Environment**: Google Colab

### Model Performance

- **Accuracy**: ~99.6%

The high accuracy is expected as ISPU classification follows rule-based thresholds that are well captured by tree-based models.

---

## 🚀 Deployment

The trained model is serialized using `joblib` and deployed using **Streamlit Community Cloud**.

### Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## ▶️ How to Run Locally

```bash
# create virtual environment
python -m venv venv

# activate venv (Windows - Git Bash)
source venv/Scripts/activate

# install dependencies
pip install -r requirements.txt

# run app
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py
├── ispu_pipeline.pkl
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🌐 Live Demo

Deployed using Streamlit Community Cloud.

> [Streamlit app URL](https://ispu-air-quality-classification-based-on-jakarta-zun57srh4l4oc.streamlit.app/)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Maulana Bryan Syahputra**

## 🙏 Acknowledgements

This project was inspired by an earlier academic group project developed together with:

- Rendi Panca Wijanarko
- Syauqillah Hadie Ahsana

---

## 💡 Notes

This project was originally inspired by a group assignment in a Data Mining course.
The current version has been **independently redesigned and rebuilt** as a personal portfolio project.

All model training, pipeline construction, and Streamlit deployment in this repository were implemented independently to demonstrate an end-to-end machine learning workflow.
