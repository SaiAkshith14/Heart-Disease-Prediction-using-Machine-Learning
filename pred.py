import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scalers
model = pickle.load(open("model40.pkl", "rb"))
standard_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
robust_scaler = pickle.load(open("robust_scaler.pkl", "rb"))

# Title
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to check the risk of heart disease.")

# Input fields (8 features)
age = st.number_input("Age (in years)", min_value=20, max_value=100, value=40)

sex = st.radio("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [
        "0: Typical Angina",
        "1: Atypical Angina",
        "2: Non-anginal Pain",
        "3: Asymptomatic"
    ]
)
cp_val = int(cp.split(":")[0])

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)

exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
exang_val = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

# Prediction
if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[age, sex_val, cp_val, trestbps, chol, thalach, exang_val, oldpeak]],
                              columns=['age','sex','cp','trestbps','chol','thalach','exang','oldpeak'])
    
    # Apply correct scalers
    input_data[['age','trestbps','thalach']] = standard_scaler.transform(input_data[['age','trestbps','thalach']])
    input_data[['chol','oldpeak']] = robust_scaler.transform(input_data[['chol','oldpeak']])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    if prediction == 0:
        st.success("✅ The person is Healthy (No Heart Disease)")
    else:
        st.error("⚠️ The person has Heart Disease")
