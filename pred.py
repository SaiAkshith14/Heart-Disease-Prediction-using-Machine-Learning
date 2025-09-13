import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and scalers
model = pickle.load(open("model40.pkl", "rb"))
standard_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
robust_scaler = pickle.load(open("robust_scaler.pkl", "rb"))

# Title
st.title("❤️ Heart Disease Prediction App")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)

# Sex dropdown - Male first
sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
])
cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, 
              "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_mapping[cp]

trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholesterol (chol)", value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
restecg_mapping = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_mapping[restecg]

thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", value=150)

# Exercise Induced Angina - Yes first
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.selectbox("ST Depression (oldpeak)", [
    "Low (0.0 - 1.0)",
    "Moderate (1.1 - 2.0)",
    "High (2.1 - 3.0)"
])
oldpeak_mapping = {"Low (0.0 - 1.0)": 0.5, "Moderate (1.1 - 2.0)": 1.5, "High (2.1 - 3.0)": 2.5}
oldpeak = oldpeak_mapping[oldpeak]

slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_mapping[slope]

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]],
                              columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope'])
    
    # Apply scalers
    input_data[['age','trestbps','thalach']] = standard_scaler.transform(input_data[['age','trestbps','thalach']])
    input_data[['chol','oldpeak']] = robust_scaler.transform(input_data[['chol','oldpeak']])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    if prediction == 0:
        st.success("✅ The person is Healthy (No Heart Disease)")
    else:
        st.error("⚠️ The person has Heart Disease")
