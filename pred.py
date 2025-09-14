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

# --- Input Fields ---

# Age (blank by default)
age = st.text_input("Age")
age = int(age) if age else None

# Sex dropdown
sex = st.selectbox("Sex", ["Select", "Male", "Female"])
if sex == "Select":
    sex = None
else:
    sex = 1 if sex == "Male" else 0

# Chest Pain Type
cp = st.selectbox("Chest Pain Type", ["Select", "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_mapping.get(cp, None)

# Resting Blood Pressure
trestbps = st.text_input("Resting Blood Pressure (trestbps)")
trestbps = int(trestbps) if trestbps else None

# Cholesterol
chol = st.text_input("Serum Cholesterol (chol)")
chol = int(chol) if chol else None

# Fasting Blood Sugar
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Select", "No", "Yes"])
if fbs == "Select":
    fbs = None
else:
    fbs = 1 if fbs == "Yes" else 0

# Resting ECG
restecg = st.selectbox("Resting ECG", ["Select", "Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
restecg_mapping = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_mapping.get(restecg, None)

# Maximum Heart Rate
thalach = st.text_input("Maximum Heart Rate Achieved (thalach)")
thalach = int(thalach) if thalach else None

# Exercise Induced Angina
exang = st.selectbox("Exercise Induced Angina", ["Select", "Yes", "No"])
if exang == "Select":
    exang = None
else:
    exang = 1 if exang == "Yes" else 0

# Oldpeak
oldpeak = st.selectbox("ST Depression (oldpeak)", [
    "Select",
    "Low (0.0 - 1.0)",
    "Moderate (1.1 - 2.0)",
    "High (2.1 - 3.0)"
])
oldpeak_mapping = {"Low (0.0 - 1.0)": 0.5, "Moderate (1.1 - 2.0)": 1.5, "High (2.1 - 3.0)": 2.5}
oldpeak = oldpeak_mapping.get(oldpeak, None)

# Slope
slope = st.selectbox("Slope of ST Segment", ["Select", "Upsloping", "Flat", "Downsloping"])
slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_mapping.get(slope, None)

# --- Prediction ---
if st.button("Predict"):
    # Check if any field is None
    if None in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]:
        st.warning("⚠️ Please fill in all required fields before predicting.")
    else:
        # Create DataFrame
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
