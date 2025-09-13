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

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0  # Convert to 0/1 for model

cp = st.selectbox("Chest Pain Type", [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
])
# Convert to numeric
cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, 
              "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_mapping[cp]

trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholesterol (chol)", value=200)

thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", value=150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0, format="%.1f")

# Prediction
if st.button("Predict"):
    # Add dummy values for removed columns (fbs, restecg, slope)
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, 0, 0, thalach, exang, oldpeak, 0]],
                              columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope'])
    
    # Apply scalers to only the columns that need scaling
    input_data[['age','trestbps','thalach']] = standard_scaler.transform(input_data[['age','trestbps','thalach']])
    input_data[['chol','oldpeak']] = robust_scaler.transform(input_data[['chol','oldpeak']])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    if prediction == 0:
        st.success("✅ The person is Healthy (No Heart Disease)")
    else:
        st.error("⚠️ The person has Heart Disease")
