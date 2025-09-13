import streamlit as st
import pickle
import numpy as np

# Load model and scalers
model = pickle.load(open("model40.pkl", "rb"))
standard_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
robust_scaler = pickle.load(open("robust_scaler.pkl", "rb"))

# Title
st.title("❤️ Heart Disease Prediction App")

st.write("Fill in the details below to predict the risk of Heart Disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox("Sex", ["Female", "Male"])
sex_val = 1 if sex == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type",
    ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"]
)
cp_val = int(cp.split(":")[0])

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang_val = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0, format="%.1f")

# Prediction
if st.button("🔍 Predict"):
    # Collect input into numpy array
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, thalach, exang_val, oldpeak]])

    # Apply scalers (match training process)
    input_data[:, [0, 3, 5]] = standard_scaler.transform(input_data[:, [0, 3, 5]])  # age, trestbps, thalach
    input_data[:, [4, 7]] = robust_scaler.transform(input_data[:, [4, 7]])          # chol, oldpeak

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 0:
        st.success("✅ The person is Healthy (No Heart Disease)")
    else:
        st.error("⚠️ The person has Heart Disease")
