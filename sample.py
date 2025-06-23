import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load saved model and preprocessor
model = load_model("heart_disease_model.keras")
scaler = joblib.load("scaler.pkl")
with open("feature_columns.json") as f:
    feature_cols = json.load(f)

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 AI-Powered Heart Disease Risk Predictor")
st.write("Enter patient data to estimate the risk of heart disease.")

# Form for inputs
with st.form("prediction_form"):
    age = st.slider("Age (years)", 0, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
    diastolic = st.number_input("Diastolic BP", min_value=50, max_value=140, value=80)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    alcoholic = st.selectbox("Alcohol Use", ["No", "Yes"])
    sleep = st.selectbox("Poor Sleep", ["No", "Yes"])
    medication_count = st.slider("No. of Medications", 0, 20, 2)
    calories = st.slider("Calories Intake (kcal)", 500, 4000, 2000)
    cholesterol = st.number_input("Cholesterol Intake (mg)", min_value=0, max_value=1000, value=200)
    sodium = st.number_input("Sodium Intake (mg)", min_value=0, max_value=5000, value=1500)
    activity = st.slider("Moderate Activity (mins/day)", 0, 180, 30)
    submit = st.form_submit_button("Predict")

if submit:
    # Map inputs
    gender_val = 1 if gender == "Male" else 2
    smoker_val = 1 if smoker == "Yes" else 0
    alcoholic_val = 1 if alcoholic == "Yes" else 0
    sleep_val = 1 if sleep == "Yes" else 0

    # Construct input DataFrame
    input_dict = {
        'RIDAGEYR': age,
        'RIAGENDR': gender_val,
        'BMXBMI': bmi,
        'BPXSY1': systolic,
        'BPXDI1': diastolic,
        'Smoker': smoker_val,
        'Alcoholic': alcoholic_val,
        'Poor_Sleep': sleep_val,
        'Medication_Count': medication_count,
        'DR1TKCAL': calories,
        'DR1TCHOL': cholesterol,      
        'DR1TSODI': sodium,           
        'PAD615': activity      
    }

    input_df = pd.DataFrame([input_dict])

    # Fill missing feature columns with 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns
    input_df = input_df[feature_cols]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict(input_scaled)[0][0]
    pred = "🛑 High Risk" if prob >= 0.5 else "✅ Low Risk"

    st.subheader("Prediction Result")
    st.metric(label="Heart Disease Risk", value=f"{prob:.2%}", delta=None)
    if prob < 0.5:
        st.success(pred)
    else:
        st.error(pred)
