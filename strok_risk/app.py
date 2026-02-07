# -----------------------------
# app.py - Stroke Risk Prediction Tool
# -----------------------------
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load saved model, scaler, encoders
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
features = ['gender', 'age', 'hypertension', 'heart_disease', 
            'ever_married', 'work_type', 'Residence_type', 
            'avg_glucose_level', 'bmi', 'smoking_status']

# -----------------------------
# Streamlit layout
# -----------------------------
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Stroke Risk Prediction Tool")
st.markdown("Enter patient information below to predict the risk of stroke.")

# -----------------------------
# Patient input form
# -----------------------------
with st.form("patient_form"):
    st.subheader("Patient Information")
    
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=65)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","Never_worked","children"])
    residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50, max_value=300, value=105)
    bmi = st.number_input("BMI", min_value=10, max_value=50, value=28)
    smoking_status = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])
    
    submitted = st.form_submit_button("Predict Stroke Risk")

# -----------------------------
# Prediction logic
# -----------------------------
if submitted:
    # Build DataFrame from user input
    new_patient = pd.DataFrame({
        'gender':[gender],
        'age':[age],
        'hypertension':[0 if hypertension=="No" else 1],
        'heart_disease':[0 if heart_disease=="No" else 1],
        'ever_married':[ever_married],
        'work_type':[work_type],
        'Residence_type':[residence_type],
        'avg_glucose_level':[avg_glucose_level],
        'bmi':[bmi],
        'smoking_status':[smoking_status]
    })

    # Encode categorical columns
    for col in categorical_cols:
        le = encoders[col]
        new_patient[col] = le.transform(new_patient[col])

    # Scale numeric columns
    new_patient[numeric_cols] = scaler.transform(new_patient[numeric_cols])

    # Ensure columns match training features
    new_patient = new_patient[features]

    # Predict stroke risk
    risk = rf_model.predict(new_patient)[0]
    prob = rf_model.predict_proba(new_patient)[0][1]

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Stroke Risk:** {'Yes' if risk==1 else 'No'}")
    st.write(f"**Probability of Stroke:** {prob*100:.2f}%")

    # Feature importance chart
    st.subheader("Top Risk Factors")
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Note:** This tool is for educational purposes only and does not replace medical advice.")
