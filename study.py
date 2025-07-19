import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load("salary_model_compressed.zip")

# Page title
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns *>50K* or *<=50K* based on features.")

# Input fields
age = st.slider("Age", 18, 70, 30)
education_num = st.slider("Education Number (1 = Least Educated, 16 = Most Educated)", 1, 16, 10)
hours_per_week = st.slider("Hours Worked per Week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", 0, 100000, step=100)
capital_loss = st.number_input("Capital Loss", 0, 5000, step=50)

# Dropdowns for categorical data (simplified encoding used)
gender = st.selectbox("Gender", ["Male", "Female"])
relationship = st.selectbox("Relationship", ["Husband", "Wife", "Own-child", "Unmarried", "Other"])

# Encode categorical inputs manually (or use label encoder if available)
gender_encoded = 1 if gender == "Male" else 0
relationship_encoded = {"Husband": 0, "Wife": 1, "Own-child": 2, "Unmarried": 3, "Other": 4}[relationship]

# Combine inputs
features = np.array([[age, education_num, capital_gain, capital_loss, hours_per_week, gender_encoded, relationship_encoded]])

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(features)
    result = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"🧠 Predicted Salary: {result}")
