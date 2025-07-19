import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("Diabetes_KNN.pkl")

# Page settings
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("This is a Diabetes Prediction Web App built using **Streamlit** and **KNN model**.")

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below and check if they are likely to have diabetes.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input data
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])

    prediction = model.predict(input_data)
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠ The person is likely to have **Diabetes**.")
    else:
        st.success("✅ The person is **NOT likely** to have Diabetes.")

st.markdown("---")
st.caption("Created with ❤️ using Streamlit.")
