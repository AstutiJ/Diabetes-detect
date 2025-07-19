# 🩺 Diabetes Prediction App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Model-orange)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

![App Screenshot](images/diabetes_app_screenshot.png)

A **Diabetes Prediction Web App** built using **Streamlit** and a **K-Nearest Neighbors (KNN)** machine learning model.  
The app predicts whether a person is likely to have diabetes based on key health parameters like glucose level, BMI, blood pressure, etc.

---

## 🚀 Live Demo
🔗 **[Click here to try the app!](https://diabetes-detect-go53oqbpmc5vzxhxsz6php.streamlit.app/)**

---

## ✨ Features
- **Real-time prediction** using a trained ML model.
- **Interactive UI** with sliders and numeric input fields.
- **Lightweight and Fast** – powered by Streamlit.
- **Sample Data Testing** – easily test predictions.

---

## 📊 Dataset
The app is trained on the **PIMA Indians Diabetes Dataset**, containing features such as:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Age

---

## ⚙️ Installation & Running Locally
Follow these steps to set up the project on your local machine:

```bash
# Clone this repository
git clone https://github.com/AstutiJ/Diabetes-detect.git

# Navigate into the project
cd Diabetes-detect

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
