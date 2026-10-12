import requests
import streamlit as st

API_URL = "http://127.0.0.1:5000/predict"

st.title("Diabetes Risk Prediction")
st.write("Enter the patient details below and click **Predict**.")

# Two columns to keep the form tidy
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (diastolic)", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm, 0 if unknown)", min_value=0.0, max_value=100.0, value=0.0)

with col2:
    insulin = st.number_input("Insulin (0 if unknown)", min_value=0.0, max_value=900.0, value=0.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4)
    age = st.number_input("Age", min_value=0, max_value=100, value=35)

if st.button("Predict"):
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    try:
        response = requests.post(API_URL, json=data, timeout=10)
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure 'python app.py' is running.")
    else:
        if response.status_code == 200:
            st.write(f"Probability of diabetes: **{result['probability'] * 100:.1f}%**")
            if result["prediction"] == 1:
                st.error(result["label"])
            else:
                st.success(result["label"])
            st.caption("This is a screening aid, not a medical diagnosis.")
        else:
            # The API rejected the input (e.g. glucose out of range)
            st.warning(result["error"])