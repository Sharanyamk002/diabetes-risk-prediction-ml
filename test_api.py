import requests

URL = "http://127.0.0.1:5000/predict"

good = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80,
        "SkinThickness": 30, "Insulin": 100, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.6, "Age": 50}
print("Valid input  :", requests.post(URL, json=good).json())

bad = dict(good, Glucose=-5)
print("Invalid input:", requests.post(URL, json=bad).json())

missing = {"Glucose": 120}
print("Missing field:", requests.post(URL, json=missing).json())