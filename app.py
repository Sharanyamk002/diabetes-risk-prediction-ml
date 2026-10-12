import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model/best_model.joblib")

# Feature names (same order as training) and allowed (min, max) values
LIMITS = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 300),
    "BloodPressure": (20, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (10, 70),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (15, 100),
}
OPTIONAL = ["SkinThickness", "Insulin"]   # may be left out


def error(message):
    return jsonify({"error": message}), 400


@app.route("/")
def home():
    return jsonify({"message": "Diabetes prediction API is running. Use POST /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return error("Please send valid JSON data")

    values = []
    for name, (low, high) in LIMITS.items():
        value = data.get(name)

        # Missing value
        if value is None or value == "":
            if name in OPTIONAL:
                values.append(np.nan)   # model will fill it in
                continue
            return error(name + " is required")

        # Must be a number
        try:
            value = float(value)
        except (TypeError, ValueError):
            return error(name + " must be a number")

        # 0 for optional fields means "not measured"
        if name in OPTIONAL and value == 0:
            values.append(np.nan)
            continue

        # Must be inside the allowed range
        if value < low or value > high:
            return error(f"{name} must be between {low} and {high}")

        values.append(value)

    df = pd.DataFrame([values], columns=list(LIMITS.keys()))
    probability = float(model.predict_proba(df)[0][1])
    result = 1 if probability >= 0.5 else 0

    return jsonify({
        "prediction": result,
        "label": "High diabetes risk" if result == 1 else "Low diabetes risk",
        "probability": round(probability, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)