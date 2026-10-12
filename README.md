# Diabetes Risk Prediction (Mini AI Solution)

A small end-to-end machine learning project. It collects real data, cleans it, trains and compares three models, picks the best one, and serves predictions through an **API** and a simple **web page**.

**Author:** YOUR NAME
**Built for:** Pokak Technologies – AI / Data Science Developer machine test

---

## 1. The Problem

Diabetes is a common disease, and many people don't find out until it has already caused damage. Catching it early makes it much easier to treat.

**This project predicts whether a person is likely to have diabetes (1) or not (0)** from 8 simple health measurements, such as glucose level, BMI and age.

**Who it helps:** clinics, health camps and health apps that want a quick first-level screening, so they can decide who should go for a proper lab test.

> ⚠️ This is a **screening aid, not a medical diagnosis**.

---

## 2. The Data

- **Dataset:** Pima Indians Diabetes Dataset (originally from the UCI repository / U.S. National Institute of Diabetes and Digestive and Kidney Diseases).
- **Size:** 768 patients (rows), 8 input features, 1 target column.
- **People in it:** women aged 21 or older of Pima Indian heritage.
- **How it was collected:** `collect_data.py` downloads the CSV file from a public GitHub link using `pandas`, adds the column names, and saves it in `data/raw_data.csv`.
- **Class balance:** about 35% of patients have diabetes and 65% don't.

| Feature | What it means |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Blood sugar level (plasma glucose concentration) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Skin-fold thickness on the arm (mm), an indicator of body fat |
| Insulin | Insulin level in blood (mu U/ml) |
| BMI | Body Mass Index (weight relative to height) |
| DiabetesPedigreeFunction | A score of diabetes in the person's family history |
| Age | Age in years |
| **Outcome** (target) | **1 = has diabetes, 0 = no diabetes** |

---

## 3. Preprocessing (Cleaning the Data)

Raw data is rarely ready for a model. These steps were done:

1. **Removed duplicate rows.** None were found.
2. **Fixed hidden missing values.** In Glucose, BloodPressure, SkinThickness, Insulin and BMI, a value of `0` is impossible for a living person, so `0` really means "not measured". These zeros were converted to `NaN` (missing).

   | Column | Missing values |
   |---|---|
   | Insulin | 374 |
   | SkinThickness | 227 |
   | BloodPressure | 35 |
   | BMI | 11 |
   | Glucose | 5 |

3. **Filled missing values (imputation).** Missing values are replaced with the **median** of that column. The median is used instead of the average because it isn't affected by extreme values.
4. **Scaling.** `StandardScaler` puts all features on a similar scale. Without it, a feature with large numbers (like Insulin) could dominate one with small numbers (like BMI).
5. **Train/test split.** 80% of the data is used to train the model and 20% is kept hidden to test it. The split is *stratified*, so both parts have the same ratio of diabetic and non-diabetic patients.

**Pipeline:** steps 3 and 4 and the model are combined into one scikit-learn `Pipeline`. This means the exact same cleaning is applied during training and during prediction, and no information from the test data leaks into training.

---

## 4. Models Tried

| Model | Short explanation |
|---|---|
| **Logistic Regression** | A simple model that gives each feature a weight and turns the result into a probability between 0 and 1. Easy to explain. |
| **Random Forest** | Builds many decision trees and takes a vote. More flexible, but harder to explain. |
| **Gradient Boosting** | Builds trees one after another, each fixing the mistakes of the previous one. |

---

## 5. Evaluation

### Terms explained

- **Accuracy:** out of all predictions, how many were correct. It can be misleading when classes are unbalanced.
- **Precision:** when the model says "diabetic", how often is it right?
- **Recall:** out of all patients who really have diabetes, how many did the model catch? This matters most for screening, because missing a sick person is the worst mistake.
- **F1 score:** one number that balances precision and recall.
- **ROC-AUC:** how well the model separates diabetic from non-diabetic patients across all thresholds. 0.5 means random guessing and 1.0 means perfect.
- **Cross-validation (CV):** the training data is split into 5 parts, and the model is trained and tested 5 times, each time on a different part. This gives a more reliable score than a single test.

### Results

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.842** | 0.734 | 0.603 | **0.704** | 0.650 | 0.813 |
| Random Forest | 0.824 | 0.734 | 0.638 | 0.556 | 0.594 | 0.812 |
| Gradient Boosting | 0.813 | 0.760 | 0.689 | 0.574 | 0.626 | 0.831 |

**In plain words:** Logistic Regression caught about 70% of the diabetic patients in the test set (recall 0.70), and about 60% of the people it flagged really had diabetes (precision 0.60).

---

## 6. Why Logistic Regression Was Selected

1. **Best cross-validated ROC-AUC (0.842).** This is the fairest score, because it uses five different splits of the data instead of one.
2. **Highest recall (0.70).** It misses the fewest diabetic patients, which is the priority for a screening tool.
3. **Differences between models are small.** With only 154 test patients, the gaps are mostly noise, so the simplest model is the safest choice.
4. **It is simple, fast and explainable.** In healthcare it matters to be able to explain why a prediction was made.

---

## 7. Limitations

- **Small dataset:** only 768 patients, so results may change with more data.
- **One population:** all patients are Pima Indian women aged 21+, so the model may not work as well for men, children or other groups.
- **Many missing values:** Insulin and SkinThickness were missing for a large share of patients and had to be filled with the median.
- **Moderate accuracy (about 73%):** good enough to screen, not good enough to diagnose.
- **Fixed threshold:** a probability of 0.5 or more is classed as "high risk". A clinic could lower it to catch more patients, at the cost of more false alarms.

---

## 8. Project Structure

```
Pokak_technologies/
├── collect_data.py     # Step 1: downloads the dataset
├── preprocess.py       # Step 2: cleans the data
├── train.py            # Step 3: trains, compares and saves the best model
├── app.py              # Step 4: Flask prediction API
├── streamlit_app.py    # Step 5: web page for entering data
├── test_api.py         # Sends sample requests to test the API
├── requirements.txt    # Python packages needed
├── data/               # raw_data.csv and clean_data.csv
└── model/              # best_model.joblib (the saved model)
```

---

## 9. Setup and Run

Python 3.9 or newer is required.

**1. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
```

**2. Install the packages**
```bash
python -m pip install -r requirements.txt
```

**3. Collect the data, clean it, and train the model** (run in this order)
```bash
python collect_data.py
python preprocess.py
python train.py
```

**4. Start the API** (keep this terminal open)
```bash
python app.py
```
The API runs at `http://127.0.0.1:5000`.

**5. Start the web page** (open a second terminal and activate the venv)
```bash
python -m streamlit run streamlit_app.py
```
Your browser opens at `http://localhost:8501`. Type the patient details and click **Predict**.

---

## 10. Using the API

An **API** is a way for one program to talk to another. You send data to a URL and get an answer back. Data is sent as **JSON**, a simple text format of `"name": value` pairs.

**Endpoint:** `POST http://127.0.0.1:5000/predict`

**Example request**
```json
{
  "Pregnancies": 2,
  "Glucose": 150,
  "BloodPressure": 80,
  "SkinThickness": 30,
  "Insulin": 100,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.6,
  "Age": 50
}
```

**Example response**
```json
{
  "label": "High diabetes risk",
  "prediction": 1,
  "probability": 0.761
}
```

- `prediction`: 1 = likely diabetic, 0 = not likely.
- `probability`: the model's confidence, from 0 to 1 (0.761 = 76.1%).

`SkinThickness` and `Insulin` are optional. If they are left out or set to 0, the model fills them in with the median.

**Quick test:** with `app.py` running, run `python test_api.py` in another terminal.

### Error handling

The API checks every input and returns a clear message instead of crashing (HTTP status 400):

| Problem | Example message |
|---|---|
| Value out of realistic range | `Glucose must be between 40 and 300` |
| Required field missing | `Pregnancies is required` |
| Text instead of a number | `Age must be a number` |
| Not valid JSON | `Please send valid JSON data` |

**Allowed ranges:** Pregnancies 0–20, Glucose 40–300, BloodPressure 20–200, SkinThickness 0–100, Insulin 0–900, BMI 10–70, DiabetesPedigreeFunction 0–3, Age 15–100.

---

## 11. Tools Used

Python, pandas, NumPy, scikit-learn, joblib, Flask, Streamlit, requests.# diabetes-risk-prediction-ml
