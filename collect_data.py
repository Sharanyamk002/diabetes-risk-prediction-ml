import os
import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# The file has no header row, so we give the column names ourselves
df = pd.read_csv(URL, header=None, names=columns)

os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())