import numpy as np
import pandas as pd

df = pd.read_csv("data/raw_data.csv")
print("Rows before cleaning:", len(df))

# 1. Remove duplicate rows
df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))

# 2. In these columns a value of 0 is impossible (e.g. glucose can't be 0),
#    so 0 really means "missing". Replace 0 with NaN.
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

print("\nMissing values per column:")
print(df.isnull().sum())

# Missing values are filled and features are scaled in train.py
df.to_csv("data/clean_data.csv", index=False)
print("\nSaved data/clean_data.csv")