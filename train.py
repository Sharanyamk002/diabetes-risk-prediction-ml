import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# Load cleaned data
df = pd.read_csv("data/clean_data.csv")
X = df.drop("Outcome", axis=1)   # input features
y = df["Outcome"]                # what we want to predict (0 or 1)

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# The three models we want to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

best_name, best_score, best_pipe = None, 0, None

for name, model in models.items():
    # Pipeline = fill missing values -> scale -> model (all in one object)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    # 5-fold cross validation on training data (a fair score)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()

    # Train and test
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    print("\n", name)
    print("  CV ROC-AUC :", round(cv_auc, 3))
    print("  Accuracy   :", round(accuracy_score(y_test, pred), 3))
    print("  Precision  :", round(precision_score(y_test, pred), 3))
    print("  Recall     :", round(recall_score(y_test, pred), 3))
    print("  F1 score   :", round(f1_score(y_test, pred), 3))
    print("  ROC-AUC    :", round(roc_auc_score(y_test, prob), 3))

    # Remember the best model (highest CV ROC-AUC)
    if cv_auc > best_score:
        best_name, best_score, best_pipe = name, cv_auc, pipe

print("\nBest model:", best_name)

os.makedirs("model", exist_ok=True)
joblib.dump(best_pipe, "model/best_model.joblib")
print("Saved model/best_model.joblib")