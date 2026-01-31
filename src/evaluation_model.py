import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os

# Load model
model_path = "models/xgb_production_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Train your model first!")

model = joblib.load(model_path)

# Load features
features_path = "data/processed_features.pkl"
if not os.path.exists(features_path):
    raise FileNotFoundError("Run feature_engineering.py first!")

df = joblib.load(features_path)

feature_cols = [c for c in df.columns if "roll" in c or "lag" in c or "ema" in c or c in ["vibration","temperature","pressure"]]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df["failure"].astype(int)

# Split for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Predict
y_pred = model.predict(X_test)
if len(np.unique(y_train)) == 2:
    y_prob = model.predict_proba(X_test)[:,1]
else:
    y_prob = np.zeros(X_test.shape[0])

# Evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
if len(np.unique(y_train)) == 2:
    print("ROC AUC:", roc_auc_score(y_test, y_prob))