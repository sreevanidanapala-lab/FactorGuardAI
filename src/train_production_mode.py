# XGBoost production model with same safeguards.

import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

#  Load processed features
file_path = "data/processed_features.pkl"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found. Run feature_engineering.py first.")
df = joblib.load(file_path)

#  Select numeric features
feature_cols = [c for c in df.columns if "roll" in c or "lag" in c or "ema" in c or c in ["vibration","temperature","pressure"]]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')

# Fill NaNs
X = X.fillna(0)

#  Ensure target is strictly 0/1 integer
y = df["failure"]

# Convert to 0/1 integers and clip any out-of-range values
y = y.astype(int)
y = y.clip(0,1)

# Check there are **at least 1 of each class**
if len(np.unique(y)) < 2:
    raise ValueError("Your dataset does not have both classes 0 and 1. Increase failures in data_ingestion.py!")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Classes in y_train:\n", y_train.value_counts())
print("Classes in y_test:\n", y_test.value_counts())

#  XGBoost classifier (safe)
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',   # ✅ for binary classification
    random_state=42
)

#  Train model
model.fit(X_train, y_train)
print("✅ XGBoost model trained successfully")

# Evaluate model
y_pred = model.predict(X_test)

# Safe predict_proba
if len(np.unique(y_train)) == 2:
    y_prob = model.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, y_prob)
else:
    y_prob = np.zeros(X_test.shape[0])
    roc = None
    print("⚠ Only one class present. ROC AUC cannot be computed.")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
if roc is not None:
    print("ROC AUC:", roc)

#  Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_production_model.pkl")
print("✅ XGBoost model saved at models/xgb_production_model.pkl")