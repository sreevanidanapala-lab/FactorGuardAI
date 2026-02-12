import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score

# -----------------------------
# Load Processed Data
# -----------------------------
df = joblib.load("data/processed_features.pkl")

# Features
feature_cols = [
    "vibration",
    "temperature",
    "pressure",
    "vibration_roll_mean_24h",
    "temp_roll_mean_24h",
    "pressure_roll_std_24h"
]

X = df[feature_cols].fillna(0)
y = df["failure_24h_ahead"].astype(int)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training class distribution:")
print(y_train.value_counts())

# -----------------------------
# Handle Imbalance Properly
# -----------------------------
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

print("Scale_pos_weight:", round(scale_pos_weight, 2))

# -----------------------------
# XGBoost Model
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # 🔥 imbalance fix
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

print("✅ XGBoost model trained successfully")

# -----------------------------
# PR-AUC Evaluation (Correct Metric)
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, y_prob)

print("\n📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

print("🔥 PR-AUC Score:", round(pr_auc, 4))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_production_model.pkl")
joblib.dump(feature_cols, "models/feature_columns.pkl")

print("✅ Production model saved successfully")
