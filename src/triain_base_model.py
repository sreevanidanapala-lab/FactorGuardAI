# Random Forest baseline with safe predict_proba and numeric feature check.

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load processed features
file_path = "data/processed_features.pkl"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found. Run feature_engineering.py first.")
df = joblib.load(file_path)

# Select numeric features only
feature_cols = [c for c in df.columns if "roll" in c or "lag" in c or "ema" in c or c in ["vibration","temperature","pressure"]]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df["failure"]

# Drop any remaining NaNs
X = X.dropna()
y = y.loc[X.index]

# Train/test split with stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Check class distribution
print("Classes in y_train:", y_train.value_counts())
print("Classes in y_test:", y_test.value_counts())

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

# Safe predict_proba
if len(model.classes_) == 2:
    y_prob = model.predict_proba(X_test)[:,1]
else:
    y_prob = np.zeros(X_test.shape[0])
    print("⚠ Only one class present. ROC AUC cannot be computed.")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

if len(model.classes_) == 2:
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_rf_model.pkl")
print("✅ Baseline Random Forest model saved")