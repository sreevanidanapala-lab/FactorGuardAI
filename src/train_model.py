import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
from xgboost import XGBClassifier

df = pd.read_csv("data/processed_features.csv")

X = df.drop("failure_24h_ahead", axis=1)
y = df["failure_24h_ahead"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print("Scale_pos_weight:", round(scale_pos_weight, 2))

model = XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("✅ XGBoost model trained successfully")

# -----------------------------
# Threshold Tuning
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# Lower threshold for better recall
threshold = 0.25
y_pred = (y_prob > threshold).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

print("🔥 PR-AUC Score:", round(pr_auc, 4))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_production_model.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("✅ Production model saved successfully")
