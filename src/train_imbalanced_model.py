import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from xgboost import XGBClassifier

# (Optional) SMOTE – only if needed
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load processed features
# -------------------------------
df = joblib.load("data/processed_features.pkl")

X = df.drop(columns=["failure"])
y = df["failure"]

print("Class distribution:\n", y.value_counts())

# -------------------------------
# 2. Train–Test Split (STRATIFIED)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------
# 3. Handle Class Imbalance
#    (Preferred: Class Weights)
# -------------------------------
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# -------------------------------
# 4. XGBoost Model (Production)
# -------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,  # KEY POINT
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluation (PR-AUC)
# -------------------------------
y_probs = model.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(y_test, y_probs)
print(f"\n✅ Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")

# High-precision threshold (reduce false alarms)
threshold = 0.8
y_pred = (y_probs >= threshold).astype(int)

print("\n📊 Classification Report (High Precision Focus):")
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------------
# 6. Save Model
# -------------------------------
joblib.dump(model, "models/xgb_imbalanced_production.pkl")
print("✅ Imbalanced XGBoost model saved")