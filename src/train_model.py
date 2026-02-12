import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------
# 1️⃣ Load Data
# ---------------------------------------------------

df = pd.read_csv("data/factory_guard_dataset.csv")

print("Training class distribution:")
print(df["failure_24h_ahead"].value_counts())

# ---------------------------------------------------
# 2️⃣ Features & Target
# ---------------------------------------------------

X = df.drop("failure_24h_ahead", axis=1)
y = df["failure_24h_ahead"]

# ---------------------------------------------------
# 3️⃣ Train-Test Split (Stratified)
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ---------------------------------------------------
# 4️⃣ Apply SMOTE (ONLY on training data)
# ---------------------------------------------------

smote = SMOTE(random_state=42, sampling_strategy=0.3)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_train_res).value_counts())

# ---------------------------------------------------
# 5️⃣ Train XGBoost Model
# ---------------------------------------------------

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # already balanced by SMOTE
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train_res, y_train_res)

print("✅ XGBoost model trained successfully")

# ---------------------------------------------------
# 6️⃣ Evaluation
# ---------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

print(f"\n🔥 PR-AUC Score: {round(pr_auc, 4)}")

# ---------------------------------------------------
# 7️⃣ Save Model
# ---------------------------------------------------

joblib.dump(model, "models/xgb_production_model.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("✅ Production model saved successfully")
