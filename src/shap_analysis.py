import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Load Model + Data
# -----------------------------
model = joblib.load("models/xgb_production_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")
df = joblib.load("data/processed_features.pkl")

# -----------------------------
# 2. Prepare Feature Matrix
# -----------------------------
X = df[feature_cols].fillna(0)
y = df["failure_24h_ahead"]

# Use sample for speed
X_sample = X.sample(min(1000, len(X)), random_state=42)

# -----------------------------
# 3. Create SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# -----------------------------
# 4. Global Feature Importance
# -----------------------------
print("Generating SHAP summary plot...")

plt.figure()
shap.summary_plot(shap_values, X_sample, show=True)

# -----------------------------
# 5. Local Explanation (One Failure Case)
# -----------------------------
failure_indices = y[y == 1].index

if len(failure_indices) > 0:
    idx = failure_indices[0]
    single_instance = X.loc[[idx]]

    shap_single = explainer.shap_values(single_instance)

    print("\nGenerating Local SHAP Waterfall Plot...")

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_single[0],
        feature_names=feature_cols
    )
else:
    print("No failure cases found for local explanation.")