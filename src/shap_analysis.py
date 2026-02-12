import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Model & Data
# -----------------------------
model = joblib.load("models/xgb_production_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

df = pd.read_csv("data/processed_features.csv")

X = df[feature_columns]

# Use small sample for speed
X_sample = X.sample(1000, random_state=42)

print("✅ Model and data loaded")

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("✅ SHAP values calculated")

# -----------------------------
# Create plots folder
# -----------------------------
os.makedirs("shap_outputs", exist_ok=True)

# -----------------------------
# 1️⃣ Global Feature Importance
# -----------------------------
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("shap_outputs/shap_summary.png", bbox_inches="tight")
plt.close()

print("✅ SHAP summary plot saved")

# -----------------------------
# 2️⃣ Bar Importance Plot
# -----------------------------
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.savefig("shap_outputs/shap_bar.png", bbox_inches="tight")
plt.close()

print("✅ SHAP bar plot saved")

print("\n🔥 Explainability complete. Check shap_outputs folder.")