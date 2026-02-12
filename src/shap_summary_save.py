import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Reproducibility
# -------------------------------
np.random.seed(42)

# -------------------------------
# 2. Load model and data
# -------------------------------
model = joblib.load("models/xgb_production_model.pkl")
df = joblib.load("data/processed_features.pkl")

X = df.drop(columns=["failure"])

# Sample for faster SHAP
X_sample = X.sample(1000, random_state=42)

# -------------------------------
# 3. Create SHAP Explainer
# -------------------------------
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# -------------------------------
# 4. Show SHAP Summary Plot
# -------------------------------
plt.figure(figsize=(10, 12))

shap.summary_plot(
    shap_values,
    X_sample,
    max_display=20,
    show=True   # IMPORTANT → shows plot
)

plt.show()