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
# 2. Load model & data
# -------------------------------
model = joblib.load("models/xgb_production_model.pkl")
df = joblib.load("data/processed_features.pkl")

X = df.drop(columns=["failure"])

# -------------------------------
# 3. Sample data (IMPORTANT)
# -------------------------------
X_sample = X.sample(2000, random_state=42)

# -------------------------------
# 4. Use NEW SHAP API (FIX)
# -------------------------------
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# -------------------------------
# 5. SHAP Summary Plot (Beeswarm)
# -------------------------------
plt.figure(figsize=(10, 14))

shap.summary_plot(
    shap_values,
    X_sample,
    max_display=20,
    show=True
)