import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# 1. Reproducibility
# -------------------------------
np.random.seed(42)

# -------------------------------
# 2. Create output folder
# -------------------------------
os.makedirs("reports/figures", exist_ok=True)

# -------------------------------
# 3. Load model & data
# -------------------------------
model = joblib.load("models/xgb_production_model.pkl")
df = joblib.load("data/processed_features.pkl")

X = df.drop(columns=["failure"])

# -------------------------------
# 4. Sample data (important for SHAP)
# -------------------------------
X_sample = X.sample(2000, random_state=42)

# -------------------------------
# 5. SHAP Explainer (SAFE API)
# -------------------------------
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# -------------------------------
# 6. Save SHAP Summary Plot
# -------------------------------
plt.figure(figsize=(10, 14))

shap.summary_plot(
    shap_values,
    X_sample,
    max_display=20,
    show=False   # IMPORTANT for saving
)

plt.title("Feature Impact on Failure Prediction (SHAP Summary)")
plt.tight_layout()

plt.savefig(
    "reports/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✅ SHAP summary plot saved to reports/figures/shap_summary.png")