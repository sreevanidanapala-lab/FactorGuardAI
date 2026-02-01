import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load model and data
# -------------------------------
model = joblib.load("models/xgb_production_model.pkl")
df = joblib.load("data/processed_features.pkl")

X = df.drop(columns=["failure"])
y = df["failure"]

# -------------------------------
# 2. Initialize SHAP Explainer
# -------------------------------
explainer = shap.Explainer(model, X)

# -------------------------------
# 3. Select a FAILURE case
# -------------------------------
failure_index = y[y == 1].index[0]
X_failure = X.loc[[failure_index]]

# -------------------------------
# 4. Generate SHAP values
# -------------------------------
shap_values = explainer(X_failure)

# -------------------------------
# 5. Local Explanation Plot
# -------------------------------
shap.plots.waterfall(shap_values[0], max_display=10)

# -------------------------------
# 6. Global Feature Importance
# -------------------------------
shap.summary_plot(
    explainer(X[:5000]),
    X[:5000],
    plot_type="bar"
)