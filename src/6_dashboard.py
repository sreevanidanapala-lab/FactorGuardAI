import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# IMPORTANT: Must run using
# streamlit run dashboard.py
# -----------------------------

st.set_page_config(page_title="FactoryGuard AI", layout="wide")

st.title("🏭 FactoryGuard AI - Predictive Maintenance")

# -----------------------------
# Load Model Safely
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_production_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_artifacts()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔧 Machine Sensor Inputs")

arm_id = st.sidebar.number_input("Arm ID", 0, 1000, 1)
hour = st.sidebar.number_input("Operating Hour", 0, 2000, 500)

vibration = st.sidebar.slider("Vibration", 0.0, 15.0, 5.0)
temperature = st.sidebar.slider("Temperature", 40.0, 120.0, 70.0)
pressure = st.sidebar.slider("Pressure", 10.0, 50.0, 30.0)

vibration_roll = st.sidebar.slider("24h Avg Vibration", 0.0, 15.0, 5.0)
temp_roll = st.sidebar.slider("24h Avg Temperature", 40.0, 120.0, 70.0)
pressure_std = st.sidebar.slider("24h Pressure Std", 0.0, 10.0, 2.0)

# -----------------------------
# Create Input Data
# -----------------------------
input_dict = {
    "arm_id": arm_id,
    "hour": hour,
    "vibration": vibration,
    "temperature": temperature,
    "pressure": pressure,
    "vibration_roll_mean_24h": vibration_roll,
    "temp_roll_mean_24h": temp_roll,
    "pressure_roll_std_24h": pressure_std,
}

input_df = pd.DataFrame([input_dict])

# Ensure exact column match
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
prob = model.predict_proba(input_df)[0][1]
threshold = 0.35
prediction = int(prob > threshold)

st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("Failure Probability (Next 24h)", f"{prob:.2%}")

with col2:
    if prediction == 1:
        st.error("⚠ HIGH FAILURE RISK")
    else:
        st.success("✅ Machine Normal")

# -----------------------------
# SHAP Explanation (Stable Version)
# -----------------------------
st.subheader("🔍 Feature Impact (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# Convert to DataFrame for clean plotting
shap_df = pd.DataFrame({
    "Feature": feature_columns,
    "SHAP Value": shap_values[0]
}).sort_values(by="SHAP Value", key=abs, ascending=False)

fig, ax = plt.subplots()
ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
ax.set_xlabel("Impact on Prediction")
ax.set_title("SHAP Feature Importance (Local Explanation)")
ax.invert_yaxis()

st.pyplot(fig)

# -----------------------------
# Simulated Trend
# -----------------------------
st.subheader("📈 Simulated 24h Trend")

trend_data = pd.DataFrame({
    "Vibration": np.random.normal(vibration, 0.5, 24),
    "Temperature": np.random.normal(temperature, 2, 24),
})

st.line_chart(trend_data)

st.markdown("---")
st.caption("FactoryGuard AI | Built by Sreevani 🚀")

with col2:
    if prediction == 1:
        st.error("⚠ HIGH FAILURE RISK in next 24 hours")
    else:
        st.success("✅ Machine Operating Normally")

# -----------------------------
# SHAP Explanation
# -----------------------------
st.subheader("🔍 Model Explanation (SHAP)")

shap_values = explainer.shap_values(input_df)

fig, ax = plt.subplots()
shap.force_plot(
    explainer.expected_value,
    shap_values,
    input_df,
    matplotlib=True,
    show=False
)

st.pyplot(fig)

# -----------------------------
# Simulated Sensor Trend
# -----------------------------
st.subheader("📈 Last 24h Sensor Trend (Simulated)")

trend_data = pd.DataFrame({
    "Vibration": np.random.normal(vibration, 0.5, 24),
    "Temperature": np.random.normal(temperature, 2, 24),
})

st.line_chart(trend_data)

st.markdown("---")
st.caption("FactoryGuard AI | Built by Sreevani 🚀")