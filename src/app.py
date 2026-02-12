from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time
import os

app = Flask(__name__)

# -----------------------------
# Load Model + Feature List
# -----------------------------
if not os.path.exists("models/xgb_production_model.pkl"):
    raise FileNotFoundError("Model not found. Train model first.")

model = joblib.load("models/xgb_production_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")


# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/")
def home():
    return "FactoryGuard AI - 24h Failure Prediction API Running"


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON input provided"}), 400

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Ensure correct feature order
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    latency = (time.time() - start_time) * 1000

    risk_level = "HIGH" if probability > 0.8 else "LOW"

    return jsonify({
        "failure_probability_24h": round(float(probability), 4),
        "risk_level": risk_level,
        "latency_ms": round(latency, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)