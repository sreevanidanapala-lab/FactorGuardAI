from flask import Flask, request, jsonify
import joblib
import numpy as np

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/xgb_production_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

app = Flask(__name__)

THRESHOLD = 0.35

@app.route("/")
def home():
    return {"message": "FactoryGuard AI API is running"}

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        input_data = [data[col] for col in feature_columns]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    input_array = np.array(input_data).reshape(1, -1)

    prob = model.predict_proba(input_array)[0][1]
    prediction = int(prob > THRESHOLD)

    result = {
        "failure_probability": round(float(prob), 4),
        "predicted_failure_24h": prediction
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)