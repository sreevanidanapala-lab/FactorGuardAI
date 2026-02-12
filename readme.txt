🏭 FactoryGuard AI
Predictive Maintenance System for Robotic Arms

FactoryGuard AI is an end-to-end Machine Learning system that predicts robotic arm failures 24 hours in advance using sensor time-series data.

The project includes:

Data simulation

Feature engineering

Imbalanced classification using XGBoost

SHAP explainability

Flask API deployment

Interactive Streamlit dashboard

🚀 Project Overview

Unexpected machine failures cause:

Production downtime

Financial loss

Safety risks

This system predicts whether a robotic arm will fail within the next 24 hours using:

Vibration

Temperature

Pressure

Rolling trend features

Instead of predicting the exact failure moment, the model predicts early warning signals, which is more practical in real-world maintenance systems.

📊 Model Performance
Metric	Value
Accuracy	99%
Recall (Failure)	99%
Precision (Failure)	72%
F1 Score	0.83
PR-AUC	0.96

Why PR-AUC?
Because failure prediction is a rare event problem, and PR-AUC is more appropriate than accuracy.

🧠 Feature Engineering

Key engineered features:

24-hour rolling mean vibration

24-hour rolling mean temperature

24-hour rolling standard deviation of pressure

24-hour ahead failure labeling

This helps capture degradation patterns, not just sudden spikes.

🤖 Model Details

Algorithm: XGBoost Classifier

Imbalance Handling: scale_pos_weight

Evaluation Metric: Precision-Recall AUC

Threshold tuning: 0.35

Why XGBoost?

Excellent for structured/tabular data

Handles imbalance well

Captures nonlinear degradation patterns

🔍 Explainability (SHAP)

To ensure model transparency:

SHAP values are used to explain predictions

Identifies which features increase failure risk

Provides both global and local explanations

This is critical for industrial AI trust.

🌐 Deployment
1️⃣ Flask API

The trained model is deployed as a REST API.

Endpoint:

POST /predict


Input:

{
    "arm_id": 10,
    "hour": 500,
    "vibration": 8.2,
    "temperature": 92.1,
    "pressure": 38.5,
    "vibration_roll_mean_24h": 7.9,
    "temp_roll_mean_24h": 90.3,
    "pressure_roll_std_24h": 2.1
}


Output:

{
    "failure_probability": 0.87,
    "predicted_failure_24h": 1
}

2️⃣ Streamlit Dashboard

Interactive monitoring dashboard includes:

Live sensor input simulation

Failure probability display

Risk alerts

SHAP feature explanation

Simulated trend visualization

Run:

streamlit run src/dashboard.py

📂 Project Structure
FactoryGuardAI/
│
├── data/
│   ├── raw_sensor_data.csv
│   └── processed_features.csv
│
├── models/
│   ├── xgb_production_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── explain_model.py
│   ├── app.py
│   └── dashboard.py
│
├── requirements.txt
└── README.md
