README.txt
FactoryGuard AI – IoT Predictive Maintenance Engine
 Project Overview

Project Title: FactoryGuard AI
Domain: IoT + Machine Learning
Problem Type: Time-Series Classification

FactoryGuard AI is an IoT-based predictive maintenance system designed to predict catastrophic failures in robotic arms 24 hours in advance using sensor data such as vibration, temperature, and pressure.

The objective is to reduce unscheduled downtime, enable preemptive maintenance, and improve operational efficiency in manufacturing plants.

 Use Case

A manufacturing plant contains 500 robotic arms equipped with multiple sensors.
Failures are rare but costly.
This system predicts potential failures early so maintenance teams can act proactively.

Project Structure
FactoryGuardAI/
│
├── data/
│   ├── raw_sensor_data.pkl
│   └── processed_features.pkl
│
├── models/
│   ├── baseline_rf_model.pkl
│   └── xgb_production_model.pkl
│
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── triain_base_model.py
│   ├── train_production_mode.py
│   └── evaluation_model.py
│
├── requirements.txt
└── README.txt

 Technologies Used

Python 3.11

Pandas & NumPy

Scikit-learn

XGBoost

Joblib

VS Code

 Week 1 – Baseline Model Development
🔹 Data Ingestion

Synthetic IoT sensor data generated for 500 robotic arms

Hourly readings of vibration, temperature, and pressure

Failure labels created with realistic imbalance (~5%)

🔹 Feature Engineering

Rolling Mean & Standard Deviation (1, 6, 12 hours)

Exponential Moving Average (EMA)

Lag features (t-1, t-2)

NaN handling and data validation

🔹 Baseline Model

Random Forest Classifier

Class imbalance handled using balanced weights

Model saved using Joblib

 Week 2 – Production Model Development
🔹 Production Model

XGBoost Classifier

scale_pos_weight used for imbalance handling

Optimized for large-scale and real-world deployment

🔹 Evaluation Metrics

Confusion Matrix

Precision, Recall, F1-score

ROC-AUC score

