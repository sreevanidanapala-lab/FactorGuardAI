import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
# Ensure models folder exists
os.makedirs("models", exist_ok=True)

#  Load processed data
df = pd.read_csv("data/processed_data.csv")

# Features and target
X = df[["vibration_mean_24h", "temp_mean_24h", "pressure_mean_24h"]]
y = df["failure"]

#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Create model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

#  Train model
model.fit(X_train, y_train)

#  Save model
joblib.dump(model, "models/factoryguard_model.pkl")
print("✅ Model trained and saved successfully")
