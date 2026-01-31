import pandas as pd
import os
import joblib
df = pd.read_csv("data/raw_sensor_data.csv")

df.sort_values(["arm_id", "hour"], inplace=True)

# Rolling window features (last 24 hours)
df["vibration_mean_24h"] = df.groupby("arm_id")["vibration"].rolling(24).mean().reset_index(0, drop=True)
df["temp_mean_24h"] = df.groupby("arm_id")["temperature"].rolling(24).mean().reset_index(0, drop=True)
df["pressure_mean_24h"] = df.groupby("arm_id")["pressure"].rolling(24).mean().reset_index(0, drop=True)

df.dropna(inplace=True)

df.to_csv("data/processed_data.csv", index=False)
print("✅ Feature engineering completed")
df = pd.read_csv("data/raw_sensor_data.csv")

# ... do feature engineering ...

# ✅ Ensure the folder exists
os.makedirs("data", exist_ok=True)  # <- creates 'data' folder if it doesn't exist

# Save processed features
joblib.dump(df, "data/processed_features.pkl")  # <- file path is here
print("✅ Processed features saved at data/processed_features.pkl")