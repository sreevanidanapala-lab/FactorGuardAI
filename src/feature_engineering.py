import pandas as pd
import joblib
import os

df = pd.read_csv("data/raw_sensor_data.csv")

df.sort_values(["arm_id", "hour"], inplace=True)

# -----------------------------
# Rolling Features (24h window)
# -----------------------------
df["vibration_roll_mean_24h"] = (
    df.groupby("arm_id")["vibration"]
    .rolling(24)
    .mean()
    .reset_index(level=0, drop=True)
)

df["temp_roll_mean_24h"] = (
    df.groupby("arm_id")["temperature"]
    .rolling(24)
    .mean()
    .reset_index(level=0, drop=True)
)

df["pressure_roll_std_24h"] = (
    df.groupby("arm_id")["pressure"]
    .rolling(24)
    .std()
    .reset_index(level=0, drop=True)
)

# -----------------------------
# 24 HOURS AHEAD PREDICTION
# -----------------------------
df["failure_24h_ahead"] = (
    df.groupby("arm_id")["failure"].shift(-24)
)

df.dropna(inplace=True)

# Keep only relevant columns
df = df.drop(columns=["failure"])

os.makedirs("data", exist_ok=True)
joblib.dump(df, "data/processed_features.pkl")

print("✅ Feature engineering completed")
print("Final dataset size:", df.shape)
print("Failure rate (24h ahead):", round(df["failure_24h_ahead"].mean() * 100, 3), "%")
