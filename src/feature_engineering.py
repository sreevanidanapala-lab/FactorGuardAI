import pandas as pd
import os

df = pd.read_csv("data/raw_sensor_data.csv")

df.sort_values(["arm_id", "hour"], inplace=True)

# Rolling features
df["vibration_roll_mean_24h"] = (
    df.groupby("arm_id")["vibration"]
    .rolling(24).mean().reset_index(level=0, drop=True)
)

df["temp_roll_mean_24h"] = (
    df.groupby("arm_id")["temperature"]
    .rolling(24).mean().reset_index(level=0, drop=True)
)

df["pressure_roll_std_24h"] = (
    df.groupby("arm_id")["pressure"]
    .rolling(24).std().reset_index(level=0, drop=True)
)

# 24h ahead failure
df["failure_24h_ahead"] = (
    df.groupby("arm_id")["failure"].shift(-24)
)

df.dropna(inplace=True)

df = df.drop(columns=["failure"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/processed_features.csv", index=False)

print("✅ Feature engineering completed")
print("Final dataset:", df.shape)
print("Failure rate (24h ahead):", round(df["failure_24h_ahead"].mean() * 100, 3), "%")

