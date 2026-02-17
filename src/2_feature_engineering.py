import pandas as pd
import os

df = pd.read_csv("data/raw_sensor_data.csv")

df.sort_values(["arm_id", "hour"], inplace=True)

# -----------------------------
# Rolling Features
# -----------------------------
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

# -----------------------------
# NEW: 24-Hour Warning Window Label
# -----------------------------
df["failure_24h_ahead"] = 0

for arm_id in df["arm_id"].unique():
    arm_data = df[df["arm_id"] == arm_id]
    failure_hours = arm_data[arm_data["failure"] == 1]["hour"]

    for fh in failure_hours:
        df.loc[
            (df["arm_id"] == arm_id) &
            (df["hour"] >= fh - 24) &
            (df["hour"] < fh),
            "failure_24h_ahead"
        ] = 1

df.dropna(inplace=True)

df = df.drop(columns=["failure"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/processed_features.csv", index=False)

print("✅ Feature engineering completed")
print("Dataset shape:", df.shape)
print("Failure rate (24h ahead):", round(df["failure_24h_ahead"].mean() * 100, 3), "%")

