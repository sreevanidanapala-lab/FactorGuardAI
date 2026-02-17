import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_ARMS = 500
HOURS = 30 * 24  # 30 days

data = []

for arm_id in range(NUM_ARMS):

    base_vibration = np.random.normal(5, 0.3)
    base_temp = np.random.normal(70, 1.5)
    base_pressure = np.random.normal(30, 1)

    # Each arm fails once in 30 days
    failure_hour = np.random.randint(400, 650)

    for hour in range(HOURS):

        # 96-hour gradual degradation window
        degradation = max(0, hour - (failure_hour - 96)) / 96

        vibration = base_vibration + degradation * 5 + np.random.normal(0, 0.2)
        temperature = base_temp + degradation * 18 + np.random.normal(0, 0.8)
        pressure = base_pressure + degradation * 7 + np.random.normal(0, 0.3)

        failure = 1 if hour == failure_hour else 0

        data.append([arm_id, hour, vibration, temperature, pressure, failure])

df = pd.DataFrame(
    data,
    columns=["arm_id", "hour", "vibration", "temperature", "pressure", "failure"]
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_sensor_data.csv", index=False)

print("✅ Raw sensor data generated")
print("Failure Rate:", round(df["failure"].mean() * 100, 3), "%")