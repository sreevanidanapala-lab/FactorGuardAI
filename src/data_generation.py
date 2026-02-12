import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_ARMS = 500
HOURS = 30 * 24  # 30 days hourly

data = []

for arm_id in range(NUM_ARMS):

    base_vibration = np.random.normal(5, 0.5)
    base_temp = np.random.normal(70, 2)
    base_pressure = np.random.normal(30, 1)

    failure_hour = np.random.randint(400, 700)  # controlled failure time

    for hour in range(HOURS):

        # Gradual degradation before failure
        degradation = max(0, hour - (failure_hour - 48)) / 48

        vibration = base_vibration + degradation * 3 + np.random.normal(0, 0.3)
        temperature = base_temp + degradation * 10 + np.random.normal(0, 1)
        pressure = base_pressure + degradation * 5 + np.random.normal(0, 0.5)

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