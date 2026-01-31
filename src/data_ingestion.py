import pandas as pd
import numpy as np

np.random.seed(42)

NUM_ARMS = 500
HOURS = 30 * 24  # 30 days hourly data

data = []

for arm_id in range(NUM_ARMS):
    for hour in range(HOURS):
        vibration = np.random.normal(5, 1)
        temperature = np.random.normal(70, 5)
        pressure = np.random.normal(30, 3)

        # Failure logic (catastrophic failure)
        failure = 1 if (vibration > 6.0 and temperature > 72) else 0

        data.append([
            arm_id, hour, vibration, temperature, pressure, failure
        ])

df = pd.DataFrame(
    data,
    columns=["arm_id", "hour", "vibration", "temperature", "pressure", "failure"]
)

df.to_csv("data/raw_sensor_data.csv", index=False)
print("✅ Raw sensor data generated")