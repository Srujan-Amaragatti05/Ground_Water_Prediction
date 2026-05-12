import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import math

# Load CSV
df = pd.read_csv("dataset.csv", low_memory=False)

# -----------------------------
# 1. Clean column names
# -----------------------------
df.columns = df.columns.str.strip().str.upper()

# -----------------------------
# 2. Rename WL column
# -----------------------------
df = df.rename(columns={"WL(MBGL)": "WL"})

# -----------------------------
# 3. Convert Date → datetime
# -----------------------------
df["DATE"] = pd.to_datetime(
    df["DATE"],
    format="%d-%m-%y",
    errors="coerce"
)


# Drop invalid dates
df = df.dropna(subset=["DATE"])

# Extract YEAR
df["YEAR"] = df["DATE"].dt.year

print(df["YEAR"].min(), df["YEAR"].max())
# -----------------------------
# 4. Create WELL_ID
# -----------------------------
df["WELL_ID"] = (
    df["STATE_UT"].astype(str) + "_" +
    df["DISTRICT"].astype(str) + "_" +
    df["BLOCK"].astype(str) + "_" +
    df["VILLAGE"].astype(str) + "_" +
    df["LATITUDE"].astype(str) + "_" +
    df["LONGITUDE"].astype(str)
)

# -----------------------------
# 5. Keep only required columns
# -----------------------------
df = df[["WELL_ID", "YEAR", "WL"]]

# -----------------------------
# 6. Sort for time-series
# -----------------------------
df = df.sort_values(by=["WELL_ID", "YEAR"])

print(df.head())

# -----------------------------
# 3. Parameters
# -----------------------------
WINDOW_SIZE = 5   # past 5 years
MIN_YEARS = 6     # minimum history required

X_list = []
y_list = []

# -----------------------------
# 4. Group by WELL_ID
# -----------------------------
for well_id, group in df.groupby("WELL_ID"):

    # Sort each well by year
    group = group.sort_values("YEAR")

    wl_values = group["WL"].values

    # Skip wells with insufficient data
    if len(wl_values) < MIN_YEARS:
        continue

    # -----------------------------
    # 5. Create sliding windows
    # -----------------------------
    for i in range(len(wl_values) - WINDOW_SIZE):
        X_seq = wl_values[i:i + WINDOW_SIZE]      # past 5 years
        y_seq = wl_values[i + WINDOW_SIZE]         # next year

        X_list.append(X_seq)
        y_list.append(y_seq)

# -----------------------------
# 6. Convert to NumPy arrays
# -----------------------------
X = np.array(X_list)
y = np.array(y_list)

# -----------------------------
# 7. Reshape for LSTM/GRU
# -----------------------------
# X -> (samples, time_steps, features)
# y -> (samples, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((-1, 1))

# -----------------------------
# 8. Print dataset info
# -----------------------------
print("Final Dataset Shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 9. Print sample sequences
# -----------------------------
print("\nSample Input (X[0]):")
print(X[0].flatten())

print("\nSample Output (y[0]):")
print(y[0])

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved X.npy and y.npy")


# -----------------------------
# 1. Load trained model
# -----------------------------
# Change filename if using GRU
model = load_model("lstm_groundwater_model.h5")

# -----------------------------
# 2. Load prepared dataset
# -----------------------------
# If already in memory, skip loading
# X = np.load("X.npy")
# y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 3. Define backcasting window
# -----------------------------
# Example: predict past known years
# Using earlier portion of data only
backcast_end = int(len(X) * 0.3)

X_backcast = X[:backcast_end]
y_actual   = y[:backcast_end]

# -----------------------------
# 4. Predict using trained model
# -----------------------------
y_pred = model.predict(X_backcast)

# -----------------------------
# 5. Compute evaluation metrics
# -----------------------------
rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)

print(f"Backcasting RMSE: {rmse:.3f}")
print(f"Backcasting MAE : {mae:.3f}")

# -----------------------------
# 6. Plot actual vs predicted
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_actual[:200], label="Actual WL", marker="o")
plt.plot(y_pred[:200], label="Predicted WL", marker="x")
plt.title("Backcasting Validation: Actual vs Predicted Groundwater Level")
plt.xlabel("Historical Time Index")
plt.ylabel("WL (mbgl)")
plt.legend()
plt.grid(True)
plt.show()
