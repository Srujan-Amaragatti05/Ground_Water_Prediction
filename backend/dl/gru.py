import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
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


# -----------------------------
# 1. Load prepared dataset
# -----------------------------
# If already in memory, skip this
# X = np.load("X.npy")
# y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 2. Time-based train-test split
# -----------------------------
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test  = X[train_size:]

y_train = y[:train_size]
y_test  = y[train_size:]

print("Train samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])

# -----------------------------
# 3. Build LSTM model
# -----------------------------
model = Sequential()

model.add(
    LSTM(
        units=50,
        activation="tanh",
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dense(1))  # output layer

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# -----------------------------
# 4. Train the model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# 5. Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Evaluation metrics
# -----------------------------
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")

# -----------------------------
# 7. Plot actual vs predicted
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test[:200], label="Actual WL")
plt.plot(y_pred[:200], label="Predicted WL")
plt.title("Actual vs Predicted Groundwater Level")
plt.xlabel("Sample Index")
plt.ylabel("WL (mbgl)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 8. Save trained model
# -----------------------------
model.save("lstm_groundwater_model.h5")
print("Model saved as lstm_groundwater_model.h5")
