import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import math

# =========================================================
# BACKCASTING EVALUATION FOR LSTM MODEL
# =========================================================
# Backcasting means:
#   Using historical input windows to predict
#   known past groundwater levels and comparing
#   predictions with actual observed values.
# =========================================================


# -----------------------------
# 1. Load trained LSTM model
# -----------------------------
model = load_model("lstm_groundwater_model.h5")


# -----------------------------
# 2. Load prepared time-series dataset
# -----------------------------
# X shape: (samples, 5, 1)
# y shape: (samples, 1)
X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# 3. Select historical windows for backcasting
# -----------------------------
# Example:
#   Using early part of the dataset where
#   true groundwater values are already known.
#
# This simulates:
#   2010–2014 -> predict 2015
#   2011–2015 -> predict 2016
#
# NOTE:
# Backcasting uses ONLY past data (no future leakage)
# -----------------------------

backcast_ratio = 0.3   # use first 30% of data for backcasting
backcast_end = int(len(X) * backcast_ratio)

X_backcast = X[:backcast_end]
y_actual = y[:backcast_end]


# -----------------------------
# 4. Generate predictions
# -----------------------------
y_pred = model.predict(X_backcast, verbose=0)


# -----------------------------
# 5. Evaluation metrics
# -----------------------------
rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print("\nLSTM Backcasting Evaluation Metrics:")
print(f"RMSE : {rmse:.3f} mbgl")
print(f"MAE  : {mae:.3f} mbgl")
print(f"R²   : {r2:.3f}")


# -----------------------------
# 6. Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    y_actual[:200],
    label="Actual Groundwater Level",
    marker="o",
    linewidth=2
)

plt.plot(
    y_pred[:200],
    label="Predicted Groundwater Level",
    marker="x",
    linewidth=2
)

plt.title("Backcasting Validation: Actual vs Predicted Groundwater Levels (LSTM)")
plt.xlabel("Historical Time Index")
plt.ylabel("Groundwater Level (mbgl)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
