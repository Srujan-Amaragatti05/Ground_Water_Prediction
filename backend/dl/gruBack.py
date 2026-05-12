import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import math

# =========================================================
# BACKCASTING EVALUATION FOR GRU MODEL
# =========================================================
# Backcasting:
#   Predict known historical groundwater levels
#   using only earlier observations.
# =========================================================


# -----------------------------
# 1. Load trained GRU model
# -----------------------------
gru_model = load_model("gru_groundwater_model.h5")


# -----------------------------
# 2. Load prepared dataset
# -----------------------------
# X shape: (samples, 5, 1)
# y shape: (samples, 1)
X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# 3. Use SAME backcasting window as LSTM
# -----------------------------
# IMPORTANT:
# This ensures fair comparison between LSTM and GRU
backcast_ratio = 0.3
backcast_end = int(len(X) * backcast_ratio)

X_backcast = X[:backcast_end]
y_actual = y[:backcast_end]


# -----------------------------
# 4. Generate GRU predictions
# -----------------------------
y_pred_gru = gru_model.predict(X_backcast, verbose=0)


# -----------------------------
# 5. Compute evaluation metrics
# -----------------------------
rmse_gru = math.sqrt(mean_squared_error(y_actual, y_pred_gru))
mae_gru = mean_absolute_error(y_actual, y_pred_gru)
r2_gru = r2_score(y_actual, y_pred_gru)

print("\nGRU Backcasting Evaluation Metrics:")
print(f"RMSE : {rmse_gru:.3f} mbgl")
print(f"MAE  : {mae_gru:.3f} mbgl")
print(f"R²   : {r2_gru:.3f}")


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
    y_pred_gru[:200],
    label="GRU Predicted Groundwater Level",
    marker="x",
    linewidth=2
)

plt.title("Backcasting Validation: Actual vs Predicted Groundwater Levels (GRU)")
plt.xlabel("Historical Time Index")
plt.ylabel("Groundwater Level (mbgl)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
