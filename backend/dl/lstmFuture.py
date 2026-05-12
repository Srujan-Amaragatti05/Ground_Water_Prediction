import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================================================
# FUTURE GROUNDWATER FORECASTING USING LSTM
# =========================================================
# This script performs SHORT-HORIZON recursive forecasting
# using the last available real groundwater observations.
#
# IMPORTANT:
# - No ground truth exists for future years (2024–2026)
# - Hence, NO RMSE / MAE is computed
# =========================================================

# NOTE:
# Future years (2024–2026) have no observed groundwater data.
# Therefore, accuracy metrics such as RMSE or MAE cannot be computed.
#
# Recursive forecasting feeds predictions back as inputs.
# Prediction errors accumulate over time.
# Hence, forecasting is limited to short horizons (1–3 years).


# -----------------------------
# 1. Load trained LSTM model
# -----------------------------
model = load_model("lstm_groundwater_model.h5")


# -----------------------------
# 2. Load prepared dataset
# -----------------------------
# X shape: (samples, 5, 1)
X = np.load("X.npy")

print("X shape:", X.shape)


# -----------------------------
# 3. Define last real data window
# -----------------------------
# Last real observed years:
# [2019, 2020, 2021, 2022, 2023]
#
# This window is used to predict 2024
last_real_window = X[-1]           # shape: (5, 1)
current_input = last_real_window.copy()


# -----------------------------
# 4. Recursive forecasting
# -----------------------------
future_years = [2024, 2025, 2026]
future_predictions = []

for year in future_years:

    # Predict groundwater level for next year
    next_wl = model.predict(
        current_input.reshape(1, 5, 1),
        verbose=0
    )[0, 0]

    future_predictions.append(next_wl)

    # Update sliding window:
    # Drop oldest year, append predicted value
    current_input = np.vstack([
        current_input[1:],      # keep last 4 years
        [[next_wl]]             # add predicted year
    ])


# -----------------------------
# 5. Print future forecasts
# -----------------------------
print("\nFuture Groundwater Level Forecasts (LSTM):")
for year, wl in zip(future_years, future_predictions):
    print(f"Year {year}: Predicted WL = {wl:.2f} mbgl")


# -----------------------------
# 6. Plot forecast trend
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    future_years,
    future_predictions,
    marker="o",
    linestyle="-",
    linewidth=2,
    label="Forecasted Groundwater Level"
)

plt.title("Short-Horizon Groundwater Forecast (LSTM)")
plt.xlabel("Year")
plt.ylabel("Groundwater Level (mbgl)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
