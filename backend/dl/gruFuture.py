import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================================================
# FUTURE FORECASTING & QUALITATIVE COMPARISON (LSTM vs GRU)
# =========================================================
# This script compares long-term forecast behavior
# using recursive forecasting.
#
# IMPORTANT:
# - No ground truth exists for future years
# - This comparison is QUALITATIVE only
# =========================================================

# This comparison is qualitative in nature.
# Future years have no observed groundwater levels,
# so numerical accuracy metrics are not applicable.
# The comparison focuses on trend direction, smoothness,
# and stability of forecasts.

# -----------------------------
# 1. Load trained models
# -----------------------------
lstm_model = load_model("lstm_groundwater_model.h5")
gru_model  = load_model("gru_groundwater_model.h5")


# -----------------------------
# 2. Load prepared dataset
# -----------------------------
# X shape: (samples, 5, 1)
X = np.load("X.npy")

print("X shape:", X.shape)


# -----------------------------
# 3. Define last real window
# -----------------------------
# Last observed years:
# [2019, 2020, 2021, 2022, 2023]
last_real_window = X[-1]

future_years = [2024, 2025, 2026]


# -----------------------------
# 4. Recursive forecasting (LSTM)
# -----------------------------
lstm_predictions = []
current_input = last_real_window.copy()

for _ in future_years:
    next_wl = lstm_model.predict(
        current_input.reshape(1, 5, 1),
        verbose=0
    )[0, 0]

    lstm_predictions.append(next_wl)

    current_input = np.vstack([
        current_input[1:],
        [[next_wl]]
    ])


# -----------------------------
# 5. Recursive forecasting (GRU)
# -----------------------------
gru_predictions = []
current_input = last_real_window.copy()

for _ in future_years:
    next_wl = gru_model.predict(
        current_input.reshape(1, 5, 1),
        verbose=0
    )[0, 0]

    gru_predictions.append(next_wl)

    current_input = np.vstack([
        current_input[1:],
        [[next_wl]]
    ])


# -----------------------------
# 6. Plot GRU future forecast
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    future_years,
    gru_predictions,
    marker="o",
    linestyle="-",
    linewidth=2,
    label="GRU Forecast"
)
plt.title("Future Groundwater Forecast (GRU)")
plt.xlabel("Year")
plt.ylabel("Groundwater Level (mbgl)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 7. Overlay LSTM vs GRU
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    future_years,
    lstm_predictions,
    marker="o",
    linewidth=2,
    label="LSTM Forecast"
)

plt.plot(
    future_years,
    gru_predictions,
    marker="x",
    linewidth=2,
    label="GRU Forecast"
)

plt.title("Qualitative Comparison: LSTM vs GRU Future Forecast")
plt.xlabel("Year")
plt.ylabel("Groundwater Level (mbgl)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 8. Print forecast values
# -----------------------------
print("\nFuture Groundwater Forecasts (Qualitative Comparison):")
for i, year in enumerate(future_years):
    print(
        f"Year {year} | "
        f"LSTM: {lstm_predictions[i]:.2f} mbgl | "
        f"GRU: {gru_predictions[i]:.2f} mbgl"
    )
