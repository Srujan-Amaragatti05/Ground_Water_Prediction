import numpy as np
from tensorflow.keras.models import load_model

# NOTE:
# Recursive forecasting feeds model predictions back as inputs.
# Small prediction errors accumulate over time.
# Therefore, forecasts are limited to short horizons (1–3 years).

# -----------------------------
# 1. Load trained DL model
# -----------------------------
# Use LSTM or GRU model
model = load_model("lstm_groundwater_model.h5")
# model = load_model("gru_groundwater_model.h5")

# -----------------------------
# 2. Load prepared dataset
# -----------------------------
X = np.load("X.npy")   # (samples, 5, 1)
y = np.load("y.npy")

print("X shape:", X.shape)

# -----------------------------
# 3. Select last real window
# -----------------------------
# Last available real historical window
last_window = X[-1]              # shape: (5, 1)
current_input = last_window.copy()

# -----------------------------
# 4. Recursive forecasting
# -----------------------------
N_FUTURE_YEARS = 3   # limit to short horizon

future_predictions = []

for step in range(N_FUTURE_YEARS):

    # Predict next year
    next_wl = model.predict(
        current_input.reshape(1, 5, 1),
        verbose=0
    )[0, 0]

    future_predictions.append(next_wl)

    # Update input window:
    # drop oldest value, append prediction
    current_input = np.vstack([
        current_input[1:],          # last 4 years
        [[next_wl]]                 # predicted year
    ])

# -----------------------------
# 5. Print results
# -----------------------------
print("\nShort-horizon groundwater forecasts:")

for i, wl in enumerate(future_predictions, start=1):
    print(f"Year +{i}: Predicted WL = {wl:.2f} mbgl")
