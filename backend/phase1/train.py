# ============================================
# PHASE-1: Groundwater DL Model (AP only)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input

# ============================================
# STEP 1 — LOAD & FILTER DATA
# ============================================
df = pd.read_csv("D:\\NewE\\College\\Project\\GW\\dataset\\fixed_real_dataset.csv")

df = df[df["STATE_UT"] == "Andhra Pradesh"]

print("Filtered shape:", df.shape)

# ============================================
# STEP 2 — PREPROCESSING
# ============================================
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

df["YEAR"] = df["Date"].dt.year
df["MONTH"] = df["Date"].dt.month

# Create WELL_ID
df["WELL_ID"] = (
    df["DISTRICT"].astype(str) + "_" +
    df["BLOCK"].astype(str) + "_" +
    df["VILLAGE"].astype(str)
)

# Keep only required columns
df = df[[
    "WELL_ID", "LATITUDE", "LONGITUDE",
    "YEAR", "MONTH",
    "WL(mbgl)", "Rainfall", "Temperature", "Humidity"
]]

# Handle missing values
df = df.dropna()

# Sort properly
df = df.sort_values(by=["WELL_ID", "YEAR", "MONTH"])

# ============================================
# STEP 3 — FEATURE SELECTION
# ============================================
features = ["WL(mbgl)", "Rainfall", "Temperature", "Humidity"]

# Scale features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# ============================================
# STEP 4 — SEQUENCE CREATION (WINDOW = 12)
# ============================================
def create_sequences(data, window=12):
    X, y = [], []
    
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][0])  # WL is target
    
    return np.array(X), np.array(y)

X_all, y_all = [], []

for well in df["WELL_ID"].unique():
    group = df[df["WELL_ID"] == well][features].values
    
    if len(group) > 12:
        X, y = create_sequences(group, window=12)
        X_all.append(X)
        y_all.append(y)

X = np.vstack(X_all)
y = np.hstack(y_all)

print("X shape:", X.shape)  # (samples, 12, 4)
print("y shape:", y.shape)

# ============================================
# STEP 5 — TIME-BASED SPLIT
# ============================================
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================
# STEP 6 — BUILD MODELS
# ============================================

# -------- RNN --------
rnn_model = Sequential([
    Input(shape=(12, 4)),
    SimpleRNN(50, activation='tanh'),
    Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# -------- LSTM --------
lstm_model = Sequential([
    Input(shape=(12, 4)),
    LSTM(64),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# -------- GRU --------
gru_model = Sequential([
    Input(shape=(12, 4)),
    GRU(64),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mse')
gru_model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# ============================================
# STEP 7 — EVALUATION
# ============================================
def evaluate(model):
    pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    return pred, rmse, mae, r2

rnn_pred, rnn_rmse, rnn_mae, rnn_r2 = evaluate(rnn_model)
lstm_pred, lstm_rmse, lstm_mae, lstm_r2 = evaluate(lstm_model)
gru_pred, gru_rmse, gru_mae, gru_r2 = evaluate(gru_model)

# Plot best model (we'll pick later)
plt.figure(figsize=(6,6))
plt.scatter(y_test, lstm_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], '--')
plt.title("Actual vs Predicted (LSTM)")
plt.xlabel("Actual WL")
plt.ylabel("Predicted WL")
plt.show()

# ============================================
# STEP 8 — FUTURE FORECAST (Recursive)
# ============================================
def forecast(model, last_seq, steps=3):
    preds = []
    seq = last_seq.copy()
    
    for _ in range(steps):
        p = model.predict(seq.reshape(1, 12, 4))[0][0]
        preds.append(p)
        
        # shift window
        new_row = [p, seq[-1][1], seq[-1][2], seq[-1][3]]
        seq = np.vstack([seq[1:], new_row])
    
    return preds

last_seq = X_test[-1]

future_preds = forecast(lstm_model, last_seq, steps=3)

print("Future Predictions:", future_preds)

# Plot forecast
plt.plot(range(len(y_test[-12:])), y_test[-12:], label="Actual")
plt.plot(range(len(y_test[-12:]), len(y_test[-12:])+3),
         future_preds, label="Forecast", marker='o')

plt.legend()
plt.title("Future Forecast (LSTM)")
plt.show()

# ============================================
# STEP 9 — MODEL COMPARISON
# ============================================
results = pd.DataFrame({
    "Model": ["RNN", "LSTM", "GRU"],
    "RMSE": [rnn_rmse, lstm_rmse, gru_rmse],
    "MAE": [rnn_mae, lstm_mae, gru_mae],
    "R2": [rnn_r2, lstm_r2, gru_r2]
})

print("\nModel Comparison:")
print(results)

best = results.sort_values("R2", ascending=False).iloc[0]
print("\nBest Model:", best["Model"])