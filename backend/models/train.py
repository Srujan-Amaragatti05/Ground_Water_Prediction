# ============================================
# Feature Engineering for Groundwater Dataset
# ============================================

# 1. Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

import pickle
# ============================================
# 2. Load dataset safely (handle encoding issues)
# ============================================
try:
    df = pd.read_csv(
        r"D:\NewE\College\Project\GW\backend\models\groundwater_ap_2014_20232.csv",
        encoding="utf-8"
    )
except UnicodeDecodeError:
    df = pd.read_csv(
        r"D:\NewE\College\Project\GW\backend\models\groundwater_ap_2014_20232.csv",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

print("Dataset loaded successfully")
print("Initial shape:", df.shape)

# ============================================
# 3. Convert Date column to datetime
# ============================================
# Convert Date to datetime for temporal feature extraction
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# ============================================
# 4. Extract temporal features
# ============================================
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Date column is no longer needed for ML
df.drop(columns=["Date"], inplace=True)

# ============================================
# 5. Drop non-ML-friendly columns
# ============================================
# STATE_UT → constant value (no predictive power)
# VILLAGE → high cardinality, sparse, weak signal
df.drop(columns=["STATE_UT", "VILLAGE"], inplace=True)

# ============================================
# 6. Handle missing values (minimal & safe)
# ============================================
df["BLOCK"].fillna("Unknown", inplace=True)
df.dropna(subset=["LATITUDE", "Year", "Month"], inplace=True)

# ============================================
# 7. Encode categorical features
# ============================================
# Using Label Encoding:
# - Suitable for tree-based & regression models
# - Keeps dimensionality low (important for small datasets)

le_district = LabelEncoder()
le_block = LabelEncoder()

df["DISTRICT_ENC"] = le_district.fit_transform(df["DISTRICT"])
df["BLOCK_ENC"] = le_block.fit_transform(df["BLOCK"])

# Drop original categorical columns
df.drop(columns=["DISTRICT", "BLOCK"], inplace=True)

# ============================================
# 8. Define Features (X) and Target (y)
# ============================================
X = df.drop(columns=["WL(mbgl)"])
y = df["WL(mbgl)"]

# ============================================
# 9. Train-Test Split (80-20)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# 10. Feature Scaling (important for regression)
# ============================================
# Scale only numerical features
scaler = StandardScaler()
print("\nFeature columns before scaling:")
print(X_test.head())  # Check feature columns before scaling

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature columns after scaling:")
print(X_test_scaled[:5])  # Check first 5 rows of scaled features


# Convert back to DataFrame (for readability)
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# ============================================
# 11. Final Output Check
# ============================================
print("\nFinal ML-ready shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nSample feature rows:")
print(X_train.head())

print("\nSample target values:")
print(y_train.head())

print("\nData preprocessing completed successfully!")

# ============================================
# Save scaler and encoders (ONE TIME)
# ============================================

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("district_encoder.pkl", "wb") as f:
    pickle.dump(le_district, f)

with open("block_encoder.pkl", "wb") as f:
    pickle.dump(le_block, f)

print("Scaler and encoders saved successfully!")


# ============================================
# 2. Create a helper function for evaluation
# ============================================
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ============================================
# 3. Train models
# ============================================

# ---- Model 1: Linear Regression (Baseline) ----
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ---- Model 2: Random Forest Regressor ----
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ---- Model 3: XGBoost Regressor ----
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# ============================================
# 4. Evaluate models
# ============================================
results = []

results.append(("Linear Regression", *evaluate_model(y_test, y_pred_lr)))
results.append(("Random Forest", *evaluate_model(y_test, y_pred_rf)))
results.append(("XGBoost", *evaluate_model(y_test, y_pred_xgb)))

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2 Score"]
)

print("\nModel Performance Comparison:")
print(results_df)

# ============================================
# 5. Identify best model (highest R2)
# ============================================
best_model_name = results_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
print("\nBest Performing Model:", best_model_name)

# ============================================
# 6. Actual vs Predicted plot (Best Model)
# ============================================
if best_model_name == "Linear Regression":
    y_best_pred = y_pred_lr
elif best_model_name == "Random Forest":
    y_best_pred = y_pred_rf
else:
    y_best_pred = y_pred_xgb

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_best_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual Groundwater Level (mbgl)")
plt.ylabel("Predicted Groundwater Level (mbgl)")
plt.title(f"Actual vs Predicted (Best Model: {best_model_name})")
# Points closer to diagonal indicate better prediction accuracy
plt.show()

# ============================================
# 2. Save the trained Random Forest model
# ============================================
# Saving model allows reuse without retraining
model_path = "random_forest_groundwater_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(rf, file)

print("Random Forest model saved successfully!")

# ============================================
# 3. Reload the saved model from disk
# ============================================
with open(model_path, "rb") as file:
    rf_loaded = pickle.load(file)

print("Random Forest model loaded successfully!")

# ============================================
# 4. Predict using both models (original & loaded)
# ============================================
y_pred_original = rf.predict(X_test)
y_pred_loaded = rf_loaded.predict(X_test)

# ============================================
# 5. Verify predictions are identical
# ============================================
same_predictions = np.allclose(y_pred_original, y_pred_loaded)

print("\nPrediction consistency check:")
print("Are predictions identical?", same_predictions)

# ============================================
# 6. Final confirmation message
# ============================================
if same_predictions:
    print("\n✅ Model persistence successful!")
    print("The loaded model produces identical predictions.")
else:
    print("\n❌ Model persistence failed!")
