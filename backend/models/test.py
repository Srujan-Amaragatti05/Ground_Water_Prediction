import pickle
import pandas as pd

# Load trained model
with open("random_forest_groundwater_model.pkl", "rb") as f:
    rf_loaded = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler_loaded = pickle.load(f)

# Load encoders
with open("district_encoder.pkl", "rb") as f:
    district_encoder = pickle.load(f)

with open("block_encoder.pkl", "rb") as f:
    block_encoder = pickle.load(f)

print("Model, scaler, and encoders loaded successfully!")


# --------------------------------------------
# RAW user input (same format as dataset)
# --------------------------------------------
user_input = pd.DataFrame([{
    "LATITUDE": 16.50,
    "LONGITUDE": 80.60,
    "Year": 2022,
    "Month": 1,
    "DISTRICT": "Ntr",
    "BLOCK": "Vijayawada Rural"
}])

print("Raw user input:")
print(user_input)

# --------------------------------------------
# Encode categorical variables INTERNALLY
# --------------------------------------------
user_input["DISTRICT_ENC"] = district_encoder.transform(user_input["DISTRICT"])
user_input["BLOCK_ENC"] = block_encoder.transform(user_input["BLOCK"])

# Drop original categorical columns
user_input.drop(columns=["DISTRICT", "BLOCK"], inplace=True)

# --------------------------------------------
# Scale numerical features
# --------------------------------------------
user_input_scaled = scaler_loaded.transform(user_input)

# --------------------------------------------
# Predict using loaded model
# --------------------------------------------
prediction = rf_loaded.predict(user_input_scaled)

print("\nPredicted Groundwater Level (mbgl):")
print(round(prediction[0], 2))
