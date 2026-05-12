# -----------------------------
# 1. Import required libraries
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# -----------------------------
# 2. Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Groundwater Level Prediction API",
    description="Predicts groundwater level (WL in mbgl) using Random Forest",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# 3. Load trained model & preprocessors
#    (Loaded once at startup)
# -----------------------------
with open("models/random_forest_groundwater_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/district_encoder.pkl", "rb") as f:
    district_encoder = pickle.load(f)

with open("models/block_encoder.pkl", "rb") as f:
    block_encoder = pickle.load(f)

# -----------------------------
# 4. Define request schema
#    (Raw user input)
# -----------------------------
class GroundwaterInput(BaseModel):
    year: int = Field(alias="Year")
    month: int = Field(alias="Month")
    latitude: float = Field(alias="LATITUDE")
    longitude: float = Field(alias="LONGITUDE")
    district: str = Field(alias="DISTRICT")
    block: str = Field(alias="BLOCK")

    class Config:
        populate_by_name = True

# -----------------------------
# 5. Define prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_groundwater(data: GroundwaterInput):
    """
    Accepts raw user inputs,
    preprocesses them,
    and returns groundwater level prediction.
    """

    # -----------------------------
    # 6. Encode categorical features
    # -----------------------------
    # Normalize text to match training data
    district_clean = data.district.strip()
    block_clean = data.block.strip()

    # Encode categorical features
    district_encoded = district_encoder.transform([district_clean])[0]
    block_encoded = block_encoder.transform([block_clean])[0]


    # -----------------------------
    # 7. Create feature array
    #    (Order MUST match training)
    # -----------------------------
    features = np.array([[
        data.year,
        data.month,
        data.latitude,
        data.longitude,
        district_encoded,
        block_encoded
    ]])

    # -----------------------------
    # 8. Scale numerical features
    # -----------------------------
    features_scaled = scaler.transform(features)

    # -----------------------------
    # 9. Predict groundwater level
    # -----------------------------
    predicted_wl = model.predict(features_scaled)[0]

    # -----------------------------
    # 10. Assign risk category
    # -----------------------------
    if predicted_wl < 3:
        risk = "Safe"
    elif 3 <= predicted_wl <= 6:
        risk = "Warning"
    else:
        risk = "Critical"

    # -----------------------------
    # 11. Return response
    # -----------------------------
    return {
        "predicted_WL": round(float(predicted_wl), 2),
        "risk_category": risk
    }
