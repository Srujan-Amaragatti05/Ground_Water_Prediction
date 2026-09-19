"""
backend/routers/phase2.py
=========================
Phase-2 FastAPI router.

Exposes POST /api/phase2/predict — delegates all inference to the
Phase-2 Unified Prediction Service (phase2_prediction_service.py).

This router does NOT load models, fit/apply scalers, or contain any
ML-inference logic.  It is a thin HTTP adapter over Step-E.
"""

from __future__ import annotations

import sys
import os
import math
from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Add the dl directory to sys.path so phase2_prediction_service is importable
# regardless of how uvicorn is started.
# ---------------------------------------------------------------------------
_DL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dl")
if _DL_DIR not in sys.path:
    sys.path.insert(0, _DL_DIR)

import phase2_prediction_service as _svc
import phase2_location_service   as _loc_svc


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/phase2", tags=["Phase-2 DL Prediction"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class Phase2PredictRequest(BaseModel):
    """
    Request body for POST /api/phase2/predict.

    temporal_sequence: 5 rows × 4 columns
        Column order: [WL(mbgl), Rainfall, Temperature, Humidity]

    latitude, longitude: required when model == 'hybrid'.
    """

    model: Literal["lstm", "gru", "hybrid"] = Field(
        ...,
        description="Phase-2 model to use: 'lstm', 'gru', or 'hybrid'.",
    )
    temporal_sequence: List[List[float]] = Field(
        ...,
        description=(
            "5 consecutive annual observations. "
            "Each row: [WL(mbgl), Rainfall, Temperature, Humidity]."
        ),
    )
    latitude: Optional[float] = Field(
        default=None,
        description="Location latitude (required for 'hybrid').",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Location longitude (required for 'hybrid').",
    )

    # --- temporal sequence validation ---
    @field_validator("temporal_sequence")
    @classmethod
    def validate_temporal_sequence(cls, v):
        if len(v) != 5:
            raise ValueError(
                f"temporal_sequence must have exactly 5 rows; got {len(v)}."
            )
        for i, row in enumerate(v):
            if len(row) != 4:
                raise ValueError(
                    f"Row {i} must have exactly 4 features "
                    f"[WL(mbgl), Rainfall, Temperature, Humidity]; got {len(row)}."
                )
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Row {i}, column {j}: value must be numeric, got {type(val).__name__}."
                    )
                if not math.isfinite(float(val)):
                    raise ValueError(
                        f"Row {i}, column {j}: value must be finite; got {val}."
                    )
        return v

    # --- spatial validation for hybrid ---
    @model_validator(mode="after")
    def validate_hybrid_spatial(self):
        if self.model == "hybrid":
            if self.latitude is None or self.longitude is None:
                raise ValueError(
                    "latitude and longitude are required when model is 'hybrid'."
                )
            if not math.isfinite(self.latitude):
                raise ValueError(f"latitude must be finite; got {self.latitude}.")
            if not math.isfinite(self.longitude):
                raise ValueError(f"longitude must be finite; got {self.longitude}.")
        return self


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class Phase2PredictResponse(BaseModel):
    """Prediction result returned by POST /api/phase2/predict."""

    model: str = Field(..., description="Model used for this prediction.")
    predicted_wl: float = Field(
        ..., description="Predicted groundwater level in meters below ground level."
    )
    unit: str = Field(default="mbgl", description="Prediction unit (mbgl).")
    sequence_length: int = Field(
        default=5, description="Length of temporal sequence used."
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=Phase2PredictResponse,
    summary="Phase-2 DL groundwater-level prediction",
    description=(
        "Given five consecutive historical annual observations, predicts the "
        "groundwater level for the following year using a Phase-2 deep-learning model "
        "(LSTM, GRU, or Hybrid Spatio-Temporal GRU)."
    ),
)
def phase2_predict(request: Phase2PredictRequest) -> Phase2PredictResponse:
    """
    Delegate to the Phase-2 Unified Prediction Service and return a clean response.
    """
    try:
        result = _svc.predict(
            model_name=request.model,
            temporal_sequence=request.temporal_sequence,
            latitude=request.latitude,
            longitude=request.longitude,
        )
    except ValueError as exc:
        # Input validation error from the service layer
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        # Missing model artifact — server-side error; don't expose full path
        raise HTTPException(
            status_code=500,
            detail=f"Model artifact unavailable for '{request.model}'. "
                   "Please contact the server administrator.",
        )
    except Exception as exc:
        # Unexpected inference failure
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        )

    return Phase2PredictResponse(
        model=result["model"],
        predicted_wl=result["predicted_wl"],
        unit=result["unit"],
        sequence_length=result["sequence_length"],
    )


# ---------------------------------------------------------------------------
# Utility endpoint: list supported models
# ---------------------------------------------------------------------------
@router.get(
    "/models",
    summary="List supported Phase-2 models",
    description="Returns the list of Phase-2 model identifiers accepted by /predict.",
)
def list_models():
    return {"supported_models": _svc.get_supported_models()}


# ---------------------------------------------------------------------------
# Step-G: Location-based prediction response schema
# ---------------------------------------------------------------------------
class Phase2LocationPredictResponse(BaseModel):
    """Response for GET /api/phase2/locations/{location_id}/predict."""

    location_id:        str   = Field(..., description="Phase-2 LOCATION_ID used.")
    model:              str   = Field(..., description="Model used for this prediction.")
    predicted_wl:       float = Field(..., description="Predicted WL in mbgl.")
    unit:               str   = Field(default="mbgl")
    sequence_length:    int   = Field(default=5)
    sequence_start_year: int  = Field(..., description="First year in the input sequence.")
    sequence_end_year:  int   = Field(..., description="Last year in the input sequence.")
    prediction_year:    int   = Field(..., description="Year being predicted (sequence_end_year + 1).")
    latitude:           float = Field(..., description="Location latitude (LATITUDE column).")
    longitude:          float = Field(..., description="Location longitude (LONGITUDE column).")


# ---------------------------------------------------------------------------
# Step-G: Location-based prediction endpoint
# ---------------------------------------------------------------------------
@router.get(
    "/locations/{location_id}/predict",
    response_model=Phase2LocationPredictResponse,
    summary="Phase-2 location-based groundwater-level prediction",
    description=(
        "Given a Phase-2 LOCATION_ID, retrieves the latest valid five-consecutive-year "
        "historical sequence from the dataset and predicts the groundwater level for the "
        "following year using the specified Phase-2 deep-learning model."
    ),
)
def location_predict(
    location_id: str,
    model: Literal["lstm", "gru", "hybrid"] = Query(
        ...,
        description="Phase-2 model to use: 'lstm', 'gru', or 'hybrid'.",
    ),
) -> Phase2LocationPredictResponse:
    """
    Retrieve historical sequence for location_id and delegate to Step-E.
    """
    # --- Step-G: retrieve historical sequence ---
    try:
        loc_result = _loc_svc.get_location_sequence(location_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Dataset unavailable.")
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location_id}' was not found in the Phase-2 dataset.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred during sequence retrieval.")

    # --- Step-E: delegate inference ---
    try:
        result = _svc.predict(
            model_name=model,
            temporal_sequence=loc_result.temporal_sequence,
            latitude=loc_result.latitude,
            longitude=loc_result.longitude,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Model artifact unavailable for '{model}'.",
        )
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")

    return Phase2LocationPredictResponse(
        location_id=loc_result.location_id,
        model=result["model"],
        predicted_wl=result["predicted_wl"],
        unit=result["unit"],
        sequence_length=result["sequence_length"],
        sequence_start_year=loc_result.sequence_start_year,
        sequence_end_year=loc_result.sequence_end_year,
        prediction_year=loc_result.prediction_year,
        latitude=loc_result.latitude,
        longitude=loc_result.longitude,
    )
