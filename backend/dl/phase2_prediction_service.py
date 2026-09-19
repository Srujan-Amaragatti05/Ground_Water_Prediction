"""
phase2_prediction_service.py
============================
Unified inference-only prediction service for Phase-2 groundwater-level models.

Supported models
----------------
  'lstm'   — Phase-2 LSTM baseline
  'gru'    — Phase-2 GRU baseline
  'hybrid' — Phase-2 Hybrid Spatio-Temporal GRU

Input contract
--------------
  temporal_sequence : list[list[float]]  shape (5, 4)
      Five consecutive annual observations, each row:
          [WL(mbgl), Rainfall, Temperature, Humidity]
  latitude  : float  (required only for 'hybrid')
  longitude : float  (required only for 'hybrid')

Output contract
---------------
  dict with keys:
    'model'          — model name string
    'predicted_wl'   — predicted WL in original mbgl scale (float)
    'unit'           — 'mbgl'
    'sequence_length'— 5

This service is INFERENCE-ONLY.  It never trains, fine-tunes, or overwrites
model artifacts.
"""

import os
import numpy as np
import joblib
import tensorflow as tf

# ---------------------------------------------------------------------------
# Artifact paths — fixed to Phase-2 project directory
# ---------------------------------------------------------------------------
_DL_DIR = os.path.dirname(os.path.abspath(__file__))

_ARTIFACT_PATHS = {
    "lstm": {
        "model":  os.path.join(_DL_DIR, "phase2_lstm_model.keras"),
        "scaler": os.path.join(_DL_DIR, "phase2_scalers.pkl"),
    },
    "gru": {
        # GRU intentionally reuses the LSTM-trained scaler (same preprocessing)
        "model":  os.path.join(_DL_DIR, "phase2_gru_model.keras"),
        "scaler": os.path.join(_DL_DIR, "phase2_scalers.pkl"),
    },
    "hybrid": {
        "model":  os.path.join(_DL_DIR, "phase2_hybrid_model.keras"),
        "scaler": os.path.join(_DL_DIR, "phase2_hybrid_scalers.pkl"),
    },
}

# Module-level cache: {model_name: (tf_model, scaler_bundle)}
_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model_and_scalers(model_name: str):
    """Load and cache a model + its scalers.  Raises on missing artifacts."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    paths = _ARTIFACT_PATHS.get(model_name)
    if paths is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {sorted(_ARTIFACT_PATHS.keys())}"
        )

    model_path  = paths["model"]
    scaler_path = paths["scaler"]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[{model_name}] Model artifact not found: {model_path}"
        )
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"[{model_name}] Scaler artifact not found: {scaler_path}"
        )

    tf_model      = tf.keras.models.load_model(model_path)
    scaler_bundle = joblib.load(scaler_path)

    _model_cache[model_name] = (tf_model, scaler_bundle)
    return tf_model, scaler_bundle


def _validate_temporal(temporal_sequence):
    """Validate temporal input; returns numpy float32 array (5,4)."""
    try:
        arr = np.array(temporal_sequence, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"temporal_sequence could not be converted to array: {exc}")

    if arr.shape != (5, 4):
        raise ValueError(
            f"temporal_sequence must have shape (5, 4); got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("temporal_sequence contains NaN or infinite values.")

    return arr.astype(np.float32)


def _validate_spatial(latitude, longitude):
    """Validate spatial scalars; returns numpy float32 array (2,)."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"latitude/longitude must be numeric scalars: {exc}")

    if not np.isfinite(lat):
        raise ValueError(f"latitude is not finite: {lat}")
    if not np.isfinite(lon):
        raise ValueError(f"longitude is not finite: {lon}")

    return np.array([lat, lon], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    model_name: str,
    temporal_sequence,
    latitude: float = None,
    longitude: float = None,
) -> dict:
    """
    Predict next-year groundwater level (mbgl) from five historical annual
    observations using the specified Phase-2 model.

    Parameters
    ----------
    model_name : str
        One of 'lstm', 'gru', 'hybrid'.
    temporal_sequence : array-like, shape (5, 4)
        Five consecutive annual observations.
        Column order: [WL(mbgl), Rainfall, Temperature, Humidity]
    latitude : float, optional
        Required when model_name == 'hybrid'.
    longitude : float, optional
        Required when model_name == 'hybrid'.

    Returns
    -------
    dict
        {
          'model':           str,
          'predicted_wl':    float,   # original mbgl scale
          'unit':            'mbgl',
          'sequence_length': 5,
        }

    Raises
    ------
    ValueError
        On unknown model name or invalid inputs.
    FileNotFoundError
        If a required model/scaler artifact is missing.
    """
    model_name = model_name.strip().lower()

    # -- Validate temporal input --
    temporal_arr = _validate_temporal(temporal_sequence)   # (5,4)

    # -- Spatial validation only for hybrid --
    spatial_arr = None
    if model_name == "hybrid":
        if latitude is None or longitude is None:
            raise ValueError(
                "latitude and longitude are required for the 'hybrid' model."
            )
        spatial_arr = _validate_spatial(latitude, longitude)  # (2,)

    # -- Load model + scalers (cached after first call) --
    tf_model, scaler_bundle = _load_model_and_scalers(model_name)

    # -- Resolve scaler references --
    if model_name in ("lstm", "gru"):
        # phase2_scalers.pkl  →  {'feature_order': ..., 'scalers': {...}, ...}
        feature_scaler = scaler_bundle["scalers"]["feature_scaler"]
        target_scaler  = scaler_bundle["scalers"]["target_scaler"]
    else:
        # phase2_hybrid_scalers.pkl  →  {'scalers': {...}, ...}
        feature_scaler  = scaler_bundle["scalers"]["feature_scaler"]
        target_scaler   = scaler_bundle["scalers"]["target_scaler"]
        spatial_scaler  = scaler_bundle["scalers"]["spatial_scaler"]

    # -- Scale temporal input (flatten → scale → reshape) --
    temporal_2d     = temporal_arr.reshape(-1, 4)          # (5, 4) → (5, 4)
    temporal_scaled = feature_scaler.transform(temporal_2d).reshape(1, 5, 4)

    # -- Predict --
    if model_name in ("lstm", "gru"):
        pred_scaled = tf_model.predict(temporal_scaled, verbose=0)   # (1,1)
    else:
        spatial_2d     = spatial_arr.reshape(1, 2)
        spatial_scaled = spatial_scaler.transform(spatial_2d)        # (1,2)
        pred_scaled    = tf_model.predict(
            [temporal_scaled, spatial_scaled], verbose=0
        )                                                             # (1,1)

    # -- Inverse-transform to original mbgl scale --
    predicted_wl = float(target_scaler.inverse_transform(pred_scaled)[0, 0])

    return {
        "model":           model_name,
        "predicted_wl":    predicted_wl,
        "unit":            "mbgl",
        "sequence_length": 5,
    }


def get_supported_models() -> list:
    """Return the list of supported model names."""
    return sorted(_ARTIFACT_PATHS.keys())


def clear_model_cache():
    """Evict all cached models (useful for testing)."""
    _model_cache.clear()
