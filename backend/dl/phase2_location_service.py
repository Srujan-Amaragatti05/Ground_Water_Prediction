"""
backend/dl/phase2_location_service.py
======================================
Step-G: Location-Based Historical Sequence Retrieval Service.

Responsibilities
----------------
1. Load and index the Phase-2 dataset once (module-level singleton).
2. Given a LOCATION_ID, locate its annual observations using the SAME
   aggregation methodology as phase2_lstm.preprocess_data().
3. Select the LATEST valid five-consecutive-year window.
4. Return the 5×4 temporal sequence + spatial coordinates, ready to be
   passed to phase2_prediction_service.predict().

LOCATION_ID definition (authoritative — matches phase2_lstm.py exactly)
------------------------------------------------------------------------
    STATE_UT _ DISTRICT _ BLOCK _ VILLAGE _ LATITUDE _ LONGITUDE

Annual aggregation (authoritative — matches phase2_lstm.py exactly)
----------------------------------------------------------------------
    features = ['WL(mbgl)', 'Rainfall', 'Temperature', 'Humidity']
    annual_df = df.groupby(['LOCATION_ID', 'YEAR'])[features].mean()

Consecutiveness rule (authoritative — matches phase2_lstm.py exactly)
----------------------------------------------------------------------
    A window of 6 annual rows at indices i..i+5 is valid iff
        years[i+5] - years[i] == 5
    (all distinct annual years, no gaps)

Latest valid window rule
------------------------
    Scan from the RIGHT of the sorted year list.
    The first (rightmost) valid window found is used.
    This gives the latest possible five-consecutive-year input for prediction.

This module is INFERENCE-ONLY and READ-ONLY with respect to the dataset.
It never fits scalers, trains models, or writes files.
"""

from __future__ import annotations

import os
import threading
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FEATURES   = ['WL(mbgl)', 'Rainfall', 'Temperature', 'Humidity']
_SEQ_LEN    = 5

# Dataset path — fixed, relative to this file's location (backend/dl/)
_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "fixed_real_dataset.csv",
)

# ---------------------------------------------------------------------------
# Module-level singleton index
#   _annual_index  : dict[location_id -> DataFrame(YEAR, WL, Rain, Temp, Hum)]
#   _spatial_index : dict[location_id -> (lat, lon)]
# ---------------------------------------------------------------------------
_annual_index: dict  = {}
_spatial_index: dict = {}
_index_loaded: bool  = False
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_index(csv_path: str) -> None:
    """Load the dataset and build the two lookup indexes."""
    global _annual_index, _spatial_index, _index_loaded

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Phase-2 dataset not found at: {csv_path}. "
            "Cannot perform location-based retrieval."
        )

    df = pd.read_csv(csv_path)

    # --- replicate phase2_lstm.preprocess_data() setup exactly ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['YEAR'] = df['Date'].dt.year

    df['LOCATION_ID'] = (
        df['STATE_UT'].astype(str) + '_' +
        df['DISTRICT'].astype(str) + '_' +
        df['BLOCK'].astype(str)    + '_' +
        df['VILLAGE'].astype(str)  + '_' +
        df['LATITUDE'].astype(str) + '_' +
        df['LONGITUDE'].astype(str)
    )

    # Annual mean aggregation — identical to preprocess_data()
    annual = (
        df.groupby(['LOCATION_ID', 'YEAR'])[_FEATURES]
        .mean()
        .reset_index()
        .sort_values(['LOCATION_ID', 'YEAR'])
    )

    # Spatial coordinates: use the first row per LOCATION_ID
    spatial = (
        df.drop_duplicates(subset='LOCATION_ID')
        .set_index('LOCATION_ID')[['LATITUDE', 'LONGITUDE']]
    )

    # Build per-location annual DataFrames
    new_annual: dict = {}
    for loc_id, grp in annual.groupby('LOCATION_ID'):
        new_annual[loc_id] = grp.sort_values('YEAR').reset_index(drop=True)

    _annual_index  = new_annual
    _spatial_index = spatial.to_dict('index')   # {loc_id: {'LATITUDE': ..., 'LONGITUDE': ...}}
    _index_loaded  = True


def _ensure_index() -> None:
    """Thread-safe lazy loading of the dataset index."""
    global _index_loaded
    if _index_loaded:
        return
    with _lock:
        if not _index_loaded:   # double-checked
            _build_index(_DATASET_PATH)


def _find_latest_consecutive_window(years: np.ndarray) -> int | None:
    """
    Find the START index of the latest valid five-consecutive-year window.

    A window starting at index i (0-based) uses rows i..i+4 as inputs and
    requires that years[i+4] - years[i] == 4 (all five years consecutive).

    This mirrors the consecutiveness check in phase2_lstm.preprocess_data():
        if years[i+5] - years[i] == 5:   (6-row window, last is target)

    For retrieval we want a 5-row input window (no target row here):
        if years[i+4] - years[i] == 4

    Returns the start index of the latest such window, or None.
    """
    n = len(years)
    if n < _SEQ_LEN:
        return None
    # Scan right-to-left to find the latest valid window
    for i in range(n - _SEQ_LEN, -1, -1):
        if years[i + _SEQ_LEN - 1] - years[i] == _SEQ_LEN - 1:
            return i
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LocationSequenceResult:
    """Holds a retrieved location sequence and associated metadata."""
    __slots__ = (
        'location_id', 'temporal_sequence', 'years',
        'sequence_start_year', 'sequence_end_year', 'prediction_year',
        'latitude', 'longitude',
    )

    def __init__(
        self,
        location_id: str,
        temporal_sequence: list,
        years: list,
        latitude: float,
        longitude: float,
    ):
        self.location_id        = location_id
        self.temporal_sequence  = temporal_sequence        # list[list[float]], shape (5,4)
        self.years              = years                    # list[int], len 5
        self.sequence_start_year = years[0]
        self.sequence_end_year   = years[-1]
        self.prediction_year     = years[-1] + 1
        self.latitude            = latitude
        self.longitude           = longitude


def get_location_sequence(location_id: str) -> LocationSequenceResult:
    """
    Retrieve the latest valid five-consecutive-year sequence for a location.

    Parameters
    ----------
    location_id : str
        Must match the Phase-2 LOCATION_ID format exactly.

    Returns
    -------
    LocationSequenceResult

    Raises
    ------
    KeyError
        If the location_id is not found in the dataset.
    ValueError
        If the location exists but has no valid five-consecutive-year window.
    FileNotFoundError
        If the dataset file is missing.
    """
    _ensure_index()

    # --- look up location ---
    if location_id not in _annual_index:
        raise KeyError(
            f"Location '{location_id}' was not found in the Phase-2 dataset."
        )

    grp   = _annual_index[location_id]
    years = grp['YEAR'].values          # sorted ascending

    # --- find latest valid five-consecutive-year window ---
    start_idx = _find_latest_consecutive_window(years)
    if start_idx is None:
        raise ValueError(
            f"Location '{location_id}' has {len(years)} annual observation(s) "
            f"(years: {years.tolist()}) but no valid five-consecutive-year "
            "window exists."
        )

    window      = grp.iloc[start_idx : start_idx + _SEQ_LEN]
    window_years= window['YEAR'].tolist()
    temporal    = window[_FEATURES].values.tolist()   # list[list[float]], shape (5,4)

    # --- spatial coordinates ---
    sp          = _spatial_index[location_id]
    lat         = float(sp['LATITUDE'])
    lon         = float(sp['LONGITUDE'])

    return LocationSequenceResult(
        location_id=location_id,
        temporal_sequence=temporal,
        years=window_years,
        latitude=lat,
        longitude=lon,
    )


def get_all_location_ids() -> list:
    """Return all known LOCATION_IDs (useful for discovery/testing)."""
    _ensure_index()
    return list(_annual_index.keys())


def reload_index() -> None:
    """Force a reload of the dataset index (e.g. after dataset update)."""
    global _index_loaded
    with _lock:
        _index_loaded = False
        _build_index(_DATASET_PATH)
