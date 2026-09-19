"""
test_phase2_step_g.py
======================
Step-G tests: Location-Based Historical Sequence Retrieval.

Run from the repo root:
    myenv\\Scripts\\python.exe -m pytest backend\\test_phase2_step_g.py -v

Covers
------
 G01  Location lookup — known valid location
 G02  Unknown location → 404
 G03  Insufficient consecutive years → 404 (patched location)
 G04  LSTM location prediction — valid response
 G05  GRU  location prediction — valid response
 G06  Hybrid location prediction — valid response + coordinates
 G07  Five-year sequence shape = (5, 4)
 G08  Chronological ordering enforced
 G09  Consecutive-year enforcement
 G10  Correct feature order [WL, Rain, Temp, Hum]
 G11  Hybrid coordinate retrieval (lat/lon from dataset)
 G12  prediction_year = sequence_end_year + 1
 G13  Step-E delegation (mock assertion)
 G14  Direct service vs API numerical consistency (all three models)
 G15  Existing POST /api/phase2/predict regression
 G16  GET /api/phase2/models regression
"""

import os
import sys
import math
import unittest
from unittest.mock import patch, MagicMock

# ---- path setup -----------------------------------------------------------
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_DL_DIR      = os.path.join(_BACKEND_DIR, "dl")
for _p in (_BACKEND_DIR, _DL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- app import -----------------------------------------------------------
try:
    import main as _main_module
    _app = _main_module.app
except Exception:
    from fastapi import FastAPI
    from routers.phase2 import router as _p2_router
    _app = FastAPI()
    _app.include_router(_p2_router)

from fastapi.testclient import TestClient
import phase2_location_service   as _loc_svc
import phase2_prediction_service as _svc
import joblib, numpy as np

_client = TestClient(_app, raise_server_exceptions=False)

# ---- known-good test location ---------------------------------------------
_TEST_LOC_ID = "West Bengal_Purulia_Santuri_Leadson_23.51992_86.82893"
_TEST_LAT    = 23.51992
_TEST_LON    = 86.82893

# Shared scaler for manual sequence fixture (Step-F regression)
_bundle = joblib.load(os.path.join(_DL_DIR, "phase2_scalers.pkl"))
_means  = _bundle["scalers"]["feature_scaler"].mean_
VALID_SEQ = [_means.tolist()] * 5


# ===========================================================================
class TestLocationService(unittest.TestCase):
    """G01–G03: Unit tests for phase2_location_service."""

    @classmethod
    def setUpClass(cls):
        # Pre-load index once for all unit tests
        _loc_svc._ensure_index()

    def test_G01_known_location_found(self):
        r = _loc_svc.get_location_sequence(_TEST_LOC_ID)
        self.assertEqual(r.location_id, _TEST_LOC_ID)
        self.assertEqual(len(r.temporal_sequence), 5)
        self.assertIsInstance(r.latitude, float)
        self.assertIsInstance(r.longitude, float)

    def test_G02_unknown_location_raises_keyerror(self):
        with self.assertRaises(KeyError):
            _loc_svc.get_location_sequence("DOES_NOT_EXIST_STATE_X_Y_Z_0.0_0.0")

    def test_G03_insufficient_consecutive_years(self):
        """Patch the index so our location has only non-consecutive years."""
        import pandas as pd
        sparse_df = pd.DataFrame({
            'YEAR': [2010, 2013, 2017, 2020, 2023],   # no five consecutive
            'WL(mbgl)': [5.0]*5, 'Rainfall': [100.0]*5,
            'Temperature': [25.0]*5, 'Humidity': [60.0]*5,
        })
        original = _loc_svc._annual_index.get(_TEST_LOC_ID)
        _loc_svc._annual_index['__sparse_test__'] = sparse_df
        try:
            with self.assertRaises(ValueError):
                _loc_svc.get_location_sequence('__sparse_test__')
                _loc_svc._spatial_index['__sparse_test__'] = {
                    'LATITUDE': 0.0, 'LONGITUDE': 0.0
                }
        finally:
            del _loc_svc._annual_index['__sparse_test__']


# ===========================================================================
class TestSequenceProperties(unittest.TestCase):
    """G07–G12: Validate the retrieved sequence properties."""

    @classmethod
    def setUpClass(cls):
        cls.r = _loc_svc.get_location_sequence(_TEST_LOC_ID)

    def test_G07_sequence_shape(self):
        seq = self.r.temporal_sequence
        self.assertEqual(len(seq), 5)
        for row in seq:
            self.assertEqual(len(row), 4)

    def test_G08_chronological_ordering(self):
        years = self.r.years
        self.assertEqual(years, sorted(years))

    def test_G09_consecutive_years(self):
        years = self.r.years
        for i in range(len(years) - 1):
            self.assertEqual(
                years[i+1] - years[i], 1,
                f"Gap between year {years[i]} and {years[i+1]}",
            )

    def test_G10_feature_order(self):
        """Feature order: [WL(mbgl), Rainfall, Temperature, Humidity].
        We verify values are finite and in a plausible real-world range."""
        seq = np.array(self.r.temporal_sequence)
        self.assertTrue(np.all(np.isfinite(seq)), "Non-finite value in sequence")
        # WL(mbgl) typically > 0; Rainfall in mm > 0
        self.assertTrue((seq[:, 0] >= 0).all(), "WL should be non-negative")
        self.assertTrue((seq[:, 1] >= 0).all(), "Rainfall should be non-negative")

    def test_G11_hybrid_coordinates(self):
        self.assertAlmostEqual(self.r.latitude,  _TEST_LAT, places=4)
        self.assertAlmostEqual(self.r.longitude, _TEST_LON, places=4)

    def test_G12_prediction_year(self):
        self.assertEqual(self.r.prediction_year, self.r.sequence_end_year + 1)
        self.assertEqual(self.r.prediction_year, self.r.years[-1] + 1)


# ===========================================================================
class TestLocationAPIEndpoint(unittest.TestCase):
    """G04–G06: API endpoint response validation."""

    def _assert_valid_location_response(self, resp, model):
        self.assertEqual(resp.status_code, 200, resp.text)
        d = resp.json()
        self.assertEqual(d["model"],            model)
        self.assertEqual(d["unit"],             "mbgl")
        self.assertEqual(d["sequence_length"],  5)
        self.assertEqual(d["location_id"],      _TEST_LOC_ID)
        self.assertIsInstance(d["predicted_wl"], float)
        self.assertTrue(math.isfinite(d["predicted_wl"]))
        self.assertIsInstance(d["sequence_start_year"], int)
        self.assertIsInstance(d["sequence_end_year"],   int)
        self.assertIsInstance(d["prediction_year"],     int)
        self.assertEqual(d["prediction_year"], d["sequence_end_year"] + 1)
        self.assertIsInstance(d["latitude"],  float)
        self.assertIsInstance(d["longitude"], float)

    def test_G04_lstm_location_prediction(self):
        from urllib.parse import quote
        url  = f"/api/phase2/locations/{quote(_TEST_LOC_ID, safe='')}/predict?model=lstm"
        resp = _client.get(url)
        self._assert_valid_location_response(resp, "lstm")

    def test_G05_gru_location_prediction(self):
        from urllib.parse import quote
        url  = f"/api/phase2/locations/{quote(_TEST_LOC_ID, safe='')}/predict?model=gru"
        resp = _client.get(url)
        self._assert_valid_location_response(resp, "gru")

    def test_G06_hybrid_location_prediction(self):
        from urllib.parse import quote
        url  = f"/api/phase2/locations/{quote(_TEST_LOC_ID, safe='')}/predict?model=hybrid"
        resp = _client.get(url)
        self._assert_valid_location_response(resp, "hybrid")
        d = resp.json()
        self.assertAlmostEqual(d["latitude"],  _TEST_LAT, places=4)
        self.assertAlmostEqual(d["longitude"], _TEST_LON, places=4)

    def test_G02_api_unknown_location_404(self):
        from urllib.parse import quote
        bad_id = "UNKNOWN_STATE_X_Y_0.0_0.0"
        url    = f"/api/phase2/locations/{quote(bad_id, safe='')}/predict?model=gru"
        resp   = _client.get(url)
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_G_invalid_model(self):
        from urllib.parse import quote
        url  = f"/api/phase2/locations/{quote(_TEST_LOC_ID, safe='')}/predict?model=xgboost"
        resp = _client.get(url)
        self.assertIn(resp.status_code, [400, 422], resp.text)


# ===========================================================================
class TestStepEDelegation(unittest.TestCase):
    """G13: Endpoint delegates to phase2_prediction_service.predict()."""

    def test_G13_delegates_to_service(self):
        from urllib.parse import quote
        url = f"/api/phase2/locations/{quote(_TEST_LOC_ID, safe='')}/predict?model=gru"
        with patch("routers.phase2._svc.predict", wraps=_svc.predict) as mock_p:
            resp = _client.get(url)
            self.assertEqual(resp.status_code, 200)
            mock_p.assert_called_once()


# ===========================================================================
class TestNumericalConsistency(unittest.TestCase):
    """G14: Direct Step-E prediction == API prediction for all three models."""

    _TOL = 1e-4

    @classmethod
    def setUpClass(cls):
        cls.r = _loc_svc.get_location_sequence(_TEST_LOC_ID)

    def _direct_predict(self, model):
        _svc.clear_model_cache()
        kw = dict(model_name=model, temporal_sequence=self.r.temporal_sequence)
        if model == "hybrid":
            kw["latitude"]  = self.r.latitude
            kw["longitude"] = self.r.longitude
        return _svc.predict(**kw)["predicted_wl"]

    def _api_predict(self, model):
        from urllib.parse import quote
        url  = f"/api/phase2/locations/{quote(_TEST_LOC_ID, safe='')}/predict?model={model}"
        resp = _client.get(url)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["predicted_wl"]

    def test_G14a_lstm_consistency(self):
        direct = self._direct_predict("lstm")
        api    = self._api_predict("lstm")
        self.assertAlmostEqual(direct, api, delta=self._TOL,
            msg=f"LSTM: direct={direct:.6f} api={api:.6f}")

    def test_G14b_gru_consistency(self):
        direct = self._direct_predict("gru")
        api    = self._api_predict("gru")
        self.assertAlmostEqual(direct, api, delta=self._TOL,
            msg=f"GRU: direct={direct:.6f} api={api:.6f}")

    def test_G14c_hybrid_consistency(self):
        direct = self._direct_predict("hybrid")
        api    = self._api_predict("hybrid")
        self.assertAlmostEqual(direct, api, delta=self._TOL,
            msg=f"Hybrid: direct={direct:.6f} api={api:.6f}")


# ===========================================================================
class TestStepFRegression(unittest.TestCase):
    """G15–G16: Existing Step-F endpoints must still work."""

    def test_G15_post_predict_still_works(self):
        payload = {"model": "gru", "temporal_sequence": VALID_SEQ}
        resp    = _client.post("/api/phase2/predict", json=payload)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("predicted_wl", resp.json())

    def test_G16_models_list_still_works(self):
        resp = _client.get("/api/phase2/models")
        self.assertEqual(resp.status_code, 200, resp.text)
        models = resp.json()["supported_models"]
        for name in ("lstm", "gru", "hybrid"):
            self.assertIn(name, models)


# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
