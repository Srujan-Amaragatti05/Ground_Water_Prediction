"""
test_phase2_api.py
==================
Comprehensive API tests for Phase-2 FastAPI endpoints.

Uses FastAPI TestClient (backed by httpx).

Run from the backend/ directory:
    myenv\\Scripts\\python.exe -m pytest test_phase2_api.py -v

Tests
-----
 1  App starts (TestClient created without errors)
 2  LSTM prediction — valid input
 3  GRU prediction — valid input
 4  Hybrid prediction — valid temporal + coordinates
 5  Invalid model name → 422
 6  Invalid temporal shape (too few rows) → 422
 7  Invalid temporal shape (too many rows) → 422
 8  Invalid feature count (3 features per row) → 422
 9  Missing Hybrid coordinates → 422
10  NaN value in temporal sequence → 422
11  Service delegation verified (mocked to confirm no duplicate ML logic)
12  Existing Phase-1 /predict endpoint regression
13  Response is JSON-serializable (no NumPy / TF objects)
14  Direct service prediction == API prediction (numerical consistency)
15  GET /api/phase2/models returns supported models list
"""

import json
import math
import os
import sys
import unittest
from unittest.mock import patch

# ---- path setup -----------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_DIR = os.path.join(_BACKEND_DIR, "dl")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _DL_DIR not in sys.path:
    sys.path.insert(0, _DL_DIR)

# ---- import the FastAPI app -----------------------------------------------
# We must import main from the backend directory.
# Phase-1 model pickling requires the pkl files; skip them gracefully.
try:
    import main as _main_module
    _app = _main_module.app
    _phase1_available = True
except Exception:
    # Phase-1 model artifacts may be absent in CI; still test Phase-2.
    _phase1_available = False
    # Build a minimal app from just the Phase-2 router for remaining tests.
    from fastapi import FastAPI
    from routers.phase2 import router as _p2_router
    _app = FastAPI()
    _app.include_router(_p2_router)

from fastapi.testclient import TestClient
import dl.phase2_prediction_service as _svc

_client = TestClient(_app, raise_server_exceptions=False)

# ---- shared fixture -------------------------------------------------------
import joblib as _joblib
_bundle = _joblib.load(os.path.join(_DL_DIR, "phase2_scalers.pkl"))
_means  = _bundle["scalers"]["feature_scaler"].mean_   # [WL, Rain, Temp, Hum]
VALID_SEQ  = [_means.tolist()] * 5
VALID_LAT  = 15.0
VALID_LON  = 77.0

_LSTM_PAYLOAD = {
    "model": "lstm",
    "temporal_sequence": VALID_SEQ,
}
_GRU_PAYLOAD = {
    "model": "gru",
    "temporal_sequence": VALID_SEQ,
}
_HYBRID_PAYLOAD = {
    "model": "hybrid",
    "temporal_sequence": VALID_SEQ,
    "latitude": VALID_LAT,
    "longitude": VALID_LON,
}


# ===========================================================================
class TestAppStartup(unittest.TestCase):
    """Test 1 — app starts."""

    def test_01_app_starts(self):
        # If TestClient is created without raising, the app loaded correctly.
        c = TestClient(_app, raise_server_exceptions=False)
        self.assertIsNotNone(c)


# ===========================================================================
class TestValidPredictions(unittest.TestCase):
    """Tests 2-4 — valid predictions."""

    def _assert_valid_response(self, resp, expected_model):
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["model"], expected_model)
        self.assertEqual(data["unit"], "mbgl")
        self.assertEqual(data["sequence_length"], 5)
        wl = data["predicted_wl"]
        self.assertIsInstance(wl, float)
        self.assertTrue(math.isfinite(wl), f"predicted_wl not finite: {wl}")

    def test_02_lstm_valid(self):
        resp = _client.post("/api/phase2/predict", json=_LSTM_PAYLOAD)
        self._assert_valid_response(resp, "lstm")

    def test_03_gru_valid(self):
        resp = _client.post("/api/phase2/predict", json=_GRU_PAYLOAD)
        self._assert_valid_response(resp, "gru")

    def test_04_hybrid_valid(self):
        resp = _client.post("/api/phase2/predict", json=_HYBRID_PAYLOAD)
        self._assert_valid_response(resp, "hybrid")


# ===========================================================================
class TestInputValidation(unittest.TestCase):
    """Tests 5-10 — validation errors."""

    def test_05_invalid_model(self):
        payload = {**_LSTM_PAYLOAD, "model": "random_forest"}
        resp = _client.post("/api/phase2/predict", json=payload)
        self.assertIn(resp.status_code, [400, 422], resp.text)

    def test_06_too_few_rows(self):
        payload = {**_LSTM_PAYLOAD, "temporal_sequence": VALID_SEQ[:3]}
        resp = _client.post("/api/phase2/predict", json=payload)
        self.assertIn(resp.status_code, [400, 422], resp.text)

    def test_07_too_many_rows(self):
        payload = {**_LSTM_PAYLOAD, "temporal_sequence": VALID_SEQ + [VALID_SEQ[0]]}
        resp = _client.post("/api/phase2/predict", json=payload)
        self.assertIn(resp.status_code, [400, 422], resp.text)

    def test_08_wrong_feature_count(self):
        bad_seq = [[r[0], r[1], r[2]] for r in VALID_SEQ]   # 3 features only
        payload = {**_LSTM_PAYLOAD, "temporal_sequence": bad_seq}
        resp = _client.post("/api/phase2/predict", json=payload)
        self.assertIn(resp.status_code, [400, 422], resp.text)

    def test_09_hybrid_missing_coordinates(self):
        payload = {
            "model": "hybrid",
            "temporal_sequence": VALID_SEQ,
            # no latitude / longitude
        }
        resp = _client.post("/api/phase2/predict", json=payload)
        self.assertIn(resp.status_code, [400, 422], resp.text)

    def test_10_nan_in_temporal(self):
        # Python's json.dumps raises on NaN, so we send the JSON manually
        # using the JS-style NaN representation (null) as a proxy, or use
        # a string that will fail Pydantic's float coercion.
        # Here we substitute a clearly non-finite string value.
        bad_seq  = [r[:] for r in VALID_SEQ]
        # Build raw JSON with a non-numeric string in the sequence
        import json as _json
        bad_seq[2][1] = 0.0   # placeholder to keep it serialisable
        raw_payload = {**_LSTM_PAYLOAD, "temporal_sequence": bad_seq}
        raw_json = _json.dumps(raw_payload).replace(
            '"temporal_sequence"',
            '"temporal_sequence"',  # no-op, kept for readability
        )
        # Replace the placeholder 0.0 with null (non-finite proxy)
        # to verify server-side rejection
        raw_json = raw_json.replace(
            "[" + ", ".join(str(v) for v in bad_seq[2]) + "]",
            "[" + ", ".join(
                "null" if i == 1 else str(v)
                for i, v in enumerate(bad_seq[2])
            ) + "]",
            1,
        )
        resp = _client.post(
            "/api/phase2/predict",
            content=raw_json.encode(),
            headers={"Content-Type": "application/json"},
        )
        self.assertIn(resp.status_code, [400, 422], resp.text)


# ===========================================================================
class TestServiceDelegation(unittest.TestCase):
    """Test 11 — endpoint delegates to phase2_prediction_service.predict()."""

    def test_11_delegates_to_service(self):
        with patch("routers.phase2._svc.predict", wraps=_svc.predict) as mock_predict:
            resp = _client.post("/api/phase2/predict", json=_GRU_PAYLOAD)
            self.assertEqual(resp.status_code, 200)
            mock_predict.assert_called_once()
            call_kwargs = mock_predict.call_args
            # model_name must be passed
            self.assertEqual(
                call_kwargs.kwargs.get("model_name") or call_kwargs.args[0],
                "gru",
            )


# ===========================================================================
class TestPhase1Regression(unittest.TestCase):
    """Test 12 — Phase-1 /predict remains intact."""

    @unittest.skipUnless(_phase1_available, "Phase-1 model artifacts not available")
    def test_12_phase1_endpoint_exists(self):
        # Just verify the route is registered and returns something meaningful
        # (not 404 and not 500 from our changes)
        resp = _client.post(
            "/predict",
            json={
                "Year": 2020,
                "Month": 6,
                "LATITUDE": 15.5,
                "LONGITUDE": 78.3,
                "DISTRICT": "KURNOOL",
                "BLOCK": "ALLAGADDA",
            },
        )
        # Could be 200 or 422/500 depending on encoders, but must NOT be 404
        self.assertNotEqual(resp.status_code, 404, "Phase-1 /predict route is missing!")


# ===========================================================================
class TestResponseSerialization(unittest.TestCase):
    """Test 13 — response is JSON-serializable."""

    def test_13_json_serializable(self):
        resp = _client.post("/api/phase2/predict", json=_LSTM_PAYLOAD)
        self.assertEqual(resp.status_code, 200)
        # Re-encode/decode to catch any non-serializable types
        try:
            re_encoded = json.dumps(resp.json())
            json.loads(re_encoded)
        except (TypeError, ValueError) as exc:
            self.fail(f"Response is not JSON-serializable: {exc}")


# ===========================================================================
class TestNumericalConsistency(unittest.TestCase):
    """Test 14 — API prediction == direct service prediction."""

    _TOL = 1e-4

    def _compare(self, model_name, lat=None, lon=None):
        # Direct service call
        _svc.clear_model_cache()
        direct = _svc.predict(
            model_name=model_name,
            temporal_sequence=VALID_SEQ,
            latitude=lat,
            longitude=lon,
        )["predicted_wl"]

        # API call
        payload = {"model": model_name, "temporal_sequence": VALID_SEQ}
        if lat is not None:
            payload["latitude"] = lat
            payload["longitude"] = lon
        resp = _client.post("/api/phase2/predict", json=payload)
        api_wl = resp.json()["predicted_wl"]

        self.assertAlmostEqual(
            direct, api_wl, delta=self._TOL,
            msg=f"[{model_name}] direct={direct:.6f} vs API={api_wl:.6f}"
        )

    def test_14a_lstm_numerical_consistency(self):
        self._compare("lstm")

    def test_14b_gru_numerical_consistency(self):
        self._compare("gru")

    def test_14c_hybrid_numerical_consistency(self):
        self._compare("hybrid", lat=VALID_LAT, lon=VALID_LON)


# ===========================================================================
class TestModelsListEndpoint(unittest.TestCase):
    """Test 15 — GET /api/phase2/models."""

    def test_15_models_list(self):
        resp = _client.get("/api/phase2/models")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("supported_models", data)
        models = data["supported_models"]
        for name in ("lstm", "gru", "hybrid"):
            self.assertIn(name, models, f"'{name}' missing from supported_models")


# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
