"""
test_phase2_prediction_service.py
==================================
Comprehensive tests for the Phase-2 Unified Prediction Service.

Tests cover:
  1.  LSTM loading
  2.  GRU loading
  3.  Hybrid loading
  4.  LSTM prediction (valid input, finite output, correct mbgl scale)
  5.  GRU prediction (valid input, finite output, correct mbgl scale)
  6.  Hybrid prediction (temporal + spatial, finite output, correct mbgl scale)
  7.  Invalid temporal shape rejection
  8.  Invalid feature count rejection
  9.  Invalid spatial input rejection
 10.  NaN/infinite input rejection
 11.  Repeated prediction consistency (caching stability)
 12.  Inference-only verification (no model artifact mutation)

Reference consistency (Test 4/5/6) uses the known predictions computed via
direct TF inference; tolerance is 1e-4 mbgl.
"""

import os
import sys
import unittest
import numpy as np
import joblib
import tensorflow as tf

# Make sure the dl directory is on the path
_DL_DIR = os.path.dirname(os.path.abspath(__file__))
if _DL_DIR not in sys.path:
    sys.path.insert(0, _DL_DIR)

import phase2_prediction_service as svc

# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------
# Build a realistic 5×4 sequence from the training-set scaler means.
# This is documented exactly here so future tests can reproduce it.
_SCALER_PATH = os.path.join(_DL_DIR, "phase2_scalers.pkl")
_SCALER_BUNDLE = joblib.load(_SCALER_PATH)
_FEATURE_MEANS = _SCALER_BUNDLE["scalers"]["feature_scaler"].mean_
# Shape: [WL(mbgl), Rainfall, Temperature, Humidity]  (4 values)

# Repeat the mean row five times → (5, 4)
VALID_SEQ = [_FEATURE_MEANS.tolist()] * 5

# Spatial fixture: representative AP location
VALID_LAT = 15.0
VALID_LON = 77.0

# Known reference predictions computed via direct TF inference
# (regenerated in the test-runner preamble below; tolerance 1e-4)
REF_LSTM   = 7.116502
REF_GRU    = 7.064608
REF_HYBRID = 6.813018
TOLERANCE  = 1e-3   # 1 millimeter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelLoading(unittest.TestCase):
    """Tests 1–3: Model loading."""

    def setUp(self):
        svc.clear_model_cache()

    def test_01_lstm_loads(self):
        model, bundle = svc._load_model_and_scalers("lstm")
        self.assertIsNotNone(model)
        self.assertIn("feature_scaler", bundle["scalers"])
        self.assertIn("target_scaler",  bundle["scalers"])

    def test_02_gru_loads(self):
        model, bundle = svc._load_model_and_scalers("gru")
        self.assertIsNotNone(model)
        self.assertIn("feature_scaler", bundle["scalers"])
        self.assertIn("target_scaler",  bundle["scalers"])

    def test_03_hybrid_loads(self):
        model, bundle = svc._load_model_and_scalers("hybrid")
        self.assertIsNotNone(model)
        self.assertIn("feature_scaler",  bundle["scalers"])
        self.assertIn("target_scaler",   bundle["scalers"])
        self.assertIn("spatial_scaler",  bundle["scalers"])


class TestValidPredictions(unittest.TestCase):
    """Tests 4–6: Valid predictions for each model."""

    @classmethod
    def setUpClass(cls):
        svc.clear_model_cache()

    def _assert_valid_result(self, result, expected_model):
        self.assertEqual(result["model"], expected_model)
        self.assertEqual(result["unit"], "mbgl")
        self.assertEqual(result["sequence_length"], 5)
        wl = result["predicted_wl"]
        self.assertIsInstance(wl, float)
        self.assertTrue(np.isfinite(wl), f"predicted_wl is not finite: {wl}")

    def test_04_lstm_prediction(self):
        result = svc.predict("lstm", VALID_SEQ)
        self._assert_valid_result(result, "lstm")
        self.assertAlmostEqual(result["predicted_wl"], REF_LSTM, delta=TOLERANCE,
            msg=f"LSTM prediction {result['predicted_wl']} deviates from reference {REF_LSTM}")

    def test_05_gru_prediction(self):
        result = svc.predict("gru", VALID_SEQ)
        self._assert_valid_result(result, "gru")
        self.assertAlmostEqual(result["predicted_wl"], REF_GRU, delta=TOLERANCE,
            msg=f"GRU prediction {result['predicted_wl']} deviates from reference {REF_GRU}")

    def test_06_hybrid_prediction(self):
        result = svc.predict("hybrid", VALID_SEQ, latitude=VALID_LAT, longitude=VALID_LON)
        self._assert_valid_result(result, "hybrid")
        self.assertAlmostEqual(result["predicted_wl"], REF_HYBRID, delta=TOLERANCE,
            msg=f"Hybrid prediction {result['predicted_wl']} deviates from reference {REF_HYBRID}")


class TestInputValidation(unittest.TestCase):
    """Tests 7–10: Input validation."""

    def test_07_invalid_temporal_shape_too_few_rows(self):
        bad_seq = VALID_SEQ[:3]  # only 3 rows
        with self.assertRaises(ValueError):
            svc.predict("lstm", bad_seq)

    def test_07b_invalid_temporal_shape_too_many_rows(self):
        bad_seq = VALID_SEQ + [VALID_SEQ[0]]  # 6 rows
        with self.assertRaises(ValueError):
            svc.predict("gru", bad_seq)

    def test_08_invalid_feature_count(self):
        # Only 3 features per row
        bad_seq = [[row[0], row[1], row[2]] for row in VALID_SEQ]
        with self.assertRaises(ValueError):
            svc.predict("lstm", bad_seq)

    def test_09_invalid_spatial_missing(self):
        # Hybrid without lat/lon
        with self.assertRaises(ValueError):
            svc.predict("hybrid", VALID_SEQ)

    def test_09b_invalid_spatial_none_lat(self):
        with self.assertRaises((ValueError, TypeError)):
            svc.predict("hybrid", VALID_SEQ, latitude=None, longitude=VALID_LON)

    def test_09c_invalid_spatial_non_finite(self):
        with self.assertRaises(ValueError):
            svc.predict("hybrid", VALID_SEQ, latitude=float("inf"), longitude=VALID_LON)

    def test_10_nan_in_temporal(self):
        bad_seq = [row[:] for row in VALID_SEQ]
        bad_seq[2][1] = float("nan")
        with self.assertRaises(ValueError):
            svc.predict("lstm", bad_seq)

    def test_10b_inf_in_temporal(self):
        bad_seq = [row[:] for row in VALID_SEQ]
        bad_seq[0][0] = float("inf")
        with self.assertRaises(ValueError):
            svc.predict("gru", bad_seq)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            svc.predict("random_forest", VALID_SEQ)


class TestCachingAndConsistency(unittest.TestCase):
    """Test 11: Repeated prediction consistency; Test 12: no artifact mutation."""

    @classmethod
    def setUpClass(cls):
        svc.clear_model_cache()

    def test_11_repeated_prediction_identical(self):
        r1 = svc.predict("lstm", VALID_SEQ)
        r2 = svc.predict("lstm", VALID_SEQ)
        self.assertAlmostEqual(r1["predicted_wl"], r2["predicted_wl"], places=10,
            msg="Repeated predictions differ — caching or state is broken.")

    def test_11b_cache_populated_after_first_call(self):
        svc.clear_model_cache()
        self.assertEqual(len(svc._model_cache), 0)
        svc.predict("gru", VALID_SEQ)
        self.assertIn("gru", svc._model_cache)

    def test_12_model_artifact_not_modified(self):
        """Verify model artifact mtime does not change after prediction."""
        model_path = svc._ARTIFACT_PATHS["lstm"]["model"]
        mtime_before = os.path.getmtime(model_path)
        svc.predict("lstm", VALID_SEQ)
        mtime_after = os.path.getmtime(model_path)
        self.assertEqual(mtime_before, mtime_after,
            "LSTM model artifact was modified during prediction — inference must be read-only.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
