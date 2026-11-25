"""
Unit Tests for Predictor Wrappers.

Tests:
- XGBoost binary prediction mapping (0 -> Normal, 1 -> Malicious)
- LightGBM category prediction dynamic class extraction
- Confidence score computations
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.inference.predictor import (
    XGBoostBinaryPredictor,
    LightGBMCategoryPredictor,
)


class TestPredictorWrappers(unittest.TestCase):
    """Test suite for predictor wrappers."""

    def setUp(self):
        self.dummy_X = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})

    def test_xgboost_predictor_maps_normal_and_malicious(self):
        """Test that XGBoost binary predictor maps 0 to Normal and 1 to Malicious."""
        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([0, 1])
        mock_xgb.predict_proba.return_value = np.array([
            [0.95, 0.05],
            [0.02, 0.98],
        ])

        predictor = XGBoostBinaryPredictor(mock_xgb)
        raw_preds, labels, confidences, probas = predictor.predict(self.dummy_X)

        self.assertEqual(labels[0], "Normal")
        self.assertEqual(labels[1], "Malicious")
        self.assertAlmostEqual(confidences[0], 0.95)
        self.assertAlmostEqual(confidences[1], 0.98)
        self.assertEqual(raw_preds[0], 0)
        self.assertEqual(raw_preds[1], 1)

    def test_lightgbm_predictor_dynamic_classes_and_mapping(self):
        """Test that LightGBM category predictor dynamically inspects model classes and maps categories."""
        mock_lgb = MagicMock()
        mock_lgb.classes_ = np.array(["DDoS", "DoS", "Reconnaissance", "Theft"])
        mock_lgb.predict.return_value = np.array(["DDoS", "Theft"])
        mock_lgb.predict_proba.return_value = np.array([
            [0.96, 0.02, 0.01, 0.01],
            [0.05, 0.05, 0.02, 0.88],
        ])

        predictor = LightGBMCategoryPredictor(mock_lgb)
        self.assertEqual(predictor.get_classes(), ["DDoS", "DoS", "Reconnaissance", "Theft"])

        categories, confidences, probas = predictor.predict(self.dummy_X)
        self.assertEqual(categories[0], "DDoS")
        self.assertEqual(categories[1], "Theft")
        self.assertAlmostEqual(confidences[0], 0.96)
        self.assertAlmostEqual(confidences[1], 0.88)


if __name__ == "__main__":
    unittest.main()
