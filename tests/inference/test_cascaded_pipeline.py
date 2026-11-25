"""
Unit Tests for CascadedInferencePipeline.

Tests:
1. XGBoost correctly identifies Normal traffic.
2. XGBoost correctly identifies Malicious traffic.
3. LightGBM is skipped for Normal predictions.
4. LightGBM executes for Malicious predictions.
5. Attack categories are mapped correctly.
6. Batch inference works correctly.
7. End-to-end integration with trained artifacts (if present).
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from config.settings import CANONICAL_22_FEATURES, VAL_V2_CLEAN_PARQUET
from src.inference.inference_pipeline import CascadedInferencePipeline
from src.inference.predictor import (
    XGBoostBinaryPredictor,
    LightGBMCategoryPredictor,
)
from src.inference.preprocessing import InferencePreprocessor


class TestCascadedInferencePipeline(unittest.TestCase):
    """Test suite verifying cascaded execution logic."""

    def setUp(self):
        self.preprocessor = InferencePreprocessor()

        # Sample dummy canonical record
        self.sample_record = {col: 1.0 for col in CANONICAL_22_FEATURES}
        self.sample_record["proto"] = "tcp"
        self.sample_record["flgs"] = "e"
        self.sample_record["state"] = "CON"

    def test_xgboost_identifies_normal_traffic_and_skips_lightgbm(self):
        """
        Requirements 1 & 3:
        - XGBoost correctly identifies Normal traffic.
        - LightGBM is SKIPPED for Normal predictions.
        """
        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([0])
        mock_xgb.predict_proba.return_value = np.array([[0.985, 0.015]])

        mock_lgb = MagicMock()
        mock_lgb.classes_ = np.array(["DDoS", "DoS", "Reconnaissance", "Theft"])

        pipeline = CascadedInferencePipeline.from_models(
            xgb_model=mock_xgb,
            lgb_model=mock_lgb,
            preprocessor=self.preprocessor,
        )

        result = pipeline.predict_single(self.sample_record)

        # Verify output structure for Normal flow
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["binary_prediction"], "Normal")
        self.assertFalse(result["attack_detected"])
        self.assertIsNone(result["attack_category"])
        self.assertAlmostEqual(result["binary_confidence"], 0.985, places=3)
        self.assertIsNone(result["category_confidence"])

        # Crucial check: LightGBM was never called!
        mock_lgb.predict.assert_not_called()
        mock_lgb.predict_proba.assert_not_called()

    def test_xgboost_identifies_malicious_traffic_and_executes_lightgbm(self):
        """
        Requirements 2, 4 & 5:
        - XGBoost correctly identifies Malicious traffic.
        - LightGBM executes for Malicious predictions.
        - Attack categories are mapped correctly.
        """
        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([1])
        mock_xgb.predict_proba.return_value = np.array([[0.01, 0.99]])

        mock_lgb = MagicMock()
        mock_lgb.classes_ = np.array(["DDoS", "DoS", "Reconnaissance", "Theft"])
        mock_lgb.predict.return_value = np.array(["DDoS"])
        mock_lgb.predict_proba.return_value = np.array([[0.95, 0.03, 0.01, 0.01]])

        pipeline = CascadedInferencePipeline.from_models(
            xgb_model=mock_xgb,
            lgb_model=mock_lgb,
            preprocessor=self.preprocessor,
        )

        result = pipeline.predict_single(self.sample_record)

        # Verify output structure for Malicious flow
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["binary_prediction"], "Malicious")
        self.assertTrue(result["attack_detected"])
        self.assertEqual(result["attack_category"], "DDoS")
        self.assertAlmostEqual(result["binary_confidence"], 0.99, places=2)
        self.assertAlmostEqual(result["category_confidence"], 0.95, places=2)

        # Crucial check: LightGBM was called exactly once
        self.assertEqual(mock_lgb.predict.call_count, 1)

    def test_batch_inference_cascaded_efficiency(self):
        """
        Requirement 10:
        Batch inference works correctly:
        - 3 records: 1 Normal, 2 Malicious
        - XGBoost runs on all 3
        - LightGBM runs ONLY on the 2 Malicious records
        """
        mock_xgb = MagicMock()
        # Predictions: Record 0: Normal (0), Record 1: Malicious (1), Record 2: Normal (0)
        mock_xgb.predict.return_value = np.array([0, 1, 0])
        mock_xgb.predict_proba.return_value = np.array([
            [0.96, 0.04],
            [0.02, 0.98],
            [0.99, 0.01],
        ])

        mock_lgb = MagicMock()
        mock_lgb.classes_ = np.array(["DDoS", "DoS", "Reconnaissance", "Theft"])
        mock_lgb.predict.return_value = np.array(["Reconnaissance"])
        mock_lgb.predict_proba.return_value = np.array([[0.05, 0.05, 0.85, 0.05]])

        pipeline = CascadedInferencePipeline.from_models(
            xgb_model=mock_xgb,
            lgb_model=mock_lgb,
            preprocessor=self.preprocessor,
        )

        batch_records = [self.sample_record, self.sample_record, self.sample_record]
        results = pipeline.predict_batch(batch_records)

        self.assertEqual(len(results), 3)

        # Record 0: Normal
        self.assertEqual(results[0]["binary_prediction"], "Normal")
        self.assertFalse(results[0]["attack_detected"])
        self.assertIsNone(results[0]["attack_category"])

        # Record 1: Malicious
        self.assertEqual(results[1]["binary_prediction"], "Malicious")
        self.assertTrue(results[1]["attack_detected"])
        self.assertEqual(results[1]["attack_category"], "Reconnaissance")
        self.assertAlmostEqual(results[1]["category_confidence"], 0.85)

        # Record 2: Normal
        self.assertEqual(results[2]["binary_prediction"], "Normal")
        self.assertFalse(results[2]["attack_detected"])

        # Crucial check: LightGBM was called with exactly 1 row (the malicious one)
        called_df = mock_lgb.predict.call_args[0][0]
        self.assertEqual(len(called_df), 1)

    def test_real_trained_models_end_to_end_integration(self):
        """Integration test using the actual loaded models on real validation data."""
        if not VAL_V2_CLEAN_PARQUET.exists():
            self.skipTest("Validation parquet dataset not found.")

        pipeline = CascadedInferencePipeline.from_loader()
        df_val = pd.read_parquet(VAL_V2_CLEAN_PARQUET)

        # 1 Real Normal flow
        normal_rows = df_val[df_val["attack"] == 0]
        if not normal_rows.empty:
            normal_sample = normal_rows.iloc[0].to_dict()
            res_norm = pipeline.predict_single(normal_sample)
            self.assertEqual(res_norm["status"], "success")
            self.assertEqual(res_norm["binary_prediction"], "Normal")
            self.assertFalse(res_norm["attack_detected"])
            self.assertIsNone(res_norm["attack_category"])
            self.assertGreater(res_norm["binary_confidence"], 0.5)

        # 1 Real Malicious flow
        attack_rows = df_val[df_val["attack"] == 1]
        if not attack_rows.empty:
            attack_sample = attack_rows.iloc[0].to_dict()
            res_attack = pipeline.predict_single(attack_sample)
            self.assertEqual(res_attack["status"], "success")
            self.assertEqual(res_attack["binary_prediction"], "Malicious")
            self.assertTrue(res_attack["attack_detected"])
            self.assertIsNotNone(res_attack["attack_category"])
            self.assertGreater(res_attack["binary_confidence"], 0.5)
            self.assertGreater(res_attack["category_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
