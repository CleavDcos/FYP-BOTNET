"""
Unit Tests for ModelLoader.

Tests:
- Loading model artifacts
- Meaningful error on missing artifacts
- In-memory caching
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.inference.model_loader import (
    ModelLoader,
    ModelArtifactNotFoundError,
    ModelArtifactLoadError,
)


class TestModelLoader(unittest.TestCase):
    """Test suite for ModelLoader artifact handling and caching."""

    def test_missing_xgboost_artifact_raises_meaningful_error(self):
        """Test that non-existent XGBoost path raises ModelArtifactNotFoundError."""
        loader = ModelLoader(xgb_path="/non_existent/path/xgb.joblib")
        with self.assertRaises(ModelArtifactNotFoundError) as ctx:
            loader.get_xgb_model()
        self.assertIn("XGBoost model artifact not found", str(ctx.exception))

    def test_missing_lightgbm_artifact_raises_meaningful_error(self):
        """Test that non-existent LightGBM path raises ModelArtifactNotFoundError."""
        loader = ModelLoader(lgb_path="/non_existent/path/lgb.joblib")
        with self.assertRaises(ModelArtifactNotFoundError) as ctx:
            loader.get_lgb_model()
        self.assertIn("LightGBM model artifact not found", str(ctx.exception))

    @patch("joblib.load")
    @patch.object(Path, "exists")
    def test_caching_avoids_repeated_disk_reads(self, mock_exists, mock_joblib_load):
        """Test that models are cached in memory after first load."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        loader = ModelLoader(xgb_path="models/fake_xgb.joblib", lgb_path="models/fake_lgb.joblib")

        # Load first time
        m1 = loader.get_xgb_model()
        # Load second time
        m2 = loader.get_xgb_model()

        self.assertIs(m1, m2)
        self.assertEqual(mock_joblib_load.call_count, 1)

    @patch("joblib.load")
    @patch.object(Path, "exists")
    def test_corrupt_artifact_raises_load_error(self, mock_exists, mock_joblib_load):
        """Test that corrupted model serialization raises ModelArtifactLoadError."""
        mock_exists.return_value = True
        mock_joblib_load.side_effect = RuntimeError("Corrupted file header")

        loader = ModelLoader(xgb_path="models/corrupt.joblib")
        with self.assertRaises(ModelArtifactLoadError) as ctx:
            loader.get_xgb_model()
        self.assertIn("Failed to load XGBoost model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
