"""
Unit Tests for Model Preprocessor and Random Forest Binary Detection
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import (
    CANONICAL_22_FEATURES,
    TARGET_LABELS,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
    PROJECT_ROOT,
)
from src.preprocessing.model_preprocessor import (
    build_feature_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


class TestModelPreprocessor(unittest.TestCase):
    """Test suite for shared model-preprocessing module."""

    def test_categorical_numerical_partition(self):
        """Verify categorical and numerical features partition correctly and reconstruct canonical 22."""
        self.assertEqual(len(CATEGORICAL_FEATURES), 3)
        self.assertEqual(set(CATEGORICAL_FEATURES), {"proto", "flgs", "state"})
        self.assertEqual(len(NUMERICAL_FEATURES), 19)
        self.assertEqual(len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES), 22)
        self.assertEqual(set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES), set(CANONICAL_22_FEATURES))

    def test_preprocessor_fit_transform(self):
        """Verify preprocessor fits on training data and transforms validation data without errors."""
        df_train_sample = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET, columns=CANONICAL_22_FEATURES).head(100)
        df_val_sample = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=CANONICAL_22_FEATURES).head(50)

        preprocessor = build_feature_preprocessor()
        X_train_trans = preprocessor.fit_transform(df_train_sample)
        X_val_trans = preprocessor.transform(df_val_sample)

        self.assertEqual(X_train_trans.shape[0], 100)
        self.assertEqual(X_val_trans.shape[0], 50)
        self.assertEqual(X_train_trans.shape[1], X_val_trans.shape[1])
        # Ensure no NaN produced
        self.assertFalse(np.isnan(X_train_trans).any())
        self.assertFalse(np.isnan(X_val_trans).any())

    def test_zero_leakage_targets_not_in_features(self):
        """Verify target columns are never present in preprocessor features."""
        for target in TARGET_LABELS:
            self.assertNotIn(target, CANONICAL_22_FEATURES)
            self.assertNotIn(target, CATEGORICAL_FEATURES)
            self.assertNotIn(target, NUMERICAL_FEATURES)


if __name__ == "__main__":
    unittest.main()
