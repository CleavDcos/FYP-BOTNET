"""
Pipeline Integrity and Reproducibility Unit Tests
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import (
    RAW_CSV_FILES,
    CANONICAL_22_FEATURES,
    TARGET_LABELS,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
    TEST_V2_CLEAN_PARQUET,
    TRAIN_V2_ROS_CAT_PARQUET,
    TRAIN_V2_SMOTE_CAT_PARQUET,
    TRAIN_V2_ROS_BIN_PARQUET,
    TRAIN_V2_SMOTE_BIN_PARQUET,
    TRAIN_34_PARQUET,
    VAL_34_PARQUET,
    TEST_34_PARQUET,
)


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite verifying data integrity, feature schema, zero leakage, and row counts."""

    def test_canonical_feature_count(self):
        """Verify canonical feature list contains exactly 22 features."""
        self.assertEqual(len(CANONICAL_22_FEATURES), 22)
        # Ensure no label is in CANONICAL_22_FEATURES
        for target in TARGET_LABELS:
            self.assertNotIn(target, CANONICAL_22_FEATURES)

    def test_clean_datasets_feature_consistency(self):
        """Verify train, val, and test clean datasets have identical 22 features in canonical order."""
        for name, fpath in [
            ("Train v2", TRAIN_V2_CLEAN_PARQUET),
            ("Val v2", VAL_V2_CLEAN_PARQUET),
            ("Test v2", TEST_V2_CLEAN_PARQUET),
        ]:
            self.assertTrue(fpath.exists(), f"{name} file does not exist at {fpath}")
            df = pd.read_parquet(fpath)
            # First 22 columns must match canonical order
            self.assertEqual(
                list(df.columns[:22]),
                CANONICAL_22_FEATURES,
                f"Feature mismatch or ordering discrepancy in {name}",
            )
            # Targets must be strictly segregated as final columns
            self.assertEqual(list(df.columns[22:]), TARGET_LABELS)
            # Zero nulls allowed
            self.assertEqual(df.isnull().sum().sum(), 0, f"Found nulls in {name}")

    def test_split_row_counts(self):
        """Verify exact row counts across 70/15/15 partitions."""
        train_rows = len(pd.read_parquet(TRAIN_V2_CLEAN_PARQUET))
        val_rows = len(pd.read_parquet(VAL_V2_CLEAN_PARQUET))
        test_rows = len(pd.read_parquet(TEST_V2_CLEAN_PARQUET))

        self.assertEqual(train_rows, 2567965)
        self.assertEqual(val_rows, 550278)
        self.assertEqual(test_rows, 550279)
        self.assertEqual(train_rows + val_rows + test_rows, 3668522)

    def test_zero_target_leakage_in_resampled_datasets(self):
        """Verify resampled datasets contain strictly 22 features and only their designated target."""
        # Category resampled
        for fpath in [TRAIN_V2_ROS_CAT_PARQUET, TRAIN_V2_SMOTE_CAT_PARQUET]:
            df = pd.read_parquet(fpath)
            self.assertIn("category", df.columns)
            self.assertNotIn("attack", df.columns)
            self.assertNotIn("subcategory", df.columns)
            self.assertEqual(list(df.columns[:22]), CANONICAL_22_FEATURES)
            self.assertEqual(len(df.columns), 23)

        # Binary resampled
        for fpath in [TRAIN_V2_ROS_BIN_PARQUET, TRAIN_V2_SMOTE_BIN_PARQUET]:
            df = pd.read_parquet(fpath)
            self.assertIn("attack", df.columns)
            self.assertNotIn("category", df.columns)
            self.assertNotIn("subcategory", df.columns)
            self.assertEqual(list(df.columns[:22]), CANONICAL_22_FEATURES)
            self.assertEqual(len(df.columns), 23)

    def test_raw_and_34_benchmarks_exist(self):
        """Verify raw datasets and 34-feature benchmark datasets remain accessible."""
        for raw_csv in RAW_CSV_FILES:
            self.assertTrue(raw_csv.exists(), f"Raw CSV missing: {raw_csv}")

        for bench in [TRAIN_34_PARQUET, VAL_34_PARQUET, TEST_34_PARQUET]:
            self.assertTrue(bench.exists(), f"Benchmark split missing: {bench}")
            df = pd.read_parquet(bench)
            self.assertEqual(len(df.columns), 37)  # 34 features + 3 targets


if __name__ == "__main__":
    unittest.main()
