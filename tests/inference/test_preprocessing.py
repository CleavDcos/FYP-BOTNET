"""
Unit Tests for Inference Preprocessing and Validation.

Tests:
- Feature ordering preservation
- Missing features validation
- Invalid input types handling
- Target leakage prevention
- Data type sanitization
"""

import unittest
import pandas as pd
import numpy as np

from config.settings import CANONICAL_22_FEATURES
from src.inference.preprocessing import (
    InferencePreprocessor,
    InferenceValidationError,
)


class TestInferencePreprocessor(unittest.TestCase):
    """Test suite for InferencePreprocessor."""

    def setUp(self):
        self.preprocessor = InferencePreprocessor()
        # Create a valid single-record dictionary
        self.valid_record = {
            "proto": "tcp",
            "flgs": "e",
            "state": "CON",
            "sport": 5000,
            "dport": 80,
            "pkts": 10,
            "spkts": 6,
            "dpkts": 4,
            "dur": 1.5,
            "mean": 0.25,
            "stddev": 0.05,
            "min": 0.01,
            "rate": 6.67,
            "TnBPSrcIP": 1200,
            "TnBPDstIP": 800,
            "TnP_PSrcIP": 10,
            "TnP_PDstIP": 10,
            "TnP_PerProto": 10,
            "TnP_Per_Dport": 10,
            "N_IN_Conn_P_DstIP": 5,
            "N_IN_Conn_P_SrcIP": 5,
            "Pkts_P_State_P_Protocol_P_SrcIP": 10,
        }

    def test_valid_single_dict_record(self):
        """Test that a valid single dictionary is transformed into 1-row canonical DataFrame."""
        df = self.preprocessor.validate_and_transform(self.valid_record)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), CANONICAL_22_FEATURES)

    def test_feature_ordering_strictly_preserved(self):
        """Test that arbitrary input column order is rearranged into exact canonical order."""
        # Scramble keys
        scrambled_keys = list(self.valid_record.keys())
        scrambled_keys.reverse()
        scrambled_record = {k: self.valid_record[k] for k in scrambled_keys}

        df = self.preprocessor.validate_and_transform(scrambled_record)
        self.assertEqual(list(df.columns), CANONICAL_22_FEATURES)

    def test_missing_features_handled_appropriately(self):
        """Test that missing required canonical features raise an InferenceValidationError."""
        incomplete_record = self.valid_record.copy()
        del incomplete_record["proto"]
        del incomplete_record["dport"]

        with self.assertRaises(InferenceValidationError) as ctx:
            self.preprocessor.validate_and_transform(incomplete_record)

        self.assertIn("proto", str(ctx.exception))
        self.assertIn("dport", str(ctx.exception))

    def test_invalid_input_types_handled(self):
        """Test that unsupported or empty inputs raise InferenceValidationError."""
        invalid_inputs = [
            None,
            123,
            "string_data",
            {},
            [],
            [1, 2, 3],
            pd.DataFrame(),
        ]
        for bad_input in invalid_inputs:
            with self.subTest(bad_input=bad_input):
                with self.assertRaises(InferenceValidationError):
                    self.preprocessor.validate_and_transform(bad_input)

    def test_target_leakage_prevention(self):
        """Test that target columns (attack, category, subcategory) are stripped out."""
        record_with_targets = self.valid_record.copy()
        record_with_targets["attack"] = 1
        record_with_targets["category"] = "DDoS"
        record_with_targets["subcategory"] = "HTTP"

        df = self.preprocessor.validate_and_transform(record_with_targets)
        self.assertEqual(len(df.columns), 22)
        for target in ["attack", "category", "subcategory"]:
            self.assertNotIn(target, df.columns)

    def test_non_numeric_in_numeric_feature_raises_error(self):
        """Test that non-numeric strings in numeric features raise InferenceValidationError."""
        corrupted_record = self.valid_record.copy()
        corrupted_record["rate"] = "not_a_number"

        with self.assertRaises(InferenceValidationError) as ctx:
            self.preprocessor.validate_and_transform(corrupted_record)

        self.assertIn("not_a_number", str(ctx.exception))
        self.assertIn("rate", str(ctx.exception))

    def test_batch_records_list_and_dataframe(self):
        """Test transformation on a list of records and a pandas DataFrame."""
        record_2 = self.valid_record.copy()
        record_2["sport"] = 5001

        batch_list = [self.valid_record, record_2]
        df_from_list = self.preprocessor.validate_and_transform(batch_list)
        self.assertEqual(len(df_from_list), 2)
        self.assertEqual(list(df_from_list.columns), CANONICAL_22_FEATURES)

        df_input = pd.DataFrame(batch_list)
        df_from_df = self.preprocessor.validate_and_transform(df_input)
        self.assertEqual(len(df_from_df), 2)
        self.assertEqual(list(df_from_df.columns), CANONICAL_22_FEATURES)


if __name__ == "__main__":
    unittest.main()
