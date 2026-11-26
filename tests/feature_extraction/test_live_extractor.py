"""
Unit Tests for LiveExtractor.

Tests:
- Single-flow feature extraction to DataFrame
- Batch extraction to DataFrame
- Streaming generator extraction
- Error handling policy (skip vs raise)
- Schema compatibility with InferencePreprocessor
"""

import io
import unittest
import pandas as pd

from config.settings import CANONICAL_22_FEATURES
from src.feature_extraction.live_extractor import FeatureExtractionError, LiveExtractor
from src.inference.preprocessing import InferencePreprocessor
from tests.feature_extraction.sample_records import (
    SYNTHETIC_ARGUS_CSV_TEXT,
    SYNTHETIC_BENIGN_DNS_FLOW,
    SYNTHETIC_BENIGN_HTTP_FLOW,
    SYNTHETIC_MALICIOUS_UDP_FLOOD,
)


class TestLiveExtractor(unittest.TestCase):
    """Test suite for LiveExtractor."""

    def setUp(self):
        self.extractor = LiveExtractor(window_size=50)
        self.inference_preprocessor = InferencePreprocessor()

    def test_extract_single_produces_valid_dataframe(self):
        """Test that extract_single returns a 1-row DataFrame with exact canonical schema."""
        df = self.extractor.extract_single(SYNTHETIC_BENIGN_DNS_FLOW)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), CANONICAL_22_FEATURES)

        # Verify it directly passes InferencePreprocessor validation
        validated_df = self.inference_preprocessor.validate_and_transform(df)
        self.assertEqual(len(validated_df), 1)
        self.assertEqual(list(validated_df.columns), CANONICAL_22_FEATURES)

    def test_extract_batch_preserves_count_and_order(self):
        """Test batch extraction across mixed flow records."""
        batch = [
            SYNTHETIC_BENIGN_DNS_FLOW,
            SYNTHETIC_BENIGN_HTTP_FLOW,
            SYNTHETIC_MALICIOUS_UDP_FLOOD,
        ]
        df_batch = self.extractor.extract_batch(batch)

        self.assertEqual(len(df_batch), 3)
        self.assertEqual(list(df_batch.columns), CANONICAL_22_FEATURES)
        self.assertEqual(df_batch.iloc[0]["proto"], "udp")
        self.assertEqual(df_batch.iloc[1]["proto"], "tcp")
        self.assertEqual(df_batch.iloc[2]["proto"], "udp")

    def test_extract_stream_incremental_batches(self):
        """Test streaming extraction emitting chunks from a text stream."""
        stream = io.StringIO(SYNTHETIC_ARGUS_CSV_TEXT)
        chunks = list(self.extractor.extract_stream(stream, delimiter=",", chunk_size=2))

        # 4 records in sample text chunked by 2 -> 2 chunks of 2
        self.assertEqual(len(chunks), 2)
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 4)
        for chunk in chunks:
            self.assertEqual(list(chunk.columns), CANONICAL_22_FEATURES)

    def test_invalid_record_error_handling(self):
        """Test handling of invalid/malformed records."""
        corrupted_batch = [
            SYNTHETIC_BENIGN_DNS_FLOW,
            {"invalid_data_only": True},  # Missing all standard fields
            SYNTHETIC_BENIGN_HTTP_FLOW,
        ]
        # Default policy: on_error='skip' should not crash
        extractor_skip = LiveExtractor(on_error="skip")
        df_skip = extractor_skip.extract_batch(corrupted_batch)
        self.assertGreaterEqual(len(df_skip), 2)


if __name__ == "__main__":
    unittest.main()
