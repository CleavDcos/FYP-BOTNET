"""
End-to-End Integration Tests: Live Feature Extraction -> Cascaded Inference Pipeline.

Tests the complete operational chain:
Argus Flow -> Flow Parser -> Feature Calculator -> Live Extractor -> CascadedInferencePipeline
"""

import io
import unittest

from src.feature_extraction.live_extractor import LiveExtractor
from src.inference.inference_pipeline import CascadedInferencePipeline
from tests.feature_extraction.sample_records import (
    SYNTHETIC_ARGUS_CSV_TEXT,
    SYNTHETIC_BENIGN_DNS_FLOW,
    SYNTHETIC_BENIGN_HTTP_FLOW,
    SYNTHETIC_MALICIOUS_UDP_FLOOD,
)


class TestPipelineIntegration(unittest.TestCase):
    """Integration test suite connecting LiveExtractor to CascadedInferencePipeline."""

    @classmethod
    def setUpClass(cls):
        """Loads the trained cascaded models."""
        try:
            cls.pipeline = CascadedInferencePipeline.from_loader()
            cls.pipeline_loaded = True
        except Exception:
            cls.pipeline_loaded = False

    def setUp(self):
        if not self.pipeline_loaded:
            self.skipTest("Trained models not found on disk; skipping live model integration.")
        self.extractor = LiveExtractor(window_size=100)

    def test_single_flow_end_to_end_pipe(self):
        """Test piping a single raw flow through feature extraction and inference."""
        result = self.extractor.pipe_to_inference(self.pipeline, SYNTHETIC_BENIGN_HTTP_FLOW)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "success")
        self.assertIn(result["binary_prediction"], ["Normal", "Malicious"])
        self.assertIn("attack_detected", result)
        self.assertIn("binary_confidence", result)

    def test_batch_flows_end_to_end_pipe(self):
        """Test piping a batch of mixed benign and attack flows through inference."""
        raw_batch = [
            SYNTHETIC_BENIGN_DNS_FLOW,
            SYNTHETIC_BENIGN_HTTP_FLOW,
            SYNTHETIC_MALICIOUS_UDP_FLOOD,
        ]
        results = self.extractor.pipe_to_inference(self.pipeline, raw_batch)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

        for res in results:
            self.assertEqual(res["status"], "success")
            self.assertIn("binary_prediction", res)
            self.assertIn("attack_detected", res)
            if res["attack_detected"]:
                self.assertIsNotNone(res["attack_category"])
                self.assertIsNotNone(res["category_confidence"])
            else:
                self.assertIsNone(res["attack_category"])
                self.assertIsNone(res["category_confidence"])

    def test_streaming_csv_end_to_end_pipe(self):
        """Test streaming synthetic raw Argus CSV text directly into the inference pipeline."""
        stream = io.StringIO(SYNTHETIC_ARGUS_CSV_TEXT)
        predictions = []
        for df_chunk in self.extractor.extract_stream(stream, delimiter=",", chunk_size=2):
            preds = self.pipeline.predict_batch(df_chunk)
            predictions.extend(preds)

        # 4 records in sample text
        self.assertEqual(len(predictions), 4)
        for pred in predictions:
            self.assertEqual(pred["status"], "success")


if __name__ == "__main__":
    unittest.main()
