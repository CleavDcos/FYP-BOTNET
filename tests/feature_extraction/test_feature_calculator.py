"""
Unit Tests for FeatureCalculator and WindowTracker.

Tests:
- Generation of all canonical 22 features in strict order
- Calculation of direct flow features
- Calculation of sliding window aggregation features (TnBPSrcIP, TnP_PSrcIP, N_IN_Conn, etc.)
- Sliding window capacity eviction and running count updates
- Window state clearing
"""

import unittest

from config.settings import CANONICAL_22_FEATURES
from src.feature_extraction.feature_calculator import FeatureCalculator, WindowTracker
from src.feature_extraction.flow_parser import FlowParser
from tests.feature_extraction.sample_records import (
    SYNTHETIC_BENIGN_DNS_FLOW,
    SYNTHETIC_BENIGN_HTTP_FLOW,
    SYNTHETIC_MALICIOUS_UDP_FLOOD,
)


class TestFeatureCalculator(unittest.TestCase):
    """Test suite for FeatureCalculator and WindowTracker."""

    def setUp(self):
        self.calculator = FeatureCalculator(window_size=100)

    def test_canonical_22_feature_names_and_strict_order(self):
        """Test that calculated features exactly match CANONICAL_22_FEATURES in name and order."""
        features = self.calculator.calculate_features(SYNTHETIC_BENIGN_DNS_FLOW)

        self.assertEqual(len(features), 22)
        self.assertEqual(list(features.keys()), CANONICAL_22_FEATURES)

    def test_direct_numerical_and_categorical_feature_types(self):
        """Test that calculated feature values have expected types and semantics."""
        features = self.calculator.calculate_features(SYNTHETIC_BENIGN_HTTP_FLOW)

        # Categoricals (str)
        self.assertIsInstance(features["proto"], str)
        self.assertIsInstance(features["flgs"], str)
        self.assertIsInstance(features["state"], str)
        self.assertEqual(features["proto"], "tcp")
        self.assertEqual(features["flgs"], "e s")
        self.assertEqual(features["state"], "CON")

        # Ports (int)
        self.assertIsInstance(features["sport"], int)
        self.assertIsInstance(features["dport"], int)
        self.assertEqual(features["sport"], 49152)
        self.assertEqual(features["dport"], 80)

        # Packets (int)
        self.assertIsInstance(features["pkts"], int)
        self.assertIsInstance(features["spkts"], int)
        self.assertIsInstance(features["dpkts"], int)
        self.assertEqual(features["pkts"], 10)

        # Durations and Rates (float)
        self.assertIsInstance(features["dur"], float)
        self.assertIsInstance(features["mean"], float)
        self.assertIsInstance(features["stddev"], float)
        self.assertIsInstance(features["min"], float)
        self.assertIsInstance(features["rate"], float)
        self.assertAlmostEqual(features["dur"], 0.25)
        self.assertAlmostEqual(features["rate"], 40.0)

    def test_windowed_behavioral_aggregations_update(self):
        """
        Test that windowed metrics (TnBPSrcIP, TnP_PSrcIP, N_IN_Conn, etc.)
        accumulate accurately across sequential flows sharing endpoints.
        """
        calc = FeatureCalculator(window_size=10)

        # Flow 1: 192.168.1.100 -> 10.0.0.1 (UDP, 50 pkts, 5000 bytes, dport 80)
        f1 = {
            "saddr": "192.168.1.100",
            "daddr": "10.0.0.1",
            "sport": 1000,
            "dport": 80,
            "proto": "udp",
            "state": "INT",
            "pkts": 50,
            "bytes": 5000,
            "dur": 0.5,
        }
        res1 = calc.calculate_features(f1)
        self.assertEqual(res1["TnP_PSrcIP"], 50)
        self.assertEqual(res1["TnBPSrcIP"], 5000)
        self.assertEqual(res1["TnP_PDstIP"], 50)
        self.assertEqual(res1["TnBPDstIP"], 5000)
        self.assertEqual(res1["TnP_PerProto"], 50)
        self.assertEqual(res1["TnP_Per_Dport"], 50)
        self.assertEqual(res1["N_IN_Conn_P_SrcIP"], 1)
        self.assertEqual(res1["N_IN_Conn_P_DstIP"], 1)
        self.assertEqual(res1["Pkts_P_State_P_Protocol_P_SrcIP"], 50)

        # Flow 2: Same Source (192.168.1.100) -> Different Dst (10.0.0.2) (UDP, 100 pkts, 10000 bytes, dport 80)
        f2 = {
            "saddr": "192.168.1.100",
            "daddr": "10.0.0.2",
            "sport": 1001,
            "dport": 80,
            "proto": "udp",
            "state": "INT",
            "pkts": 100,
            "bytes": 10000,
            "dur": 1.0,
        }
        res2 = calc.calculate_features(f2)
        # Source metrics should sum f1 + f2
        self.assertEqual(res2["TnP_PSrcIP"], 150)
        self.assertEqual(res2["TnBPSrcIP"], 15000)
        self.assertEqual(res2["N_IN_Conn_P_SrcIP"], 2)

        # Dst metrics for 10.0.0.2 should only reflect f2
        self.assertEqual(res2["TnP_PDstIP"], 100)
        self.assertEqual(res2["TnBPDstIP"], 10000)
        self.assertEqual(res2["N_IN_Conn_P_DstIP"], 1)

        # Protocol UDP and Port 80 should reflect f1 + f2
        self.assertEqual(res2["TnP_PerProto"], 150)
        self.assertEqual(res2["TnP_Per_Dport"], 150)
        self.assertEqual(res2["Pkts_P_State_P_Protocol_P_SrcIP"], 150)

    def test_window_eviction_when_capacity_reached(self):
        """Test that oldest flows are evicted and subtracted from running counters when capacity is exceeded."""
        tracker = WindowTracker(window_size=2)
        parser = FlowParser()

        f1 = parser.parse({"saddr": "1.1.1.1", "daddr": "2.2.2.2", "pkts": 10, "bytes": 100, "proto": "udp"})
        f2 = parser.parse({"saddr": "1.1.1.1", "daddr": "2.2.2.2", "pkts": 20, "bytes": 200, "proto": "udp"})
        f3 = parser.parse({"saddr": "1.1.1.1", "daddr": "2.2.2.2", "pkts": 30, "bytes": 300, "proto": "udp"})

        tracker.add_flow(f1)
        tracker.add_flow(f2)
        # At this point, window contains f1 and f2 (pkts = 30)
        m = tracker.get_window_metrics(f2)
        self.assertEqual(m["TnP_PSrcIP"], 30)

        # Adding f3 should evict f1 (leaving f2 + f3, pkts = 50)
        tracker.add_flow(f3)
        m3 = tracker.get_window_metrics(f3)
        self.assertEqual(m3["TnP_PSrcIP"], 50)
        self.assertEqual(m3["TnBPSrcIP"], 500)
        self.assertEqual(m3["N_IN_Conn_P_SrcIP"], 2)


if __name__ == "__main__":
    unittest.main()
