"""
Unit Tests for FlowParser.

Tests:
- Normalization of flow records
- Hexadecimal port parsing (characteristic of BoT-IoT dataset)
- Missing field derivation (dur, pkts, bytes, rate)
- Protocol, state, and flag formatting
"""

import unittest

from src.feature_extraction.flow_parser import FlowParser, ParsedFlow
from tests.feature_extraction.sample_records import (
    SYNTHETIC_BENIGN_DNS_FLOW,
    SYNTHETIC_HEX_PORT_ICMP_FLOW,
    SYNTHETIC_MALICIOUS_UDP_FLOOD,
)


class TestFlowParser(unittest.TestCase):
    """Test suite for FlowParser."""

    def setUp(self):
        self.parser = FlowParser()

    def test_parse_standard_benign_record(self):
        """Test parsing standard benign dictionary record."""
        flow = self.parser.parse(SYNTHETIC_BENIGN_DNS_FLOW)

        self.assertIsInstance(flow, ParsedFlow)
        self.assertEqual(flow.proto, "udp")
        self.assertEqual(flow.sport, 54321)
        self.assertEqual(flow.dport, 53)
        self.assertEqual(flow.pkts, 2)
        self.assertEqual(flow.spkts, 1)
        self.assertEqual(flow.dpkts, 1)
        self.assertEqual(flow.bytes, 140)
        self.assertEqual(flow.state, "CON")
        self.assertEqual(flow.flgs, "e")
        self.assertAlmostEqual(flow.dur, 0.022222)
        self.assertAlmostEqual(flow.rate, 90.0)

    def test_hexadecimal_port_sanitization(self):
        """Test conversion of hex-encoded port representations (0x0303 -> 771)."""
        flow = self.parser.parse(SYNTHETIC_HEX_PORT_ICMP_FLOW)

        self.assertEqual(flow.sport, 771)  # 0x0303 in hex == 771 in decimal
        self.assertEqual(flow.dport, 0)    # 0x0000 in hex == 0
        self.assertEqual(flow.proto, "icmp")

    def test_derivation_of_missing_fields(self):
        """Test automatic derivation of duration, rate, pkts, and bytes if missing."""
        incomplete_record = {
            "proto": "TCP",
            "state": "con",
            "flgs": "e",
            "saddr": "10.0.0.1",
            "daddr": "10.0.0.2",
            "sport": "80",
            "dport": "443",
            "spkts": "5",
            "dpkts": "5",
            "sbytes": "500",
            "dbytes": "1500",
            "stime": "100.0",
            "ltime": "102.0",
        }
        flow = self.parser.parse(incomplete_record)

        self.assertEqual(flow.proto, "tcp")
        self.assertEqual(flow.state, "CON")
        self.assertEqual(flow.pkts, 10)         # spkts (5) + dpkts (5)
        self.assertEqual(flow.bytes, 2000)      # sbytes (500) + dbytes (1500)
        self.assertAlmostEqual(flow.dur, 2.0)   # ltime (102.0) - stime (100.0)
        self.assertAlmostEqual(flow.rate, 5.0)  # pkts (10) / dur (2.0)
        self.assertEqual(flow.sport, 80)
        self.assertEqual(flow.dport, 443)

    def test_zero_duration_rate_handling(self):
        """Test that rate defaults safely to 0.0 when duration is zero."""
        record_zero_dur = {
            "proto": "udp",
            "sport": 1000,
            "dport": 2000,
            "pkts": 5,
            "dur": 0.0,
        }
        flow = self.parser.parse(record_zero_dur)
        self.assertEqual(flow.rate, 0.0)


if __name__ == "__main__":
    unittest.main()
