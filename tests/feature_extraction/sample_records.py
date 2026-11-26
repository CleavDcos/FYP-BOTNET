"""
Synthetic Sample Network Flow Records for Feature Extraction Testing.

DISCLAIMER:
All records below are ARTIFICIALLY GENERATED SYNTHETIC TEST SAMPLES designed
to validate parsing, aggregation math, and inference compatibility.
They DO NOT represent actual real-world network traffic captures.
"""

from typing import Any, Dict, List

# -------------------------------------------------------------------------
# 1. Synthetic Single Flow Dictionaries (Argus-like fields)
# -------------------------------------------------------------------------

# Benign IoT sensor DNS query
SYNTHETIC_BENIGN_DNS_FLOW: Dict[str, Any] = {
    "stime": 1528099200.123456,
    "ltime": 1528099200.145678,
    "flgs": "e",
    "proto": "udp",
    "saddr": "192.168.1.105",
    "sport": 54321,
    "daddr": "192.168.1.1",
    "dport": 53,
    "pkts": 2,
    "spkts": 1,
    "dpkts": 1,
    "bytes": 140,
    "sbytes": 60,
    "dbytes": 80,
    "state": "CON",
    "dur": 0.022222,
    "mean": 0.022222,
    "stddev": 0.0,
    "min": 0.022222,
    "max": 0.022222,
    "rate": 90.0,
}

# Benign Smart Thermostat HTTP request
SYNTHETIC_BENIGN_HTTP_FLOW: Dict[str, Any] = {
    "stime": 1528099201.000000,
    "ltime": 1528099201.250000,
    "flgs": "e s",
    "proto": "tcp",
    "saddr": "192.168.1.110",
    "sport": 49152,
    "daddr": "10.0.0.50",
    "dport": 80,
    "pkts": 10,
    "spkts": 6,
    "dpkts": 4,
    "bytes": 1250,
    "sbytes": 450,
    "dbytes": 800,
    "state": "CON",
    "dur": 0.25,
    "mean": 0.025,
    "stddev": 0.005,
    "min": 0.01,
    "max": 0.05,
    "rate": 40.0,
}

# Malicious High-Rate UDP Flood Flow
SYNTHETIC_MALICIOUS_UDP_FLOOD: Dict[str, Any] = {
    "stime": 1528099202.000000,
    "ltime": 1528099202.500000,
    "flgs": "e",
    "proto": "udp",
    "saddr": "192.168.1.200",
    "sport": 12345,
    "daddr": "192.168.1.1",
    "dport": 80,
    "pkts": 500,
    "spkts": 500,
    "dpkts": 0,
    "bytes": 50000,
    "sbytes": 50000,
    "dbytes": 0,
    "state": "INT",
    "dur": 0.5,
    "mean": 0.001,
    "stddev": 0.0002,
    "min": 0.0005,
    "max": 0.002,
    "rate": 1000.0,
}

# Malicious Port Scanning Flow (Single SYN Packet)
SYNTHETIC_MALICIOUS_TCP_SCAN: Dict[str, Any] = {
    "stime": 1528099203.000000,
    "ltime": 1528099203.001000,
    "flgs": "e s",
    "proto": "tcp",
    "saddr": "192.168.1.200",
    "sport": 60001,
    "daddr": "192.168.1.1",
    "dport": 22,
    "pkts": 1,
    "spkts": 1,
    "dpkts": 0,
    "bytes": 64,
    "sbytes": 64,
    "dbytes": 0,
    "state": "REQ",
    "dur": 0.001,
    "mean": 0.001,
    "stddev": 0.0,
    "min": 0.001,
    "max": 0.001,
    "rate": 1000.0,
}

# Synthetic Flow with Hex Port Encoding (Characteristic of BoT-IoT dataset ICMP representations)
SYNTHETIC_HEX_PORT_ICMP_FLOW: Dict[str, Any] = {
    "stime": 1528099204.000000,
    "ltime": 1528099204.050000,
    "flgs": "e",
    "proto": "icmp",
    "saddr": "192.168.1.105",
    "sport": "0x0303",  # Hex representation -> 771
    "daddr": "192.168.1.1",
    "dport": "0x0000",  # Hex representation -> 0
    "pkts": 1,
    "spkts": 1,
    "dpkts": 0,
    "bytes": 74,
    "sbytes": 74,
    "dbytes": 0,
    "state": "INT",
    "dur": 0.05,
    "mean": 0.05,
    "stddev": 0.0,
    "min": 0.05,
    "max": 0.05,
    "rate": 20.0,
}

# -------------------------------------------------------------------------
# 2. Synthetic Raw Argus ra Output Stream (CSV Format)
# -------------------------------------------------------------------------

SYNTHETIC_ARGUS_CSV_TEXT: str = """# Argus ra flow capture - Synthetic test dataset
# Date: 2026-11-26
stime,ltime,flgs,proto,saddr,sport,daddr,dport,pkts,spkts,dpkts,bytes,sbytes,dbytes,state,dur,mean,stddev,min,max,rate
1528099200.10,1528099200.12,e,udp,192.168.1.100,5353,224.0.0.251,5353,2,2,0,140,140,0,INT,0.02,0.02,0.0,0.02,0.02,100.0
1528099201.00,1528099201.50,e s,tcp,192.168.1.105,49152,10.0.0.1,80,10,6,4,1500,500,1000,CON,0.50,0.05,0.01,0.02,0.08,20.0
1528099202.00,1528099202.10,e,udp,192.168.1.200,1234,10.0.0.1,80,100,100,0,10000,10000,0,INT,0.10,0.001,0.0,0.001,0.001,1000.0
# Comment inside stream
1528099203.00,1528099203.05,e,icmp,192.168.1.105,0x0303,10.0.0.1,0x0000,1,1,0,74,74,0,INT,0.05,0.05,0.0,0.05,0.05,20.0
"""
