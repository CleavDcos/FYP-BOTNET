"""
Feature Calculator for UNSW BoT-IoT Canonical 22 Input Features.

Calculates both direct flow-level features (duration, packet counts, rates, ports,
protocols, flags, states) and stateful, windowed behavioral aggregation features
(total packets/bytes per host IP, connection density, and multi-dimensional state groupings)
following the exact definitions of the UNSW BoT-IoT dataset.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import CANONICAL_22_FEATURES
from src.feature_extraction.flow_parser import FlowParser, ParsedFlow


# =========================================================================
# FEATURE REPRODUCTION & WINDOW CLASSIFICATION DOCUMENTATION
# =========================================================================
# The 22 canonical features required by the models are divided into:
#
# 1. DIRECT PER-FLOW FEATURES (13 features):
#    Directly extracted/computed from the flow record itself:
#    - proto, flgs, state: Categorical flow attributes.
#    - sport, dport: Flow source/destination ports (sanitized).
#    - pkts, spkts, dpkts: Packet volumes across the flow transaction.
#    - dur: Flow transaction duration (seconds).
#    - mean, stddev, min: Duration / inter-arrival statistics of the flow.
#    - rate: Transmission velocity (pkts / sec = pkts / dur).
#
# 2. WINDOWED BEHAVIORAL & CONNECTION DENSITY FEATURES (9 features):
#    Calculated across a sliding observation window of network transactions:
#    - TnBPSrcIP: Total number of Bytes per Source IP in the active window.
#    - TnBPDstIP: Total number of Bytes per Destination IP in the active window.
#    - TnP_PSrcIP: Total number of Packets per Source IP in the active window.
#    - TnP_PDstIP: Total number of Packets per Destination IP in the active window.
#    - TnP_PerProto: Total number of Packets per Protocol in the active window.
#    - TnP_Per_Dport: Total number of Packets per Destination Port in the active window.
#    - N_IN_Conn_P_DstIP: Inbound connection count to Destination IP in window.
#    - N_IN_Conn_P_SrcIP: Inbound connection count from Source IP in window.
#    - Pkts_P_State_P_Protocol_P_SrcIP: Total packets grouped by (state, proto, saddr) in window.
# =========================================================================


class WindowTracker:
    """
    Lightweight, bounded sliding observation window tracker.

    Maintains running counters with O(1) addition and eviction to achieve
    ultra-low memory usage on constrained devices like the Raspberry Pi.
    """

    def __init__(self, window_size: int = 100, window_time_sec: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        window_size : int
            Maximum number of flow records in the observation window (default: 100,
            matching UNSW BoT-IoT connection windowing).
        window_time_sec : Optional[float]
            Optional maximum temporal duration (in seconds) for record validity in window.
        """
        self.window_size = max(1, window_size)
        self.window_time_sec = window_time_sec

        self._queue: deque = deque()

        # O(1) Running Aggregation Counters
        self.bytes_src_ip: Dict[str, int] = defaultdict(int)
        self.bytes_dst_ip: Dict[str, int] = defaultdict(int)
        self.pkts_src_ip: Dict[str, int] = defaultdict(int)
        self.pkts_dst_ip: Dict[str, int] = defaultdict(int)
        self.pkts_proto: Dict[str, int] = defaultdict(int)
        self.pkts_dport: Dict[int, int] = defaultdict(int)
        self.conn_dst_ip: Dict[str, int] = defaultdict(int)
        self.conn_src_ip: Dict[str, int] = defaultdict(int)
        self.pkts_state_proto_src: Dict[Tuple[str, str, str], int] = defaultdict(int)

    def clear(self) -> None:
        """Resets the observation window state."""
        self._queue.clear()
        self.bytes_src_ip.clear()
        self.bytes_dst_ip.clear()
        self.pkts_src_ip.clear()
        self.pkts_dst_ip.clear()
        self.pkts_proto.clear()
        self.pkts_dport.clear()
        self.conn_dst_ip.clear()
        self.conn_src_ip.clear()
        self.pkts_state_proto_src.clear()

    def _evict_flow(self, flow: ParsedFlow) -> None:
        """Removes an expired flow's contributions from running counters."""
        self.bytes_src_ip[flow.saddr] -= flow.bytes
        if self.bytes_src_ip[flow.saddr] <= 0:
            del self.bytes_src_ip[flow.saddr]

        self.bytes_dst_ip[flow.daddr] -= flow.bytes
        if self.bytes_dst_ip[flow.daddr] <= 0:
            del self.bytes_dst_ip[flow.daddr]

        self.pkts_src_ip[flow.saddr] -= flow.pkts
        if self.pkts_src_ip[flow.saddr] <= 0:
            del self.pkts_src_ip[flow.saddr]

        self.pkts_dst_ip[flow.daddr] -= flow.pkts
        if self.pkts_dst_ip[flow.daddr] <= 0:
            del self.pkts_dst_ip[flow.daddr]

        self.pkts_proto[flow.proto] -= flow.pkts
        if self.pkts_proto[flow.proto] <= 0:
            del self.pkts_proto[flow.proto]

        self.pkts_dport[flow.dport] -= flow.pkts
        if self.pkts_dport[flow.dport] <= 0:
            del self.pkts_dport[flow.dport]

        self.conn_dst_ip[flow.daddr] -= 1
        if self.conn_dst_ip[flow.daddr] <= 0:
            del self.conn_dst_ip[flow.daddr]

        self.conn_src_ip[flow.saddr] -= 1
        if self.conn_src_ip[flow.saddr] <= 0:
            del self.conn_src_ip[flow.saddr]

        key = (flow.state, flow.proto, flow.saddr)
        self.pkts_state_proto_src[key] -= flow.pkts
        if self.pkts_state_proto_src[key] <= 0:
            del self.pkts_state_proto_src[key]

    def add_flow(self, flow: ParsedFlow) -> None:
        """
        Incorporates a new flow record into the observation window.
        Evicts expired or over-capacity records first.
        """
        # Time-based expiration if configured
        if self.window_time_sec is not None and flow.ltime > 0.0:
            min_allowed_time = flow.ltime - self.window_time_sec
            while self._queue and self._queue[0].ltime < min_allowed_time:
                expired = self._queue.popleft()
                self._evict_flow(expired)

        # Capacity-based expiration
        while len(self._queue) >= self.window_size:
            oldest = self._queue.popleft()
            self._evict_flow(oldest)

        # Add current flow to deque
        self._queue.append(flow)

        # Update O(1) running counters
        self.bytes_src_ip[flow.saddr] += flow.bytes
        self.bytes_dst_ip[flow.daddr] += flow.bytes
        self.pkts_src_ip[flow.saddr] += flow.pkts
        self.pkts_dst_ip[flow.daddr] += flow.pkts
        self.pkts_proto[flow.proto] += flow.pkts
        self.pkts_dport[flow.dport] += flow.pkts
        self.conn_dst_ip[flow.daddr] += 1
        self.conn_src_ip[flow.saddr] += 1
        key = (flow.state, flow.proto, flow.saddr)
        self.pkts_state_proto_src[key] += flow.pkts

    def get_window_metrics(self, flow: ParsedFlow) -> Dict[str, int]:
        """
        Extracts the 9 window aggregation metrics for the provided flow.
        """
        return {
            "TnBPSrcIP": int(self.bytes_src_ip.get(flow.saddr, flow.bytes)),
            "TnBPDstIP": int(self.bytes_dst_ip.get(flow.daddr, flow.bytes)),
            "TnP_PSrcIP": int(self.pkts_src_ip.get(flow.saddr, flow.pkts)),
            "TnP_PDstIP": int(self.pkts_dst_ip.get(flow.daddr, flow.pkts)),
            "TnP_PerProto": int(self.pkts_proto.get(flow.proto, flow.pkts)),
            "TnP_Per_Dport": int(self.pkts_dport.get(flow.dport, flow.pkts)),
            "N_IN_Conn_P_DstIP": int(self.conn_dst_ip.get(flow.daddr, 1)),
            "N_IN_Conn_P_SrcIP": int(self.conn_src_ip.get(flow.saddr, 1)),
            "Pkts_P_State_P_Protocol_P_SrcIP": int(
                self.pkts_state_proto_src.get((flow.state, flow.proto, flow.saddr), flow.pkts)
            ),
        }


class FeatureCalculator:
    """
    Computes canonical 22 input features for live network flows.

    Strictly produces the features and column ordering defined in `config/settings.py`.
    """

    def __init__(
        self,
        window_size: int = 100,
        window_time_sec: Optional[float] = None,
        parser: Optional[FlowParser] = None,
    ) -> None:
        self.parser = parser or FlowParser()
        self.window_tracker = WindowTracker(
            window_size=window_size,
            window_time_sec=window_time_sec,
        )

    def reset_window(self) -> None:
        """Resets the sliding window state."""
        self.window_tracker.clear()

    def calculate_features(
        self,
        flow_input: Union[ParsedFlow, Dict[str, Any]],
        update_window: bool = True,
    ) -> Dict[str, Any]:
        """
        Computes the canonical 22 features for a parsed or raw flow record.

        Parameters
        ----------
        flow_input : ParsedFlow or Dict[str, Any]
            Standardized ParsedFlow object or raw flow key-value dictionary.
        update_window : bool
            Whether to register this flow into the sliding window tracker.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing exactly the 22 canonical features in exact order.
        """
        # Parse if raw dictionary
        if isinstance(flow_input, ParsedFlow):
            flow = flow_input
        else:
            flow = self.parser.parse(flow_input)

        # Update sliding window state
        if update_window:
            self.window_tracker.add_flow(flow)

        # Retrieve windowed aggregations
        window_metrics = self.window_tracker.get_window_metrics(flow)

        # 1. Direct Per-Flow Features
        proto = str(flow.proto)
        flgs = str(flow.flgs)
        state = str(flow.state)
        sport = int(flow.sport)
        dport = int(flow.dport)
        pkts = int(flow.pkts)
        spkts = int(flow.spkts)
        dpkts = int(flow.dpkts)
        dur = float(flow.dur)
        mean = float(flow.mean)
        stddev = float(flow.stddev)
        min_dur = float(flow.min)
        rate = float(flow.rate)

        # 2. Assemble canonical 22 features in strict canonical order
        canonical_record = {
            # Categoricals (3)
            "proto": proto,
            "flgs": flgs,
            "state": state,
            # Flow Ports (2)
            "sport": sport,
            "dport": dport,
            # Packet Counts (3)
            "pkts": pkts,
            "spkts": spkts,
            "dpkts": dpkts,
            # Duration & Inter-arrival Time Statistics (4)
            "dur": dur,
            "mean": mean,
            "stddev": stddev,
            "min": min_dur,
            # Flow Rate & Host Window Behavioral Metrics (7)
            "rate": rate,
            "TnBPSrcIP": window_metrics["TnBPSrcIP"],
            "TnBPDstIP": window_metrics["TnBPDstIP"],
            "TnP_PSrcIP": window_metrics["TnP_PSrcIP"],
            "TnP_PDstIP": window_metrics["TnP_PDstIP"],
            "TnP_PerProto": window_metrics["TnP_PerProto"],
            "TnP_Per_Dport": window_metrics["TnP_Per_Dport"],
            # Connection Density Metrics (3)
            "N_IN_Conn_P_DstIP": window_metrics["N_IN_Conn_P_DstIP"],
            "N_IN_Conn_P_SrcIP": window_metrics["N_IN_Conn_P_SrcIP"],
            "Pkts_P_State_P_Protocol_P_SrcIP": window_metrics["Pkts_P_State_P_Protocol_P_SrcIP"],
        }

        # Verify exact column order match
        assert list(canonical_record.keys()) == CANONICAL_22_FEATURES, (
            "Feature ordering mismatch against config.settings.CANONICAL_22_FEATURES"
        )

        return canonical_record
