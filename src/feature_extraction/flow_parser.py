"""
Argus Flow Record Parser.

Normalizes raw Argus/ra flow fields into standardized typed Python records,
sanitizes hexadecimal port notations (characteristic of BoT-IoT), derives missing
flow-level aggregates (dur, pkts, bytes, rate), and documents the precise mapping
from Argus fields to training feature inputs.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union


# -------------------------------------------------------------------------
# ARGUS TO UNSW BOT-IOT TRAINING FEATURE MAPPING DOCUMENTATION
# -------------------------------------------------------------------------
# The UNSW BoT-IoT dataset was derived from network captures processed by
# Argus (Audit Record Generation and Utilization System) clients (ra / racluster).
#
# Argus Field Names (CLI ra output)      Training Feature Name    Type / Notes
# ---------------------------------      ---------------------    ------------
# Proto / proto                          proto                    str: tcp, udp, icmp, arp, ipv6-icmp
# Flgs / Flags / flgs                    flgs                     str: e, e s, e d, e g, eU, etc.
# State / state                          state                    str: INT, CON, REQ, RST, URP, etc.
# Sport / SPort / sport                  sport                    int64: Sanitized port or hex ICMP code
# Dport / DPort / dport                  dport                    int64: Sanitized port or hex ICMP code
# TotPkts / Pkts / pkts                  pkts                     int64: Total transaction packet volume
# SrcPkts / sPkts / spkts                spkts                    int64: Source -> Destination packets
# DstPkts / dPkts / dpkts                dpkts                    int64: Destination -> Source packets
# TotBytes / Bytes / bytes               bytes                    int64: Total transaction bytes
# SrcBytes / sBytes / sbytes             sbytes                   int64: Source -> Destination bytes
# DstBytes / dBytes / dbytes             dbytes                   int64: Destination -> Source bytes
# Dur / dur                              dur                      float64: Flow duration in seconds
# Mean / mean                            mean                     float64: Mean packet duration / interval
# StdDev / Stddev / stddev               stddev                   float64: Standard deviation of packet duration
# Min / min                              min                      float64: Minimum packet duration
# Max / max                              max                      float64: Maximum packet duration
# Rate / rate                            rate                     float64: Overall flow velocity (pkts / sec)
# SrcAddr / saddr / SrcIP                saddr                    str: Source IPv4/IPv6 (Host windowing)
# DstAddr / daddr / DstIP                daddr                    str: Destination IPv4/IPv6 (Host windowing)
# StartTime / stime                      stime                    float64: Flow initiation timestamp
# LastTime / ltime                       ltime                    float64: Flow termination timestamp
# -------------------------------------------------------------------------


@dataclass
class ParsedFlow:
    """Standardized representation of an individual parsed flow record."""
    saddr: str
    daddr: str
    sport: int
    dport: int
    proto: str
    state: str
    flgs: str
    pkts: int
    spkts: int
    dpkts: int
    bytes: int
    sbytes: int
    dbytes: int
    dur: float
    mean: float
    stddev: float
    min: float
    max: float
    rate: float
    stime: float
    ltime: float
    raw_attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Returns flow as dictionary."""
        return asdict(self)


class FlowParser:
    """
    Parses and standardizes raw flow records into typed `ParsedFlow` objects.

    Handles:
    - Case-insensitive field aliases commonly produced by Argus / ra clients.
    - Sanitization of hex port representations (e.g. '0x0303' -> 771).
    - Derivation of missing duration, packet, byte, and rate metrics.
    - Normalization of network protocols, connection flags, and flow states.
    """

    # Field aliases mapping candidate keys to canonical parsed names
    FIELD_ALIASES: Dict[str, List[str]] = {
        "saddr": ["saddr", "srcaddr", "source", "srcip", "s_ip"],
        "daddr": ["daddr", "dstaddr", "destination", "dstip", "d_ip"],
        "sport": ["sport", "srcport", "s_port"],
        "dport": ["dport", "dstport", "d_port"],
        "proto": ["proto", "protocol"],
        "state": ["state", "status"],
        "flgs": ["flgs", "flags", "flag"],
        "pkts": ["pkts", "totpkts", "packets"],
        "spkts": ["spkts", "srcpkts", "s_pkts"],
        "dpkts": ["dpkts", "dstpkts", "d_pkts"],
        "bytes": ["bytes", "totbytes"],
        "sbytes": ["sbytes", "srcbytes", "s_bytes"],
        "dbytes": ["dbytes", "dstbytes", "d_bytes"],
        "dur": ["dur", "duration"],
        "mean": ["mean", "avg_dur"],
        "stddev": ["stddev", "std_dev"],
        "min": ["min", "min_dur"],
        "max": ["max", "max_dur"],
        "rate": ["rate", "flow_rate"],
        "stime": ["stime", "starttime", "start"],
        "ltime": ["ltime", "lasttime", "end"],
    }

    @staticmethod
    def parse_port(val: Any) -> int:
        """
        Parses port representations into integer.
        Handles numeric integers, strings, and hex representations like '0x0303' or '0x5000'.
        Defaults to -1 if missing or unparseable.
        """
        if val is None:
            return -1
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("-", "none", "nan", "null"):
            return -1

        try:
            if val_str.lower().startswith("0x"):
                return int(val_str, 16)
            return int(float(val_str))
        except (ValueError, TypeError):
            return -1

    @staticmethod
    def parse_float(val: Any, default: float = 0.0) -> float:
        """Safely parses float numbers."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def parse_int(val: Any, default: int = 0) -> int:
        """Safely parses integer numbers."""
        if val is None:
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def _resolve_field(self, record_lower: Dict[str, Any], field_key: str, default: Any = None) -> Any:
        """Finds field value across known alias variations."""
        for alias in self.FIELD_ALIASES.get(field_key, [field_key]):
            if alias in record_lower and record_lower[alias] is not None and record_lower[alias] != "":
                return record_lower[alias]
        return default

    def parse(self, raw_record: Dict[str, Any]) -> ParsedFlow:
        """
        Parses a single raw dictionary record into a typed `ParsedFlow`.

        Parameters
        ----------
        raw_record : Dict[str, Any]
            Raw key-value pairs from ArgusReader or synthetic generator.

        Returns
        -------
        ParsedFlow
            Clean, typed, and normalized flow record.
        """
        # Normalize all incoming keys to lowercase for flexible matching
        norm_map = {str(k).strip().lower(): v for k, v in raw_record.items()}

        saddr = str(self._resolve_field(norm_map, "saddr", default="0.0.0.0")).strip()
        daddr = str(self._resolve_field(norm_map, "daddr", default="0.0.0.0")).strip()

        sport = self.parse_port(self._resolve_field(norm_map, "sport", default=-1))
        dport = self.parse_port(self._resolve_field(norm_map, "dport", default=-1))

        proto = str(self._resolve_field(norm_map, "proto", default="udp")).strip().lower()
        state = str(self._resolve_field(norm_map, "state", default="INT")).strip().upper()
        flgs = str(self._resolve_field(norm_map, "flgs", default="e")).strip()

        # Packet counters
        spkts = self.parse_int(self._resolve_field(norm_map, "spkts", default=None), default=0)
        dpkts = self.parse_int(self._resolve_field(norm_map, "dpkts", default=None), default=0)
        raw_pkts = self._resolve_field(norm_map, "pkts", default=None)
        if raw_pkts is not None:
            pkts = self.parse_int(raw_pkts, default=spkts + dpkts)
        else:
            pkts = spkts + dpkts

        # Byte counters
        sbytes = self.parse_int(self._resolve_field(norm_map, "sbytes", default=None), default=0)
        dbytes = self.parse_int(self._resolve_field(norm_map, "dbytes", default=None), default=0)
        raw_bytes = self._resolve_field(norm_map, "bytes", default=None)
        if raw_bytes is not None:
            total_bytes = self.parse_int(raw_bytes, default=sbytes + dbytes)
        else:
            total_bytes = sbytes + dbytes

        # Timing and statistics
        stime = self.parse_float(self._resolve_field(norm_map, "stime", default=0.0), default=0.0)
        ltime = self.parse_float(self._resolve_field(norm_map, "ltime", default=0.0), default=0.0)

        raw_dur = self._resolve_field(norm_map, "dur", default=None)
        if raw_dur is not None:
            dur = max(0.0, self.parse_float(raw_dur, default=0.0))
        elif stime > 0.0 and ltime >= stime:
            dur = max(0.0, ltime - stime)
        else:
            dur = 0.0

        # Duration stats
        mean = self.parse_float(self._resolve_field(norm_map, "mean", default=None), default=dur)
        stddev = self.parse_float(self._resolve_field(norm_map, "stddev", default=None), default=0.0)
        min_dur = self.parse_float(self._resolve_field(norm_map, "min", default=None), default=dur)
        max_dur = self.parse_float(self._resolve_field(norm_map, "max", default=None), default=dur)

        # Flow rate
        raw_rate = self._resolve_field(norm_map, "rate", default=None)
        if raw_rate is not None:
            rate = max(0.0, self.parse_float(raw_rate, default=0.0))
        elif dur > 0.0:
            rate = float(pkts / dur)
        else:
            rate = 0.0

        return ParsedFlow(
            saddr=saddr,
            daddr=daddr,
            sport=sport,
            dport=dport,
            proto=proto,
            state=state,
            flgs=flgs,
            pkts=pkts,
            spkts=spkts,
            dpkts=dpkts,
            bytes=total_bytes,
            sbytes=sbytes,
            dbytes=dbytes,
            dur=dur,
            mean=mean,
            stddev=stddev,
            min=min_dur,
            max=max_dur,
            rate=rate,
            stime=stime,
            ltime=ltime if ltime > 0.0 else (stime + dur),
            raw_attributes=raw_record,
        )
