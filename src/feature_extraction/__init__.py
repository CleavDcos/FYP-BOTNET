"""
Live Network Feature Extraction Module for IoT Botnet Detection.

Provides:
- ArgusReader: Incremental stream and file reader with bounded deduplication.
- FlowParser: Normalization of Argus flow records and hex-encoded ports.
- FeatureCalculator: State-aware sliding-window calculation of the canonical 22 features.
- LiveExtractor: Public interface for single-record, batch, and streaming feature extraction.
"""

from src.feature_extraction.argus_reader import (
    ArgusReader,
    ArgusReaderError,
)
from src.feature_extraction.flow_parser import (
    FlowParser,
    ParsedFlow,
)
from src.feature_extraction.feature_calculator import (
    FeatureCalculator,
    WindowTracker,
)
from src.feature_extraction.live_extractor import (
    FeatureExtractionError,
    LiveExtractor,
)

__all__ = [
    "ArgusReader",
    "ArgusReaderError",
    "FlowParser",
    "ParsedFlow",
    "FeatureCalculator",
    "WindowTracker",
    "LiveExtractor",
    "FeatureExtractionError",
]
