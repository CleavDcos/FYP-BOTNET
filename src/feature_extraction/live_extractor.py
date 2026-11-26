"""
Live Network Feature Extractor.

Main public interface orchestrating Argus reading, flow parsing, and feature calculation.
Produces validated pandas DataFrames conforming strictly to the canonical 22 features,
and integrates directly with the existing CascadedInferencePipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, TextIO, Union
import pandas as pd

from config.settings import CANONICAL_22_FEATURES
from src.feature_extraction.argus_reader import ArgusReader
from src.feature_extraction.feature_calculator import FeatureCalculator
from src.feature_extraction.flow_parser import FlowParser, ParsedFlow

logger = logging.getLogger(__name__)


class FeatureExtractionError(Exception):
    """Exception raised when feature extraction encounters an unrecoverable error."""
    pass


class LiveExtractor:
    """
    Orchestrates the live extraction of canonical 22 features from raw network flows.

    Capabilities:
    - Single-flow extraction (`extract_single`).
    - Batch-flow extraction (`extract_batch`).
    - Incremental streaming extraction (`extract_stream`).
    - Direct integration with `CascadedInferencePipeline` (`pipe_to_inference`).
    - Guarantees exact column ordering and dtypes expected by the models.
    """

    def __init__(
        self,
        window_size: int = 100,
        window_time_sec: Optional[float] = None,
        on_error: str = "skip",  # "skip", "log", or "raise"
    ) -> None:
        """
        Parameters
        ----------
        window_size : int
            Size of the sliding observation window for behavioral host aggregation.
        window_time_sec : Optional[float]
            Optional temporal threshold in seconds for sliding window expiration.
        on_error : str
            Error policy on unparseable / malformed flows: "skip", "log", or "raise".
        """
        self.parser = FlowParser()
        self.calculator = FeatureCalculator(
            window_size=window_size,
            window_time_sec=window_time_sec,
            parser=self.parser,
        )
        self.on_error = on_error.lower()

    def reset(self) -> None:
        """Resets the internal observation window state."""
        self.calculator.reset_window()

    def _format_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Constructs a DataFrame strictly ordered with CANONICAL_22_FEATURES."""
        if not records:
            return pd.DataFrame(columns=CANONICAL_22_FEATURES)

        df = pd.DataFrame(records)[CANONICAL_22_FEATURES]

        # Enforce exact dtypes expected by preprocessing/inference
        df["proto"] = df["proto"].astype(str)
        df["flgs"] = df["flgs"].astype(str)
        df["state"] = df["state"].astype(str)

        int_cols = [
            "sport", "dport", "pkts", "spkts", "dpkts",
            "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP",
            "TnP_PerProto", "TnP_Per_Dport", "N_IN_Conn_P_DstIP",
            "N_IN_Conn_P_SrcIP", "Pkts_P_State_P_Protocol_P_SrcIP",
        ]
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

        float_cols = ["dur", "mean", "stddev", "min", "rate"]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")

        return df

    def extract_single(self, raw_flow: Union[Dict[str, Any], ParsedFlow]) -> pd.DataFrame:
        """
        Extracts canonical 22 features for an individual network flow.

        Parameters
        ----------
        raw_flow : Dict[str, Any] or ParsedFlow
            Raw flow record or pre-parsed flow object.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame containing strictly the canonical 22 features.
        """
        features_dict = self.calculator.calculate_features(raw_flow, update_window=True)
        return self._format_dataframe([features_dict])

    def extract_single_dict(self, raw_flow: Union[Dict[str, Any], ParsedFlow]) -> Dict[str, Any]:
        """
        Extracts canonical 22 features as a raw dictionary (convenient for single dict inference).
        """
        return self.calculator.calculate_features(raw_flow, update_window=True)

    def extract_batch(
        self,
        raw_flows: Union[List[Dict[str, Any]], pd.DataFrame, Iterable[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """
        Extracts canonical 22 features across a batch of flow records.

        Parameters
        ----------
        raw_flows : list of dicts or pd.DataFrame
            Collection of network flow records.

        Returns
        -------
        pd.DataFrame
            DataFrame containing canonical 22 features in identical row order.
        """
        if isinstance(raw_flows, pd.DataFrame):
            records_iter = raw_flows.to_dict(orient="records")
        else:
            records_iter = raw_flows

        extracted: List[Dict[str, Any]] = []
        for record in records_iter:
            try:
                feat = self.calculator.calculate_features(record, update_window=True)
                extracted.append(feat)
            except Exception as exc:
                if self.on_error == "raise":
                    raise FeatureExtractionError(f"Feature calculation failed: {exc}") from exc
                if self.on_error == "log":
                    logger.warning("Skipping invalid flow record: %s", exc)
                continue

        return self._format_dataframe(extracted)

    def extract_stream(
        self,
        source: Union[str, Path, TextIO, Iterable[str]],
        delimiter: Optional[str] = None,
        chunk_size: int = 10,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Streams flow records incrementally from a file or stream in small batches.

        Ideal for low-memory Raspberry Pi execution.

        Parameters
        ----------
        source : str, Path, TextIO, or Iterable[str]
            Flow source.
        delimiter : Optional[str]
            Delimiter (e.g. ',' or '\t').
        chunk_size : int
            Number of flows to accumulate before emitting a DataFrame chunk.

        Yields
        ------
        pd.DataFrame
            DataFrame of size <= chunk_size containing canonical 22 features.
        """
        reader = ArgusReader(
            source=source,
            delimiter=delimiter,
            deduplicate=True,
            error_handling=self.on_error,
        )

        buffer: List[Dict[str, Any]] = []
        with reader:
            for raw_record in reader.read_records():
                try:
                    feat = self.calculator.calculate_features(raw_record, update_window=True)
                    buffer.append(feat)
                    if len(buffer) >= chunk_size:
                        yield self._format_dataframe(buffer)
                        buffer = []
                except Exception as exc:
                    if self.on_error == "raise":
                        raise FeatureExtractionError(f"Streaming extraction error: {exc}") from exc
                    if self.on_error == "log":
                        logger.warning("Error processing streaming flow: %s", exc)
                    continue

            if buffer:
                yield self._format_dataframe(buffer)

    def pipe_to_inference(
        self,
        pipeline: Any,
        raw_flows: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame, str, Path],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Bridges the feature extractor directly to the CascadedInferencePipeline.

        Flow:
        Raw Argus input -> Parser -> Calculator -> 22 Features DataFrame -> Pipeline Predict.
        """
        if isinstance(raw_flows, (str, Path)):
            # If path, stream and accumulate predictions
            all_preds: List[Dict[str, Any]] = []
            for df_chunk in self.extract_stream(raw_flows, chunk_size=50):
                preds = pipeline.predict_batch(df_chunk)
                all_preds.extend(preds)
            return all_preds

        if isinstance(raw_flows, dict):
            # Single flow
            feat_df = self.extract_single(raw_flows)
            # Use predict_single with dict
            return pipeline.predict_single(feat_df.iloc[0].to_dict())

        # Batch of flows
        feat_df = self.extract_batch(raw_flows)
        return pipeline.predict_batch(feat_df)
