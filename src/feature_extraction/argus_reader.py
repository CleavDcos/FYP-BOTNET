"""
Argus Flow Record Reader.

Reads network flow records from files or live text streams incrementally,
safely handling incomplete/malformed records, deduplicating flows, and
maintaining minimal memory footprint suitable for low-resource environments (e.g. Raspberry Pi).
"""

import csv
import io
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, TextIO, Union

logger = logging.getLogger(__name__)


class ArgusReaderError(Exception):
    """Exception raised for unrecoverable reader errors."""
    pass


class ArgusReader:
    """
    Incremental reader for Argus (ra/racluster) flow records.

    Features:
    - Incremental line-by-line reading (low memory generator).
    - Adaptable to CSV, TSV, or whitespace-delimited Argus outputs.
    - Handles incomplete, malformed, or blank records safely without crashing.
    - Bounded-memory deduplication to avoid re-processing identical flow records.
    - Supports files, standard input / pipes, and memory streams.
    """

    def __init__(
        self,
        source: Union[str, Path, TextIO, Iterable[str]],
        delimiter: Optional[str] = None,
        fieldnames: Optional[List[str]] = None,
        has_header: bool = True,
        deduplicate: bool = True,
        max_seen_cache: int = 5000,
        error_handling: str = "skip",  # "skip", "log", or "raise"
    ) -> None:
        """
        Parameters
        ----------
        source : str, Path, TextIO, or Iterable[str]
            Path to file, open file-like stream, or iterable of raw text lines.
        delimiter : Optional[str]
            Delimiter character (e.g. ',', '\t', or ' '). If None, auto-detected from header.
        fieldnames : Optional[List[str]]
            Explicit column names if input stream has no header line.
        has_header : bool
            Whether the first valid data line is a header row.
        deduplicate : bool
            Whether to track and filter duplicate records.
        max_seen_cache : int
            Maximum number of recent flow fingerprints to keep in memory for deduplication.
        error_handling : str
            Action on malformed lines: "skip" (silently skip), "log" (log and skip), or "raise".
        """
        self.source = source
        self.delimiter = delimiter
        self.fieldnames = [f.strip() for f in fieldnames] if fieldnames else None
        self.has_header = has_header
        self.deduplicate = deduplicate
        self.max_seen_cache = max(100, max_seen_cache)
        self.error_handling = error_handling.lower()

        self._seen_set: Set[int] = set()
        self._seen_queue: deque = deque(maxlen=self.max_seen_cache)
        self._is_file_owner = False
        self._file_handle: Optional[TextIO] = None

        # Statistics tracking
        self.stats = {
            "total_lines_read": 0,
            "valid_records_emitted": 0,
            "malformed_records_skipped": 0,
            "duplicate_records_skipped": 0,
            "comment_lines_skipped": 0,
        }

    def __enter__(self) -> "ArgusReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Closes the underlying file if opened by this reader."""
        if self._is_file_owner and self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _get_line_iterator(self) -> Generator[str, None, None]:
        """Provides an incremental line generator regardless of source type."""
        if isinstance(self.source, (str, Path)):
            path = Path(self.source)
            if not path.exists():
                raise ArgusReaderError(f"Flow source file does not exist: {path}")
            self._file_handle = open(path, "r", encoding="utf-8", errors="replace")
            self._is_file_owner = True
            for line in self._file_handle:
                yield line
        elif isinstance(self.source, io.TextIOBase) or hasattr(self.source, "readline"):
            for line in self.source:
                yield line
        elif isinstance(self.source, Iterable):
            for line in self.source:
                yield line
        else:
            raise ArgusReaderError(f"Unsupported source type: {type(self.source).__name__}")

    def _auto_detect_delimiter(self, sample_line: str) -> str:
        """Heuristically detects delimiter from a sample line."""
        if "," in sample_line:
            return ","
        if "\t" in sample_line:
            return "\t"
        if " " in sample_line:
            return " "
        return ","

    def _fingerprint_record(self, raw_line: str, record: Dict[str, Any]) -> int:
        """Computes a stable hash for record deduplication."""
        # Prefer domain key if common flow tuple is present
        tuple_keys = ("stime", "saddr", "sport", "daddr", "dport", "proto")
        if all(k in record for k in tuple_keys):
            key = tuple(record.get(k) for k in tuple_keys)
            return hash(key)
        # Fallback to normalized raw line hash
        return hash(" ".join(raw_line.strip().split()))

    def _record_seen(self, record_hash: int) -> bool:
        """Checks if hash was recently seen; if not, records it in bounded cache."""
        if record_hash in self._seen_set:
            return True
        if len(self._seen_queue) == self._seen_queue.maxlen:
            oldest = self._seen_queue.popleft()
            self._seen_set.discard(oldest)
        self._seen_queue.append(record_hash)
        self._seen_set.add(record_hash)
        return False

    def read_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields structured flow dictionaries incrementally.

        Yields
        ------
        Dict[str, Any]
            Raw key-value record mapping field names to raw string values.
        """
        line_iter = self._get_line_iterator()
        delimiter = self.delimiter

        # 1. Parse header if required
        if self.has_header and not self.fieldnames:
            for raw_line in line_iter:
                self.stats["total_lines_read"] += 1
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    self.stats["comment_lines_skipped"] += 1
                    continue

                if delimiter is None:
                    delimiter = self._auto_detect_delimiter(stripped)
                    self.delimiter = delimiter

                if delimiter == " ":
                    headers = [h.strip() for h in stripped.split() if h.strip()]
                else:
                    reader = csv.reader([stripped], delimiter=delimiter)
                    headers = [h.strip() for h in next(reader) if h.strip()]

                if not headers:
                    continue

                self.fieldnames = headers
                break

            if not self.fieldnames:
                return

        # 2. Stream records line by line
        for raw_line in line_iter:
            self.stats["total_lines_read"] += 1
            stripped = raw_line.strip()

            # Skip empty lines and comment lines
            if not stripped or stripped.startswith("#"):
                self.stats["comment_lines_skipped"] += 1
                continue

            if delimiter is None:
                delimiter = self._auto_detect_delimiter(stripped)
                self.delimiter = delimiter

            # Parse delimited line
            try:
                if delimiter == " ":
                    tokens = [t.strip() for t in stripped.split()]
                else:
                    reader = csv.reader([stripped], delimiter=delimiter)
                    tokens = [t.strip() for t in next(reader)]
            except Exception as exc:
                self.stats["malformed_records_skipped"] += 1
                if self.error_handling == "raise":
                    raise ArgusReaderError(f"Malformed record syntax: {raw_line}") from exc
                if self.error_handling == "log":
                    logger.warning("Skipping unparseable flow line: %s", raw_line[:80])
                continue

            # Validate column alignment
            if self.fieldnames:
                if len(tokens) != len(self.fieldnames):
                    self.stats["malformed_records_skipped"] += 1
                    if self.error_handling == "raise":
                        raise ArgusReaderError(
                            f"Column count mismatch: expected {len(self.fieldnames)} values, got {len(tokens)}."
                        )
                    if self.error_handling == "log":
                        logger.warning(
                            "Record length mismatch: expected %d, got %d. Line: %s",
                            len(self.fieldnames),
                            len(tokens),
                            raw_line[:60],
                        )
                    continue

                record = dict(zip(self.fieldnames, tokens))
            else:
                record = {f"col_{i}": val for i, val in enumerate(tokens)}

            # Deduplication check
            if self.deduplicate:
                rec_hash = self._fingerprint_record(raw_line, record)
                if self._record_seen(rec_hash):
                    self.stats["duplicate_records_skipped"] += 1
                    continue

            self.stats["valid_records_emitted"] += 1
            yield record
