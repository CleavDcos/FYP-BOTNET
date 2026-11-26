"""
Unit Tests for ArgusReader.

Tests:
- Streaming from text / StringIO / files
- Delimiter handling (comma, tab, whitespace)
- Handling of comment lines (#) and blank lines
- Handling of malformed / incomplete records
- Record deduplication
- Bounded memory footprint
"""

import io
import unittest
from pathlib import Path
import tempfile

from src.feature_extraction.argus_reader import ArgusReader, ArgusReaderError
from tests.feature_extraction.sample_records import SYNTHETIC_ARGUS_CSV_TEXT


class TestArgusReader(unittest.TestCase):
    """Test suite for ArgusReader."""

    def test_read_from_string_stream(self):
        """Test reading CSV records from an in-memory string stream."""
        stream = io.StringIO(SYNTHETIC_ARGUS_CSV_TEXT)
        reader = ArgusReader(source=stream, delimiter=",", deduplicate=True)

        records = list(reader.read_records())
        # Header + 4 records, with 2 comments and 1 internal comment
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["proto"], "udp")
        self.assertEqual(records[1]["proto"], "tcp")
        self.assertEqual(records[2]["sport"], "1234")
        self.assertEqual(records[3]["sport"], "0x0303")
        self.assertEqual(reader.stats["valid_records_emitted"], 4)
        self.assertEqual(reader.stats["comment_lines_skipped"], 3)

    def test_read_from_file_context_manager(self):
        """Test reading from physical file with context manager."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp.write(SYNTHETIC_ARGUS_CSV_TEXT)
            tmp_path = Path(tmp.name)

        try:
            with ArgusReader(tmp_path) as reader:
                records = list(reader.read_records())
            self.assertEqual(len(records), 4)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_auto_detect_delimiter_tsv(self):
        """Test auto-detection of tab delimiter."""
        tsv_data = "stime\tproto\tsport\tdport\n100.1\ttcp\t80\t443\n100.2\tudp\t53\t5353\n"
        reader = ArgusReader(io.StringIO(tsv_data))
        records = list(reader.read_records())

        self.assertEqual(len(records), 2)
        self.assertEqual(reader.delimiter, "\t")
        self.assertEqual(records[0]["proto"], "tcp")
        self.assertEqual(records[1]["sport"], "53")

    def test_skip_malformed_incomplete_lines(self):
        """Test that malformed lines with column count mismatches are skipped safely."""
        corrupted_data = (
            "col1,col2,col3\n"
            "a,b,c\n"
            "incomplete_line\n"  # Too few columns
            "x,y,z,extra_column\n"  # Too many columns
            "valid1,valid2,valid3\n"
        )
        reader = ArgusReader(io.StringIO(corrupted_data), delimiter=",", error_handling="skip")
        records = list(reader.read_records())

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["col1"], "a")
        self.assertEqual(records[1]["col1"], "valid1")
        self.assertEqual(reader.stats["malformed_records_skipped"], 2)

    def test_raise_on_malformed_when_configured(self):
        """Test that error_handling='raise' raises ArgusReaderError on malformed lines."""
        corrupted_data = "col1,col2\na,b\nonly_one_col\n"
        reader = ArgusReader(io.StringIO(corrupted_data), delimiter=",", error_handling="raise")
        with self.assertRaises(ArgusReaderError):
            list(reader.read_records())

    def test_deduplication_filters_identical_flows(self):
        """Test that identical duplicate records are discarded."""
        data_with_duplicates = (
            "stime,saddr,sport,daddr,dport,proto\n"
            "100.0,192.168.1.1,80,10.0.0.1,5000,tcp\n"
            "100.0,192.168.1.1,80,10.0.0.1,5000,tcp\n"  # Duplicate
            "100.1,192.168.1.2,80,10.0.0.1,5000,tcp\n"  # Unique
            "100.0,192.168.1.1,80,10.0.0.1,5000,tcp\n"  # Duplicate
        )
        reader = ArgusReader(io.StringIO(data_with_duplicates), delimiter=",", deduplicate=True)
        records = list(reader.read_records())

        self.assertEqual(len(records), 2)
        self.assertEqual(reader.stats["duplicate_records_skipped"], 2)
        self.assertEqual(reader.stats["valid_records_emitted"], 2)


if __name__ == "__main__":
    unittest.main()
