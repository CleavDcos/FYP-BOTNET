"""
Standardized Preprocessing Pipeline Module
"""

import time
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import (
    RAW_CSV_FILES,
    PREPROCESSED_PARQUET,
    PREPROCESSING_REPORTS_DIR,
)

CHUNKSIZE = 250000

# Feature Exclusions
EXCLUDE_LEAKAGE_IDENTIFIERS = [
    "pkSeqID",
    "seq",
    "stime",
    "ltime",
    "saddr",
    "daddr",
]

EXCLUDE_REDUNDANT_NUMERIC = [
    "proto_number",
    "flgs_number",
    "state_number",
]

TARGET_LABELS = [
    "attack",
    "category",
    "subcategory",
]

READ_DTYPES = {
    "sport": str,
    "dport": str,
    "drate": float,
    "rate": float,
    "srate": float,
    "dur": float,
    "mean": float,
    "stddev": float,
    "sum": float,
    "min": float,
    "max": float,
    "AR_P_Proto_P_SrcIP": float,
    "AR_P_Proto_P_DstIP": float,
    "AR_P_Proto_P_Sport": float,
    "AR_P_Proto_P_Dport": float,
    "proto": str,
    "flgs": str,
    "state": str,
    "attack": int,
    "category": str,
    "subcategory": str,
    "pkts": "int64",
    "bytes": "int64",
    "spkts": "int64",
    "dpkts": "int64",
    "sbytes": "int64",
    "dbytes": "int64",
    "TnBPSrcIP": "int64",
    "TnBPDstIP": "int64",
    "TnP_PSrcIP": "int64",
    "TnP_PDstIP": "int64",
    "TnP_PerProto": "int64",
    "TnP_Per_Dport": "int64",
    "N_IN_Conn_P_DstIP": "int64",
    "N_IN_Conn_P_SrcIP": "int64",
    "Pkts_P_State_P_Protocol_P_DestIP": "int64",
    "Pkts_P_State_P_Protocol_P_SrcIP": "int64",
}


def clean_port_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()
    is_hex = s_str.str.startswith(("0x", "0X"))
    res = pd.Series(-1, index=s.index, dtype="int64")

    if is_hex.any():
        res[is_hex] = s_str[is_hex].apply(lambda x: int(x, 16)).astype("int64")
    if (~is_hex).any():
        res[~is_hex] = pd.to_numeric(s_str[~is_hex], errors="coerce").fillna(-1).astype("int64")

    return res


def run_clean_pipeline():
    print("Starting Preprocessing Pipeline...")
    start_time = time.time()

    first_csv = RAW_CSV_FILES[0]
    all_raw_cols = pd.read_csv(first_csv, nrows=1).columns.tolist()

    cols_to_exclude = set(EXCLUDE_LEAKAGE_IDENTIFIERS + EXCLUDE_REDUNDANT_NUMERIC)
    cols_to_load = [c for c in all_raw_cols if c not in cols_to_exclude]
    chunk_read_dtypes = {k: v for k, v in READ_DTYPES.items() if k in cols_to_load}

    total_rows = 0
    parquet_writer = None

    for fpath in RAW_CSV_FILES:
        print(f"  Streaming {fpath.name}...")
        for chunk in pd.read_csv(
            fpath,
            usecols=cols_to_load,
            dtype=chunk_read_dtypes,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            total_rows += len(chunk)
            chunk["sport"] = clean_port_series(chunk["sport"])
            chunk["dport"] = clean_port_series(chunk["dport"])
            chunk["drate"] = chunk["drate"].astype("float64")

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    str(PREPROCESSED_PARQUET),
                    table.schema,
                    compression="snappy",
                )
            parquet_writer.write_table(table)

    if parquet_writer is not None:
        parquet_writer.close()

    print(f"Preprocessed parquet saved to {PREPROCESSED_PARQUET} ({total_rows:,} rows in {time.time() - start_time:.2f}s)")


if __name__ == "__main__":
    run_clean_pipeline()
