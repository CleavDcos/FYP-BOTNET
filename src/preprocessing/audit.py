"""
Dataset Audit and Inspection Module
"""

import time
import json
from pathlib import Path
from collections import Counter
import pandas as pd

from config.settings import (
    RAW_CSV_FILES,
    PREPROCESSING_REPORTS_DIR,
)

CHUNKSIZE = 100000


def run_audit():
    start_time = time.time()
    print("Starting Comprehensive Dataset Audit across raw CSV partitions...")

    file_stats = {}
    file_schemas = {}
    total_rows = 0

    overall_attack_counts = Counter()
    overall_category_counts = Counter()
    overall_subcategory_counts = Counter()
    overall_nulls = Counter()

    for fpath in RAW_CSV_FILES:
        if not fpath.exists():
            print(f"ERROR: {fpath.name} does not exist at {fpath}")
            continue

        fsize = fpath.stat().st_size
        print(f"\nProcessing {fpath.name} ({fsize:,} bytes)...")

        row_count = 0
        file_nulls = Counter()
        file_attack = Counter()
        file_cat = Counter()
        file_subcat = Counter()
        first_chunk = True
        schema_types = {}

        for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE, low_memory=False):
            n = len(chunk)
            row_count += n

            if first_chunk:
                schema_types = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
                first_chunk = False

            null_dict = chunk.isnull().sum().to_dict()
            for col, count in null_dict.items():
                file_nulls[col] += count
                overall_nulls[col] += count

            if "attack" in chunk.columns:
                file_attack.update(chunk["attack"].value_counts().to_dict())
                overall_attack_counts.update(chunk["attack"].value_counts().to_dict())
            if "category" in chunk.columns:
                file_cat.update(chunk["category"].value_counts().to_dict())
                overall_category_counts.update(chunk["category"].value_counts().to_dict())
            if "subcategory" in chunk.columns:
                file_subcat.update(chunk["subcategory"].value_counts().to_dict())
                overall_subcategory_counts.update(chunk["subcategory"].value_counts().to_dict())

        total_rows += row_count
        file_stats[fpath.name] = {
            "size_bytes": fsize,
            "size_mb": round(fsize / (1024 * 1024), 2),
            "rows": row_count,
            "cols": len(schema_types),
            "nulls": dict(file_nulls),
            "attack_dist": dict(file_attack),
            "category_dist": dict(file_cat),
            "subcategory_dist": dict(file_subcat),
        }
        file_schemas[fpath.name] = schema_types
        print(f"Done {fpath.name}: {row_count:,} rows.")

    metrics = {
        "file_stats": file_stats,
        "file_schemas": file_schemas,
        "total_rows": total_rows,
        "overall_attack_counts": dict(overall_attack_counts),
        "overall_category_counts": dict(overall_category_counts),
        "overall_subcategory_counts": dict(overall_subcategory_counts),
        "overall_nulls": dict(overall_nulls),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    metrics_out = PREPROCESSING_REPORTS_DIR / "audit_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAudit complete. Total rows: {total_rows:,}. Metrics saved to {metrics_out}")
    return metrics


if __name__ == "__main__":
    run_audit()
