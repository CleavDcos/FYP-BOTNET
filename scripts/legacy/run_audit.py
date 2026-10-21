import os
import sys
import time
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

DATA_DIR = Path(r"c:\Users\Cleaven Dcosta\OneDrive\Desktop\BOTNET")
FILES = [
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv",
]

CHUNKSIZE = 100000

def audit():
    start_time = time.time()
    print("Starting Comprehensive Dataset Audit...")

    file_stats = {}
    file_schemas = {}
    total_rows = 0
    
    # Global tracking
    overall_attack_counts = Counter()
    overall_category_counts = Counter()
    overall_subcategory_counts = Counter()
    
    # We will track null counts per column per file and overall
    overall_nulls = Counter()
    
    # Tracking min and max of numeric columns or set of values for potential constant columns
    col_summary = {}

    for fname in FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"ERROR: {fname} does not exist!")
            continue
        
        fsize = fpath.stat().st_size
        print(f"\nProcessing {fname} (Size: {fsize:,} bytes)...")
        
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
            
            # Nulls
            null_dict = chunk.isnull().sum().to_dict()
            for col, count in null_dict.items():
                file_nulls[col] += count
                overall_nulls[col] += count
                
            # Labels
            if 'attack' in chunk.columns:
                file_attack.update(chunk['attack'].value_counts().to_dict())
                overall_attack_counts.update(chunk['attack'].value_counts().to_dict())
            if 'category' in chunk.columns:
                file_cat.update(chunk['category'].value_counts().to_dict())
                overall_category_counts.update(chunk['category'].value_counts().to_dict())
            if 'subcategory' in chunk.columns:
                file_subcat.update(chunk['subcategory'].value_counts().to_dict())
                overall_subcategory_counts.update(chunk['subcategory'].value_counts().to_dict())

        total_rows += row_count
        file_stats[fname] = {
            "size_bytes": fsize,
            "size_mb": round(fsize / (1024 * 1024), 2),
            "rows": row_count,
            "cols": len(schema_types),
            "nulls": dict(file_nulls),
            "attack_dist": dict(file_attack),
            "category_dist": dict(file_cat),
            "subcategory_dist": dict(file_subcat),
        }
        file_schemas[fname] = schema_types
        print(f"Done {fname}: {row_count:,} rows.")

    print(f"\nTotal rows across all files: {total_rows:,}")

    # Save intermediate results
    results = {
        "file_stats": file_stats,
        "file_schemas": file_schemas,
        "total_rows": total_rows,
        "overall_attack_counts": dict(overall_attack_counts),
        "overall_category_counts": dict(overall_category_counts),
        "overall_subcategory_counts": dict(overall_subcategory_counts),
        "overall_nulls": dict(overall_nulls)
    }

    with open(DATA_DIR / "audit_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Elapsed: {time.time() - start_time:.2f}s. Saved audit_metrics.json.")

if __name__ == "__main__":
    audit()
