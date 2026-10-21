"""
Dataset Partitioning Module (Stratified 70/15/15)
"""

import time
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    PREPROCESSED_PARQUET,
    TRAIN_34_PARQUET,
    VAL_34_PARQUET,
    TEST_34_PARQUET,
    RANDOM_SEED,
)


def run_split():
    print(f"Loading {PREPROCESSED_PARQUET.name}...")
    df = pd.read_parquet(PREPROCESSED_PARQUET)
    total_rows = len(df)

    y_strat = df["subcategory"]
    idx = df.index

    print("Splitting 70% Train, 15% Val, 15% Test stratified by subcategory...")
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y_strat,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_strat.loc[temp_idx],
    )

    df.loc[train_idx].to_parquet(TRAIN_34_PARQUET, index=False, compression="snappy")
    df.loc[val_idx].to_parquet(VAL_34_PARQUET, index=False, compression="snappy")
    df.loc[test_idx].to_parquet(TEST_34_PARQUET, index=False, compression="snappy")

    print(f"Saved 34-feature splits: {len(train_idx):,} train, {len(val_idx):,} val, {len(test_idx):,} test.")


if __name__ == "__main__":
    run_split()
