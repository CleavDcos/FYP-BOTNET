"""
Master Execution Script to Reproduce All Preprocessing Stages
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.audit import run_audit
from src.preprocessing.preprocess import run_clean_pipeline
from src.preprocessing.split import run_split
from src.preprocessing.imbalance import run_feature_reduction_and_resampling


def main():
    print("=" * 80)
    print("REPRODUCIBLE END-TO-END PREPROCESSING PIPELINE")
    print("=" * 80)
    t0 = time.time()

    print("\n--- STAGE 1: Dataset Audit ---")
    run_audit()

    print("\n--- STAGE 2: Standardized Preprocessing & Cleaning ---")
    run_clean_pipeline()

    print("\n--- STAGE 3: Stratified 70/15/15 Splitting ---")
    run_split()

    print("\n--- STAGE 4: Feature Reduction & Imbalance Handling (v2 Clean) ---")
    run_feature_reduction_and_resampling()

    print("\n" + "=" * 80)
    print(f"ALL PREPROCESSING STAGES COMPLETED SUCCESSFULLY IN {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
