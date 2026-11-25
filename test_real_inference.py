"""
Temporary real-world smoke test script for the Cascaded Inference Pipeline.

Validates:
1. Loading real validation data from data/processed/botnet_val_v2_clean.parquet.
2. Inspecting label columns and extracting real Normal and Malicious records.
3. Target leakage prevention (stripping target labels before inference).
4. Cascaded execution: LightGBM is SKIPPED for Normal and EXECUTED for Malicious.
5. Accurate prediction outputs, confidence scores, and concise PASS/FAIL summary.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import VAL_V2_CLEAN_PARQUET, TARGET_LABELS
from src.inference import CascadedInferencePipeline, InferenceError


def main() -> int:
    print("=" * 70)
    print("CASCADED INFERENCE PIPELINE: REAL-WORLD SMOKE TEST")
    print("=" * 70)

    # 1. Load validation dataset
    val_path = VAL_V2_CLEAN_PARQUET
    if not val_path.exists():
        print(f"ERROR: Validation dataset not found at {val_path}")
        return 1

    print(f"\n[1/5] Loading validation dataset: {val_path}...")
    try:
        df_val = pd.read_parquet(val_path)
    except Exception as exc:
        print(f"ERROR: Failed to read validation parquet: {exc}")
        return 1

    print(f"      Loaded {len(df_val):,} rows and {len(df_val.columns)} columns.")

    # 2. Inspect label columns and unique values
    print("\n[2/5] Inspecting actual label columns in dataset...")
    target_cols = [c for c in TARGET_LABELS if c in df_val.columns]
    if not target_cols:
        print(f"ERROR: None of the target columns {TARGET_LABELS} were found in the dataset.")
        return 1

    for col in target_cols:
        unique_vals = df_val[col].unique().tolist()
        val_counts = df_val[col].value_counts().to_dict()
        print(f"      Column '{col}': {len(unique_vals)} unique values -> {unique_vals}")
        print(f"        Distribution: {val_counts}")

    # 3. Select 1 Normal and 1 Malicious record by inspecting dataset semantics
    print("\n[3/5] Identifying Normal and Malicious samples...")
    # Determine Normal vs Malicious mask dynamically without assuming rigid encodings
    if "attack" in df_val.columns:
        # Check if attack values are 0/1 or string representations
        attack_vals = df_val["attack"].unique()
        if 0 in attack_vals and 1 in attack_vals:
            normal_mask = df_val["attack"] == 0
            malicious_mask = df_val["attack"] == 1
        elif "0" in attack_vals and "1" in attack_vals:
            normal_mask = df_val["attack"] == "0"
            malicious_mask = df_val["attack"] == "1"
        else:
            normal_mask = df_val["category"].astype(str).str.lower() == "normal"
            malicious_mask = df_val["category"].astype(str).str.lower() != "normal"
    elif "category" in df_val.columns:
        normal_mask = df_val["category"].astype(str).str.lower() == "normal"
        malicious_mask = df_val["category"].astype(str).str.lower() != "normal"
    else:
        print("ERROR: Could not identify target columns for binary distinction.")
        return 1

    normal_count = int(normal_mask.sum())
    malicious_count = int(malicious_mask.sum())
    print(f"      Found {normal_count:,} Normal flows and {malicious_count:,} Malicious flows.")

    if normal_count == 0:
        print("ERROR: Validation dataset does not contain any Normal traffic samples.")
        return 1
    if malicious_count == 0:
        print("ERROR: Validation dataset does not contain any Malicious traffic samples.")
        return 1

    # Extract 1 representative row from each class
    normal_row = df_val[normal_mask].iloc[0]
    malicious_row = df_val[malicious_mask].iloc[0]

    # Store actual labels for verification
    actual_normal_binary = normal_row["attack"] if "attack" in df_val.columns else "Normal"
    actual_normal_category = normal_row.get("category", "Normal")

    actual_malicious_binary = malicious_row["attack"] if "attack" in df_val.columns else "Malicious"
    actual_malicious_category = malicious_row.get("category", "Attack")

    # 4. Remove all target columns to guarantee zero target leakage
    print("\n[4/5] Preparing clean input records (stripping target labels)...")
    clean_normal_record = {
        col: normal_row[col] for col in normal_row.index if col not in TARGET_LABELS
    }
    clean_malicious_record = {
        col: malicious_row[col] for col in malicious_row.index if col not in TARGET_LABELS
    }

    # Verify zero leakage
    for target in TARGET_LABELS:
        assert target not in clean_normal_record, f"Leakage detected: {target} in normal record"
        assert target not in clean_malicious_record, f"Leakage detected: {target} in malicious record"

    print(f"      Normal record features: {len(clean_normal_record)} canonical features.")
    print(f"      Malicious record features: {len(clean_malicious_record)} canonical features.")

    # 5. Load pipeline and attach spy to verify LightGBM conditional execution
    print("\n[5/5] Executing Cascaded Inference Pipeline...")
    try:
        pipeline = CascadedInferencePipeline.from_loader()
    except Exception as exc:
        print(f"ERROR: Failed to load models or pipeline: {exc}")
        return 1

    # Wrap Stage 2 predict to spy on invocation counts
    original_lgb_predict = pipeline.lgb_predictor.predict
    spy_lgb_predict = MagicMock(side_effect=original_lgb_predict)
    pipeline.lgb_predictor.predict = spy_lgb_predict

    checks_passed = True
    failure_reasons = []

    # Run Normal Record
    print("\n--- Running Normal Traffic Inference ---")
    spy_lgb_predict.reset_mock()
    try:
        pred_normal = pipeline.predict_single(clean_normal_record)
    except Exception as exc:
        print(f"ERROR: Inference failed for Normal record: {exc}")
        return 1

    lgb_calls_normal = spy_lgb_predict.call_count
    print(f"  Actual Binary Label      : {actual_normal_binary}")
    print(f"  Actual Category          : {actual_normal_category}")
    print(f"  Predicted Binary Label   : {pred_normal.get('binary_prediction')}")
    print(f"  Predicted Category       : {pred_normal.get('attack_category')}")
    print(f"  Binary Confidence        : {pred_normal.get('binary_confidence')}")
    print(f"  Category Confidence      : {pred_normal.get('category_confidence')}")
    print(f"  LightGBM Executed        : {'YES (ERROR)' if lgb_calls_normal > 0 else 'NO (CORRECT - SKIPPED)'}")

    # Normal Verification Rules
    if pred_normal.get("binary_prediction") != "Normal" or pred_normal.get("attack_detected") is not False:
        checks_passed = False
        failure_reasons.append("Normal traffic was incorrectly flagged as Malicious.")
    if lgb_calls_normal != 0:
        checks_passed = False
        failure_reasons.append(f"LightGBM was executed {lgb_calls_normal} times for Normal traffic (expected 0).")
    if pred_normal.get("attack_category") is not None:
        checks_passed = False
        failure_reasons.append(f"attack_category for Normal should be None, got '{pred_normal.get('attack_category')}'.")

    # Run Malicious Record
    print("\n--- Running Malicious Traffic Inference ---")
    spy_lgb_predict.reset_mock()
    try:
        pred_malicious = pipeline.predict_single(clean_malicious_record)
    except Exception as exc:
        print(f"ERROR: Inference failed for Malicious record: {exc}")
        return 1

    lgb_calls_malicious = spy_lgb_predict.call_count
    print(f"  Actual Binary Label      : {actual_malicious_binary}")
    print(f"  Actual Category          : {actual_malicious_category}")
    print(f"  Predicted Binary Label   : {pred_malicious.get('binary_prediction')}")
    print(f"  Predicted Category       : {pred_malicious.get('attack_category')}")
    print(f"  Binary Confidence        : {pred_malicious.get('binary_confidence')}")
    print(f"  Category Confidence      : {pred_malicious.get('category_confidence')}")
    print(f"  LightGBM Executed        : {'YES (CORRECT - EXECUTED)' if lgb_calls_malicious == 1 else f'UNEXPECTED ({lgb_calls_malicious} times)'}")

    # Malicious Verification Rules
    if pred_malicious.get("binary_prediction") != "Malicious" or pred_malicious.get("attack_detected") is not True:
        checks_passed = False
        failure_reasons.append("Malicious traffic was incorrectly flagged as Normal.")
    if lgb_calls_malicious != 1:
        checks_passed = False
        failure_reasons.append(f"LightGBM was executed {lgb_calls_malicious} times for Malicious traffic (expected exactly 1).")
    if pred_malicious.get("attack_category") is None:
        checks_passed = False
        failure_reasons.append("attack_category for Malicious traffic was None (expected attack category name).")

    # Final Summary
    print("\n" + "=" * 70)
    if checks_passed:
        print("SMOKE TEST SUMMARY: PASS (All checks succeeded)")
        print("  ✓ Normal traffic correctly classified as Normal")
        print("  ✓ LightGBM strictly skipped for Normal traffic")
        print("  ✓ Malicious traffic correctly detected and categorized")
        print("  ✓ LightGBM strictly executed for Malicious traffic")
        print("  ✓ Zero target leakage verified (no target labels in input)")
        print("=" * 70)
        return 0
    else:
        print("SMOKE TEST SUMMARY: FAIL")
        for reason in failure_reasons:
            print(f"  ✗ {reason}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
