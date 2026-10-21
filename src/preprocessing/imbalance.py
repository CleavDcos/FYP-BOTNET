"""
Feature Reduction and Train-Only Class Imbalance Module (v2 Clean Pipeline)
"""

import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.utils.class_weight import compute_class_weight

from config.settings import (
    TRAIN_34_PARQUET,
    VAL_34_PARQUET,
    TEST_34_PARQUET,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
    TEST_V2_CLEAN_PARQUET,
    TRAIN_V2_ROS_CAT_PARQUET,
    TRAIN_V2_SMOTE_CAT_PARQUET,
    TRAIN_V2_ROS_BIN_PARQUET,
    TRAIN_V2_SMOTE_BIN_PARQUET,
    CLASS_WEIGHTS_V2_JSON,
    CANONICAL_22_FEATURES,
    TARGET_LABELS,
    RANDOM_SEED,
)


def run_feature_reduction_and_resampling():
    start_time = time.time()
    clean_split_columns = CANONICAL_22_FEATURES + TARGET_LABELS

    print("1. Standardizing Train, Val, and Test to Canonical 22 Features...")
    df_train_raw = pd.read_parquet(TRAIN_34_PARQUET)
    df_train_clean = df_train_raw[clean_split_columns].copy()
    df_train_clean.to_parquet(TRAIN_V2_CLEAN_PARQUET, index=False, compression="snappy")

    df_val_raw = pd.read_parquet(VAL_34_PARQUET)
    df_val_clean = df_val_raw[clean_split_columns].copy()
    df_val_clean.to_parquet(VAL_V2_CLEAN_PARQUET, index=False, compression="snappy")

    df_test_raw = pd.read_parquet(TEST_34_PARQUET)
    df_test_clean = df_test_raw[clean_split_columns].copy()
    df_test_clean.to_parquet(TEST_V2_CLEAN_PARQUET, index=False, compression="snappy")

    print("2. Isolating Feature Matrix X from Target Labels (Zero Leakage)...")
    X_train = df_train_clean[CANONICAL_22_FEATURES].copy()
    y_train_cat = df_train_clean["category"].copy()
    y_train_bin = df_train_clean["attack"].copy()
    y_train_subcat = df_train_clean["subcategory"].copy()

    print("3. Precomputing Cost-Sensitive Class Weights...")
    classes_bin = np.array([0, 1])
    w_bin = compute_class_weight("balanced", classes=classes_bin, y=y_train_bin)
    class_weights_bin = {int(c): round(float(w), 4) for c, w in zip(classes_bin, w_bin)}

    classes_cat = np.unique(y_train_cat)
    w_cat = compute_class_weight("balanced", classes=classes_cat, y=y_train_cat)
    class_weights_cat = {str(c): round(float(w), 4) for c, w in zip(classes_cat, w_cat)}

    classes_subcat = np.unique(y_train_subcat)
    w_subcat = compute_class_weight("balanced", classes=classes_subcat, y=y_train_subcat)
    class_weights_subcat = {str(c): round(float(w), 4) for c, w in zip(classes_subcat, w_subcat)}

    weights_v2 = {
        "binary_weights": class_weights_bin,
        "category_weights": class_weights_cat,
        "subcategory_weights": class_weights_subcat,
    }
    with open(CLASS_WEIGHTS_V2_JSON, "w", encoding="utf-8") as f:
        json.dump(weights_v2, f, indent=2)

    print("4. Executing Clean RandomOverSampler (Category & Binary)...")
    cat_counts_orig = y_train_cat.value_counts().to_dict()
    ros_cat_strategy = {
        "DDoS": cat_counts_orig["DDoS"],
        "DoS": cat_counts_orig["DoS"],
        "Reconnaissance": cat_counts_orig["Reconnaissance"],
        "Normal": 10000,
        "Theft": 5000,
    }
    ros_cat = RandomOverSampler(sampling_strategy=ros_cat_strategy, random_state=RANDOM_SEED)
    X_ros_cat, y_ros_cat = ros_cat.fit_resample(X_train, y_train_cat)
    df_ros_cat = X_ros_cat.copy()
    df_ros_cat["category"] = y_ros_cat
    df_ros_cat.to_parquet(TRAIN_V2_ROS_CAT_PARQUET, index=False, compression="snappy")

    ros_bin_strategy = {1: (y_train_bin == 1).sum(), 0: 10000}
    ros_bin = RandomOverSampler(sampling_strategy=ros_bin_strategy, random_state=RANDOM_SEED)
    X_ros_bin, y_ros_bin = ros_bin.fit_resample(X_train, y_train_bin)
    df_ros_bin = X_ros_bin.copy()
    df_ros_bin["attack"] = y_ros_bin
    df_ros_bin.to_parquet(TRAIN_V2_ROS_BIN_PARQUET, index=False, compression="snappy")

    print("5. Executing Clean SMOTENC (Category & Binary)...")
    cat_cols = ["proto", "flgs", "state"]
    cat_indices = [CANONICAL_22_FEATURES.index(c) for c in cat_cols]

    smotenc_cat = SMOTENC(
        categorical_features=cat_indices,
        sampling_strategy=ros_cat_strategy,
        k_neighbors=5,
        random_state=RANDOM_SEED,
    )
    X_smote_cat, y_smote_cat = smotenc_cat.fit_resample(X_train, y_train_cat)
    df_smote_cat = X_smote_cat.copy()
    df_smote_cat["category"] = y_smote_cat
    df_smote_cat.to_parquet(TRAIN_V2_SMOTE_CAT_PARQUET, index=False, compression="snappy")

    smotenc_bin = SMOTENC(
        categorical_features=cat_indices,
        sampling_strategy=ros_bin_strategy,
        k_neighbors=5,
        random_state=RANDOM_SEED,
    )
    X_smote_bin, y_smote_bin = smotenc_bin.fit_resample(X_train, y_train_bin)
    df_smote_bin = X_smote_bin.copy()
    df_smote_bin["attack"] = y_smote_bin
    df_smote_bin.to_parquet(TRAIN_V2_SMOTE_BIN_PARQUET, index=False, compression="snappy")

    print(f"Resampling completed successfully in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    run_feature_reduction_and_resampling()
