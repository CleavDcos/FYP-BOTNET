"""
Global Path and Feature Configurations for IoT Botnet Detection Project
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Reports & Metrics Directories
REPORTS_DIR = PROJECT_ROOT / "reports"
PREPROCESSING_REPORTS_DIR = REPORTS_DIR / "preprocessing"
EXPERIMENTS_REPORTS_DIR = REPORTS_DIR / "experiments"

# Configuration Directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Random Seed
RANDOM_SEED = 42

# Canonical 22 Selected Input Features (Strict Canonical Order)
CANONICAL_22_FEATURES = [
    # Categoricals (3)
    "proto",
    "flgs",
    "state",
    # Flow Ports (2)
    "sport",
    "dport",
    # Packet Counts (3)
    "pkts",
    "spkts",
    "dpkts",
    # Duration & Inter-arrival Time Statistics (4)
    "dur",
    "mean",
    "stddev",
    "min",
    # Flow Rate & Host Window Behavioral Metrics (7)
    "rate",
    "TnBPSrcIP",
    "TnBPDstIP",
    "TnP_PSrcIP",
    "TnP_PDstIP",
    "TnP_PerProto",
    "TnP_Per_Dport",
    # Connection Density Metrics (3)
    "N_IN_Conn_P_DstIP",
    "N_IN_Conn_P_SrcIP",
    "Pkts_P_State_P_Protocol_P_SrcIP",
]

# Excluded 12 Features (for 34 vs 22 Benchmarking)
EXCLUDED_12_FEATURES = [
    "bytes",
    "sbytes",
    "dbytes",
    "sum",
    "srate",
    "drate",
    "AR_P_Proto_P_Dport",
    "max",
    "Pkts_P_State_P_Protocol_P_DestIP",
    "AR_P_Proto_P_Sport",
    "AR_P_Proto_P_SrcIP",
    "AR_P_Proto_P_DstIP",
]

# Target Labels (Never Included in Feature Matrices)
TARGET_LABELS = ["attack", "category", "subcategory"]

# Raw CSV File Names
RAW_CSV_FILES = [
    RAW_DATA_DIR / f"UNSW_2018_IoT_Botnet_Full5pc_{i}.csv" for i in range(1, 5)
]

# Processed Parquet File Paths
PREPROCESSED_PARQUET = PROCESSED_DATA_DIR / "botnet_preprocessed.parquet"

# Original 34-feature splits
TRAIN_34_PARQUET = PROCESSED_DATA_DIR / "botnet_train.parquet"
VAL_34_PARQUET = PROCESSED_DATA_DIR / "botnet_val.parquet"
TEST_34_PARQUET = PROCESSED_DATA_DIR / "botnet_test.parquet"

# Clean 22-feature versioned splits
TRAIN_V2_CLEAN_PARQUET = PROCESSED_DATA_DIR / "botnet_train_v2_clean.parquet"
VAL_V2_CLEAN_PARQUET = PROCESSED_DATA_DIR / "botnet_val_v2_clean.parquet"
TEST_V2_CLEAN_PARQUET = PROCESSED_DATA_DIR / "botnet_test_v2_clean.parquet"

# Resampled Category Training Datasets
TRAIN_V2_ROS_CAT_PARQUET = PROCESSED_DATA_DIR / "botnet_train_v2_ros_category.parquet"
TRAIN_V2_SMOTE_CAT_PARQUET = PROCESSED_DATA_DIR / "botnet_train_v2_smote_category.parquet"

# Resampled Binary Training Datasets
TRAIN_V2_ROS_BIN_PARQUET = PROCESSED_DATA_DIR / "botnet_train_v2_ros_binary.parquet"
TRAIN_V2_SMOTE_BIN_PARQUET = PROCESSED_DATA_DIR / "botnet_train_v2_smote_binary.parquet"

# Loss Weights
CLASS_WEIGHTS_V2_JSON = CONFIG_DIR / "class_weights_v2.json"
