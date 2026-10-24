"""
Random Forest Binary Detection Model Training & Evaluation
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
)

from config.settings import (
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
    EXPERIMENTS_REPORTS_DIR,
    CANONICAL_22_FEATURES,
    TARGET_LABELS,
    RANDOM_SEED,
    PROJECT_ROOT,
)
from src.preprocessing.model_preprocessor import build_feature_preprocessor

# Save directory for serialized model artefacts (outside git tracking)
MODELS_DIR = PROJECT_ROOT / "models"


def load_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads train and val clean datasets, strictly extracting canonical 22 features
    and separating the binary 'attack' target label (zero leakage).
    """
    print(f"Loading training data from {TRAIN_V2_CLEAN_PARQUET}...")
    df_train = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET)
    X_train = df_train[CANONICAL_22_FEATURES].copy()
    y_train = df_train["attack"].astype(int).copy()

    print(f"Loading validation data from {VAL_V2_CLEAN_PARQUET}...")
    df_val = pd.read_parquet(VAL_V2_CLEAN_PARQUET)
    X_val = df_val[CANONICAL_22_FEATURES].copy()
    y_val = df_val["attack"].astype(int).copy()

    # Integrity verification
    assert list(X_train.columns) == CANONICAL_22_FEATURES, "X_train features mismatch canonical order"
    assert list(X_val.columns) == CANONICAL_22_FEATURES, "X_val features mismatch canonical order"
    for target in TARGET_LABELS:
        assert target not in X_train.columns, f"Leakage detected: {target} found in X_train"
        assert target not in X_val.columns, f"Leakage detected: {target} found in X_val"

    return X_train, y_train, X_val, y_val


def evaluate_binary_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of binary classification model:
    Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report, PR-AUC, ROC-AUC.
    """
    print("Evaluating model on validation set...")
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]  # Probability of attack (class 1)

    acc = float(accuracy_score(y_val, y_pred))
    prec_macro = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

    # Per-class metrics (Class 0: Normal, Class 1: Attack)
    prec_per_class = precision_score(y_val, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_val, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0).tolist()

    normal_precision = float(prec_per_class[0])
    normal_recall = float(rec_per_class[0])
    normal_f1 = float(f1_per_class[0])

    attack_precision = float(prec_per_class[1])
    attack_recall = float(rec_per_class[1])
    attack_f1 = float(f1_per_class[1])

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_val, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # PR-AUC for minority/positive class (Attack)
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
    pr_auc_attack = float(auc(recall_vals, precision_vals))

    # PR-AUC for Normal class (Class 0)
    y_proba_normal = pipeline.predict_proba(X_val)[:, 0]
    precision_norm, recall_norm, _ = precision_recall_curve((y_val == 0).astype(int), y_proba_normal)
    pr_auc_normal = float(auc(recall_norm, precision_norm))

    roc_auc = float(roc_auc_score(y_val, y_proba))
    clf_report = classification_report(y_val, y_pred, target_names=["Normal (0)", "Attack (1)"], output_dict=True)

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "normal_class_metrics": {
            "precision": normal_precision,
            "recall": normal_recall,
            "f1_score": normal_f1,
            "support": int((y_val == 0).sum()),
            "pr_auc": pr_auc_normal,
        },
        "attack_class_metrics": {
            "precision": attack_precision,
            "recall": attack_recall,
            "f1_score": attack_f1,
            "support": int((y_val == 1).sum()),
            "pr_auc": pr_auc_attack,
        },
        "confusion_matrix": {
            "true_negatives_normal": tn,
            "false_positives_normal_flagged_as_attack": fp,
            "false_negatives_attack_missed": fn,
            "true_positives_attack_detected": tp,
            "matrix_format": cm,
        },
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "classification_report": clf_report,
    }
    return metrics


def train_and_evaluate_random_forest():
    """Main execution routine for Stage 5 Random Forest training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 5: Random Forest Binary Classification (Normal vs Attack)")
    print("=" * 70)

    # 1. Load Data
    data_start = time.time()
    X_train, y_train, X_val, y_val = load_datasets()
    print(f"Loaded datasets in {time.time() - data_start:.2f}s.")
    print(f"X_train shape: {X_train.shape}, y_train distribution: {dict(y_train.value_counts())}")
    print(f"X_val shape: {X_val.shape}, y_val distribution: {dict(y_val.value_counts())}")

    # 2. Build Pipeline
    preprocessor = build_feature_preprocessor()
    rf_config = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": 1,
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_config)),
        ]
    )

    # 3. Fit Pipeline (Transformations fitted on training data only)
    print("\nFitting Random Forest Pipeline on training data...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    training_duration_seconds = time.time() - train_start
    print(f"Training completed in {training_duration_seconds:.2f} seconds ({training_duration_seconds / 60:.2f} mins).")

    # 4. Evaluate on Validation Set
    eval_start = time.time()
    metrics = evaluate_binary_model(pipeline, X_val, y_val)
    eval_duration_seconds = time.time() - eval_start
    print(f"Validation evaluation completed in {eval_duration_seconds:.2f}s.")

    # 5. Save Model and Pipeline (Outside Git Tracking)
    model_save_path = MODELS_DIR / "rf_binary_pipeline.joblib"
    print(f"\nSaving trained pipeline to {model_save_path}...")
    joblib.dump(pipeline, model_save_path)
    model_size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"Model saved successfully. File size: {model_size_mb:.2f} MB.")

    # 6. Save Experiment Record in reports/experiments/
    experiment_record = {
        "model_name": "RandomForestClassifier",
        "stage": 5,
        "task": "binary_detection",
        "target_column": "attack",
        "features": CANONICAL_22_FEATURES,
        "feature_count": len(CANONICAL_22_FEATURES),
        "train_dataset": str(TRAIN_V2_CLEAN_PARQUET.name),
        "val_dataset": str(VAL_V2_CLEAN_PARQUET.name),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "model_hyperparameters": rf_config,
        "training_time_seconds": round(training_duration_seconds, 2),
        "validation_evaluation_time_seconds": round(eval_duration_seconds, 2),
        "model_save_path": str(model_save_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }

    json_report_path = EXPERIMENTS_REPORTS_DIR / "rf_binary_experiment.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2)
    print(f"Experiment metrics saved to {json_report_path}.")

    # Write Markdown summary for scannability
    md_report_path = EXPERIMENTS_REPORTS_DIR / "rf_binary_experiment.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 5: Random Forest Binary Detection Experiment\n\n")
        f.write(f"- **Task**: Binary Botnet Detection (`attack`: 0=Normal, 1=Attack)\n")
        f.write(f"- **Input Features**: 22 Canonical Features from `config/settings.py`\n")
        f.write(f"- **Training Dataset**: `{TRAIN_V2_CLEAN_PARQUET.name}` ({len(X_train):,} samples)\n")
        f.write(f"- **Validation Dataset**: `{VAL_V2_CLEAN_PARQUET.name}` ({len(X_val):,} samples)\n")
        f.write(f"- **Training Time**: {training_duration_seconds:.2f}s ({training_duration_seconds / 60:.2f} mins)\n")
        f.write(f"- **Model Saved Path**: `{model_save_path.relative_to(PROJECT_ROOT)}` ({model_size_mb:.2f} MB)\n\n")
        f.write("## Validation Performance Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| **Overall Accuracy** | {metrics['accuracy'] * 100:.4f}% |\n")
        f.write(f"| **ROC-AUC** | {metrics['roc_auc']:.6f} |\n")
        f.write(f"| **Attack PR-AUC** | {metrics['attack_class_metrics']['pr_auc']:.6f} |\n")
        f.write(f"| **Normal PR-AUC** | {metrics['normal_class_metrics']['pr_auc']:.6f} |\n")
        f.write(f"| **False Positive Rate** | {metrics['false_positive_rate'] * 100:.4f}% |\n\n")
        f.write("### Per-Class Metrics\n\n")
        f.write(f"| Class | Support | Precision | Recall | F1-Score | PR-AUC |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        f.write(f"| **Normal (0)** | {metrics['normal_class_metrics']['support']} | {metrics['normal_class_metrics']['precision']:.6f} | {metrics['normal_class_metrics']['recall']:.6f} | {metrics['normal_class_metrics']['f1_score']:.6f} | {metrics['normal_class_metrics']['pr_auc']:.6f} |\n")
        f.write(f"| **Attack (1)** | {metrics['attack_class_metrics']['support']} | {metrics['attack_class_metrics']['precision']:.6f} | {metrics['attack_class_metrics']['recall']:.6f} | {metrics['attack_class_metrics']['f1_score']:.6f} | {metrics['attack_class_metrics']['pr_auc']:.6f} |\n\n")
        f.write("### Confusion Matrix\n\n")
        f.write(f"| True \\ Pred | Predicted Normal | Predicted Attack |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **Actual Normal** | {metrics['confusion_matrix']['true_negatives_normal']} (TN) | {metrics['confusion_matrix']['false_positives_normal_flagged_as_attack']} (FP) |\n")
        f.write(f"| **Actual Attack** | {metrics['confusion_matrix']['false_negatives_attack_missed']} (FN) | {metrics['confusion_matrix']['true_positives_attack_detected']} (TP) |\n\n")

    print(f"Experiment markdown summary saved to {md_report_path}.")
    print("\n" + "=" * 70)
    print("STAGE 5 RANDOM FOREST TRAINING AND EVALUATION COMPLETED.")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate_random_forest()
