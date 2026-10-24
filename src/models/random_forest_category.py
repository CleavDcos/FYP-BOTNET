"""
Random Forest Attack-Category Multiclass Classification Model Training & Evaluation
Stage 5B: Attack-Category Classification (DDoS, DoS, Reconnaissance, Theft)
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

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

# Explicit canonical attack categories expected in malicious flows
EXPECTED_ATTACK_CATEGORIES: List[str] = ["DDoS", "DoS", "Reconnaissance", "Theft"]

# Minimum sample threshold to flag rare/underrepresented classes
MIN_SAMPLES_RELIABLE_EVAL: int = 50


def load_malicious_category_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads train and val clean datasets, filters strictly for malicious records (attack == 1),
    extracts the canonical 22 features, and targets 'category' (zero target leakage).
    Validates presence of all expected attack categories.
    """
    print(f"Loading training data from {TRAIN_V2_CLEAN_PARQUET}...")
    df_train_raw = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET)
    malicious_train_mask = df_train_raw["attack"] == 1
    df_train_mal = df_train_raw[malicious_train_mask].copy()

    X_train = df_train_mal[CANONICAL_22_FEATURES].copy()
    y_train = df_train_mal["category"].astype(str).copy()

    print(f"Loading validation data from {VAL_V2_CLEAN_PARQUET}...")
    df_val_raw = pd.read_parquet(VAL_V2_CLEAN_PARQUET)
    malicious_val_mask = df_val_raw["attack"] == 1
    df_val_mal = df_val_raw[malicious_val_mask].copy()

    X_val = df_val_mal[CANONICAL_22_FEATURES].copy()
    y_val = df_val_mal["category"].astype(str).copy()

    # Zero leakage assertions
    assert list(X_train.columns) == CANONICAL_22_FEATURES, "X_train features mismatch canonical order"
    assert list(X_val.columns) == CANONICAL_22_FEATURES, "X_val features mismatch canonical order"
    for target in TARGET_LABELS:
        assert target not in X_train.columns, f"Leakage detected: {target} found in X_train"
        assert target not in X_val.columns, f"Leakage detected: {target} found in X_val"

    # Category completeness validation
    train_cats = set(y_train.unique())
    val_cats = set(y_val.unique())
    for cat in EXPECTED_ATTACK_CATEGORIES:
        if cat not in train_cats:
            raise ValueError(f"Expected attack category '{cat}' missing from training malicious data.")
        if cat not in val_cats:
            raise ValueError(f"Expected attack category '{cat}' missing from validation malicious data.")

    # Verify no 'Normal' records leak into malicious classification
    assert "Normal" not in train_cats, "Contamination: 'Normal' found in malicious training set"
    assert "Normal" not in val_cats, "Contamination: 'Normal' found in malicious validation set"

    return X_train, y_train, X_val, y_val


def evaluate_category_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_names: List[str] = EXPECTED_ATTACK_CATEGORIES,
) -> Dict[str, Any]:
    """
    Comprehensive multiclass evaluation:
    Overall accuracy, macro/weighted precision/recall/F1, per-class metrics,
    confusion matrix, and flags for underrepresented classes.
    """
    print("Evaluating multiclass model on malicious validation set...")
    y_pred = pipeline.predict(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    prec_macro = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

    # Confusion matrix with explicit labels
    cm = confusion_matrix(y_val, y_pred, labels=target_names).tolist()

    # Per-class metrics
    prec_per_class = precision_score(y_val, y_pred, labels=target_names, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_val, y_pred, labels=target_names, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_val, y_pred, labels=target_names, average=None, zero_division=0).tolist()

    per_class_metrics = {}
    underrepresented_flags = {}

    for idx, cat in enumerate(target_names):
        support = int((y_val == cat).sum())
        is_reliable = support >= MIN_SAMPLES_RELIABLE_EVAL
        per_class_metrics[cat] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
            "support": support,
            "reliable_evaluation": is_reliable,
        }
        if not is_reliable:
            underrepresented_flags[cat] = (
                f"Support of {support} samples is below the reliability threshold ({MIN_SAMPLES_RELIABLE_EVAL}). "
                "Metrics should be interpreted with caution due to high variance."
            )

    clf_report = classification_report(
        y_val,
        y_pred,
        labels=target_names,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "overall_accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "per_class_metrics": per_class_metrics,
        "underrepresented_classes_flags": underrepresented_flags,
        "confusion_matrix": {
            "labels": target_names,
            "matrix": cm,
        },
        "classification_report": clf_report,
    }
    return metrics


def extract_feature_importances(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """Extracts and sorts feature importances from the fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    fi_list = [
        {"feature": str(name), "importance": round(float(imp), 6)}
        for name, imp in zip(feature_names, importances)
    ]
    fi_list.sort(key=lambda x: x["importance"], reverse=True)
    return fi_list


def train_and_evaluate_category_random_forest():
    """Main execution routine for Stage 5B multiclass attack-category training and evaluation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STAGE 5B: Random Forest Multiclass Attack-Category Classification")
    print("=" * 75)

    # 1. Load Data
    data_start = time.time()
    X_train, y_train, X_val, y_val = load_malicious_category_datasets()
    print(f"Loaded malicious datasets in {time.time() - data_start:.2f}s.")

    train_counts = dict(y_train.value_counts())
    val_counts = dict(y_val.value_counts())
    print(f"X_train malicious shape: {X_train.shape}, y_train distribution: {train_counts}")
    print(f"X_val malicious shape: {X_val.shape}, y_val distribution: {val_counts}")

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

    # 3. Fit Pipeline (Transformations fitted strictly on training data only)
    print("\nFitting Multiclass Random Forest Pipeline on training data...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    training_duration_seconds = time.time() - train_start
    print(f"Training completed in {training_duration_seconds:.2f} seconds ({training_duration_seconds / 60:.2f} mins).")

    # 4. Evaluate on Malicious Validation Set (No oversampling, strictly clean val data)
    eval_start = time.time()
    metrics = evaluate_category_model(pipeline, X_val, y_val, target_names=EXPECTED_ATTACK_CATEGORIES)
    eval_duration_seconds = time.time() - eval_start
    print(f"Validation evaluation completed in {eval_duration_seconds:.2f}s.")

    # 5. Extract Feature Importance
    feature_importances = extract_feature_importances(pipeline)

    # 6. Save Model and Pipeline (Outside Git Tracking)
    model_save_path = MODELS_DIR / "rf_category_pipeline.joblib"
    print(f"\nSaving trained category pipeline to {model_save_path}...")
    joblib.dump(pipeline, model_save_path)
    model_size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"Category model saved successfully. File size: {model_size_mb:.2f} MB.")

    # 7. Save Experiment Record in reports/experiments/
    experiment_record = {
        "model_name": "RandomForestClassifier",
        "stage": "5B",
        "task": "attack_category_classification",
        "target_column": "category",
        "target_categories": EXPECTED_ATTACK_CATEGORIES,
        "features": CANONICAL_22_FEATURES,
        "feature_count": len(CANONICAL_22_FEATURES),
        "train_dataset": str(TRAIN_V2_CLEAN_PARQUET.name),
        "val_dataset": str(VAL_V2_CLEAN_PARQUET.name),
        "train_category_counts": {str(k): int(v) for k, v in train_counts.items()},
        "val_category_counts": {str(k): int(v) for k, v in val_counts.items()},
        "model_hyperparameters": rf_config,
        "training_time_seconds": round(training_duration_seconds, 2),
        "validation_evaluation_time_seconds": round(eval_duration_seconds, 2),
        "model_save_path": str(model_save_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "feature_importances_top_15": feature_importances[:15],
        "all_feature_importances": feature_importances,
    }

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_report_path = EXPERIMENTS_REPORTS_DIR / "rf_category_experiment.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2, default=convert_numpy)
    print(f"Experiment metrics saved to {json_report_path}.")

    # Write Markdown summary
    md_report_path = EXPERIMENTS_REPORTS_DIR / "rf_category_experiment.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 5B: Random Forest Attack-Category Classification Experiment\n\n")
        f.write("- **Task**: Multiclass Attack-Category Classification (Malicious traffic only)\n")
        f.write(f"- **Target Categories**: {', '.join(EXPECTED_ATTACK_CATEGORIES)}\n")
        f.write(f"- **Input Features**: 22 Canonical Features from `config/settings.py`\n")
        f.write(f"- **Training Dataset**: `{TRAIN_V2_CLEAN_PARQUET.name}` ({len(X_train):,} malicious samples)\n")
        f.write(f"- **Validation Dataset**: `{VAL_V2_CLEAN_PARQUET.name}` ({len(X_val):,} malicious samples)\n")
        f.write(f"- **Training Time**: {training_duration_seconds:.2f}s ({training_duration_seconds / 60:.2f} mins)\n")
        f.write(f"- **Evaluation Time**: {eval_duration_seconds:.2f}s\n")
        f.write(f"- **Model Saved Path**: `{model_save_path.relative_to(PROJECT_ROOT)}` ({model_size_mb:.2f} MB)\n\n")

        f.write("## Data Sample Distribution\n\n")
        f.write("| Category | Training Count | Validation Count | Representation Status |\n")
        f.write("|---|---|---|---|\n")
        for cat in EXPECTED_ATTACK_CATEGORIES:
            tr_c = train_counts.get(cat, 0)
            val_c = val_counts.get(cat, 0)
            status = "Standard" if val_c >= MIN_SAMPLES_RELIABLE_EVAL else f"⚠️ Rare Class (< {MIN_SAMPLES_RELIABLE_EVAL} samples)"
            f.write(f"| **{cat}** | {tr_c:,} | {val_c:,} | {status} |\n")
        f.write("\n")

        f.write("## Validation Performance Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| **Overall Accuracy** | {metrics['overall_accuracy'] * 100:.4f}% |\n")
        f.write(f"| **Macro Precision** | {metrics['precision_macro']:.6f} |\n")
        f.write(f"| **Macro Recall** | {metrics['recall_macro']:.6f} |\n")
        f.write(f"| **Macro F1-Score** | {metrics['f1_macro']:.6f} |\n")
        f.write(f"| **Weighted Precision** | {metrics['precision_weighted']:.6f} |\n")
        f.write(f"| **Weighted Recall** | {metrics['recall_weighted']:.6f} |\n")
        f.write(f"| **Weighted F1-Score** | {metrics['f1_weighted']:.6f} |\n\n")

        f.write("### Per-Class Metrics\n\n")
        f.write("| Category | Support | Precision | Recall | F1-Score | Reliable Evaluation |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cat in EXPECTED_ATTACK_CATEGORIES:
            cm_cat = metrics["per_class_metrics"][cat]
            rel_str = "Yes" if cm_cat["reliable_evaluation"] else "⚠️ No (High variance due to low support)"
            f.write(
                f"| **{cat}** | {cm_cat['support']:,} | {cm_cat['precision']:.6f} | "
                f"{cm_cat['recall']:.6f} | {cm_cat['f1_score']:.6f} | {rel_str} |\n"
            )
        f.write("\n")

        if metrics["underrepresented_classes_flags"]:
            f.write("### Data Imbalance & Evaluation Caveats\n\n")
            for cat, flag_msg in metrics["underrepresented_classes_flags"].items():
                f.write(f"- **{cat}**: {flag_msg}\n")
            f.write("\n")

        f.write("### Confusion Matrix\n\n")
        f.write(f"Predicted Categories across columns: `{', '.join(EXPECTED_ATTACK_CATEGORIES)}`\n\n")
        f.write("| Actual \\ Predicted | " + " | ".join(EXPECTED_ATTACK_CATEGORIES) + " |\n")
        f.write("|---" * (len(EXPECTED_ATTACK_CATEGORIES) + 1) + "|\n")
        for idx, actual_cat in enumerate(EXPECTED_ATTACK_CATEGORIES):
            row_vals = metrics["confusion_matrix"]["matrix"][idx]
            f.write(f"| **{actual_cat}** | " + " | ".join(f"{v:,}" for v in row_vals) + " |\n")
        f.write("\n")

        f.write("### Top 15 Feature Importances\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|---|---|---|\n")
        for rank, fi in enumerate(feature_importances[:15], 1):
            f.write(f"| {rank} | `{fi['feature']}` | {fi['importance']:.6f} |\n")
        f.write("\n")

    print(f"Experiment markdown summary saved to {md_report_path}.")
    print("\n" + "=" * 75)
    print("STAGE 5B MULTICLASS RANDOM FOREST TRAINING & EVALUATION COMPLETED.")
    print("=" * 75)


if __name__ == "__main__":
    train_and_evaluate_category_random_forest()
