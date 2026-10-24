"""
Controlled Binary XGBoost Imbalance Weighting Experiments
Stage 5: ML-based IoT Botnet Attack Detection (Normal vs Attack)

Evaluates:
  1. compute_sample_weight('balanced', y_train)
  2. scale_pos_weight tuning targeting minority class weighting
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
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

MODELS_DIR = PROJECT_ROOT / "models"


def load_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads train and val clean datasets, extracting canonical 22 features
    and separating binary 'attack' target label (zero leakage).
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


def evaluate_binary_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics (0: Normal, 1: Attack)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    normal_precision = float(prec_per_class[0])
    normal_recall = float(rec_per_class[0])
    normal_f1 = float(f1_per_class[0])

    attack_precision = float(prec_per_class[1])
    attack_recall = float(rec_per_class[1])
    attack_f1 = float(f1_per_class[1])

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # PR-AUC for minority/positive class (Attack)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_attack = float(auc(recall_vals, precision_vals))

    # PR-AUC for Normal class (Class 0)
    precision_norm, recall_norm, _ = precision_recall_curve((y_true == 0).astype(int), 1.0 - y_proba)
    pr_auc_normal = float(auc(recall_norm, precision_norm))

    roc_auc = float(roc_auc_score(y_true, y_proba))
    clf_report = classification_report(y_true, y_pred, target_names=["Normal (0)", "Attack (1)"], output_dict=True)

    return {
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
            "support": int((y_true == 0).sum()),
            "pr_auc": pr_auc_normal,
        },
        "attack_class_metrics": {
            "precision": attack_precision,
            "recall": attack_recall,
            "f1_score": attack_f1,
            "support": int((y_true == 1).sum()),
            "pr_auc": pr_auc_attack,
        },
        "confusion_matrix": {
            "true_negatives_normal": tn,
            "false_positives_normal_flagged_as_attack": fp,
            "false_negatives_attack_missed": fn,
            "true_positives_attack_detected": tp,
            "matrix_format": cm,
            "labels_order": ["Normal (0)", "Attack (1)"],
        },
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "classification_report": clf_report,
    }


def extract_feature_importances(preprocessor, classifier: xgb.XGBClassifier) -> Dict[str, float]:
    """Extracts feature importances mapped to preprocessed feature names."""
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    return {
        name: float(imp) for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    }


def run_single_binary_experiment(
    candidate_id: str,
    description: str,
    weighting_type: str,
    scale_pos_weight: Optional[float],
    X_train_trans: np.ndarray,
    y_train: pd.Series,
    X_val_trans: np.ndarray,
    y_val: pd.Series,
    preprocessor,
) -> Dict[str, Any]:
    """Runs a single binary XGBoost configuration."""
    print(f"\n--- Running Candidate: {candidate_id} ({description}) ---")

    xgb_config = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    if scale_pos_weight is not None:
        xgb_config["scale_pos_weight"] = scale_pos_weight

    classifier = xgb.XGBClassifier(**xgb_config)

    # Compute sample weights if sample_weight strategy
    sample_weight = None
    if weighting_type == "sample_weight_balanced":
        sample_weight = compute_sample_weight("balanced", y_train)

    train_start = time.time()
    if sample_weight is not None:
        classifier.fit(X_train_trans, y_train, sample_weight=sample_weight)
    else:
        classifier.fit(X_train_trans, y_train)
    train_duration = time.time() - train_start

    eval_start = time.time()
    y_pred = classifier.predict(X_val_trans)
    y_proba = classifier.predict_proba(X_val_trans)[:, 1]
    eval_duration = time.time() - eval_start

    metrics = evaluate_binary_predictions(y_val, y_pred, y_proba)
    feature_importances = extract_feature_importances(preprocessor, classifier)

    # Serialize candidate pipeline bundle
    save_filename = f"xgb_binary_{candidate_id}_pipeline.joblib"
    save_path = MODELS_DIR / save_filename
    bundle = {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "candidate_id": candidate_id,
        "hyperparameters": xgb_config,
    }
    joblib.dump(bundle, save_path)
    model_size_mb = save_path.stat().st_size / (1024 * 1024)

    print(f"Completed in {train_duration:.2f}s train, {eval_duration:.2f}s eval. Accuracy: {metrics['accuracy']*100:.4f}%, Normal F1: {metrics['normal_class_metrics']['f1_score']:.6f}, FPR: {metrics['false_positive_rate']*100:.4f}%")

    return {
        "candidate_id": candidate_id,
        "description": description,
        "weighting_type": weighting_type,
        "scale_pos_weight": scale_pos_weight,
        "hyperparameters": xgb_config,
        "training_time_seconds": round(train_duration, 2),
        "validation_evaluation_time_seconds": round(eval_duration, 2),
        "model_save_path": str(save_path.relative_to(PROJECT_ROOT)),
        "model_size_mb": round(model_size_mb, 2),
        "metrics": metrics,
        "feature_importances": feature_importances,
    }


def run_binary_weighting_experiments():
    """Runs controlled binary imbalance weighting experiments."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 5: CONTROLLED BINARY XGBOOST IMBALANCE EXPERIMENTS")
    print("=" * 70)

    # 1. Load Data
    X_train, y_train, X_val, y_val = load_datasets()
    train_dist = {str(k): int(v) for k, v in y_train.value_counts().items()}
    val_dist = {str(k): int(v) for k, v in y_val.value_counts().items()}
    print(f"Loaded datasets. Train: {train_dist}, Val: {val_dist}")

    # 2. Fit Feature Preprocessor Exclusively on Training Data
    print("\nFitting feature preprocessor on training data...")
    prep_start = time.time()
    preprocessor = build_feature_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    print(f"Preprocessing completed in {time.time() - prep_start:.2f}s. Transformed shape: {X_train_trans.shape}")

    # Theoretical exact negative/positive ratio: 334 / 2567631 = ~0.000130081
    # Scale_pos_weight is weight of class 1 relative to class 0.
    ratio_exact = 334.0 / 2567631.0

    candidates_def = [
        {
            "id": "sample_weight_balanced",
            "description": "compute_sample_weight('balanced', y_train) on training samples",
            "weighting_type": "sample_weight_balanced",
            "scale_pos_weight": None,
        },
        {
            "id": "scale_pos_exact_ratio",
            "description": f"scale_pos_weight={ratio_exact:.8f} (N_neg / N_pos = 334 / 2567631)",
            "weighting_type": "scale_pos_weight",
            "scale_pos_weight": ratio_exact,
        },
        {
            "id": "scale_pos_mild_dampened",
            "description": "scale_pos_weight=0.001 (mild dampening of attack class weight)",
            "weighting_type": "scale_pos_weight",
            "scale_pos_weight": 0.001,
        },
        {
            "id": "scale_pos_moderate_dampened",
            "description": "scale_pos_weight=0.01 (moderate dampening of attack class weight)",
            "weighting_type": "scale_pos_weight",
            "scale_pos_weight": 0.01,
        },
    ]

    candidate_results = {}
    for c in candidates_def:
        candidate_results[c["id"]] = run_single_binary_experiment(
            candidate_id=c["id"],
            description=c["description"],
            weighting_type=c["weighting_type"],
            scale_pos_weight=c["scale_pos_weight"],
            X_train_trans=X_train_trans,
            y_train=y_train,
            X_val_trans=X_val_trans,
            y_val=y_val,
            preprocessor=preprocessor,
        )

    # 3. Load Baselines for Comprehensive Comparison
    rf_report_path = EXPERIMENTS_REPORTS_DIR / "rf_binary_experiment.json"
    unweighted_xgb_report_path = EXPERIMENTS_REPORTS_DIR / "xgb_binary_experiment.json"

    with open(rf_report_path, "r", encoding="utf-8") as f:
        rf_baseline = json.load(f)
    with open(unweighted_xgb_report_path, "r", encoding="utf-8") as f:
        unweighted_xgb_baseline = json.load(f)

    # 4. Save JSON Report
    full_report = {
        "experiment": "Stage 5 Binary XGBoost Controlled Imbalance Weighting Experiment",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_class_counts": train_dist,
        "val_class_counts": val_dist,
        "baselines": {
            "random_forest_balanced": {
                "model_name": "RandomForestClassifier",
                "class_weight": "balanced",
                "training_time_seconds": rf_baseline["training_time_seconds"],
                "metrics": rf_baseline["metrics"],
            },
            "xgboost_unweighted": {
                "model_name": "XGBClassifier",
                "class_weight": "unweighted (default)",
                "training_time_seconds": unweighted_xgb_baseline["training_time_seconds"],
                "metrics": unweighted_xgb_baseline["metrics"],
            },
        },
        "candidates": candidate_results,
    }

    json_report_path = EXPERIMENTS_REPORTS_DIR / "xgb_binary_imbalance_experiment.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nExperiment comparison JSON saved to {json_report_path}.")

    # 5. Save Markdown Summary
    md_report_path = EXPERIMENTS_REPORTS_DIR / "xgb_binary_imbalance_experiment.md"
    generate_markdown_report(full_report, md_report_path)
    print(f"Experiment comparison Markdown saved to {md_report_path}.")

    print("\n" + "=" * 70)
    print("STAGE 5 BINARY XGBOOST IMBALANCE EXPERIMENTS COMPLETED.")
    print("=" * 70)


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path):
    """Formats and writes markdown report."""
    rf_base = report_data["baselines"]["random_forest_balanced"]["metrics"]
    xgb_base = report_data["baselines"]["xgboost_unweighted"]["metrics"]
    cands = report_data["candidates"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Stage 5: Binary XGBoost Imbalance Weighting Experiment\n\n")
        f.write("Controlled evaluation of class-weighting and sample-weighting strategies for binary botnet detection (`attack`: 0=Normal, 1=Attack).\n\n")
        f.write(f"- **Training Dataset**: `botnet_train_v2_clean.parquet` ({report_data['train_samples']:,} samples: Normal={report_data['train_class_counts']['0']:,}, Attack={report_data['train_class_counts']['1']:,})\n")
        f.write(f"- **Validation Dataset**: `botnet_val_v2_clean.parquet` ({report_data['val_samples']:,} samples: Normal={report_data['val_class_counts']['0']:,}, Attack={report_data['val_class_counts']['1']:,})\n")
        f.write("- **Test Dataset**: Completely untouched and never loaded.\n\n")

        f.write("## Overall Performance Comparison\n\n")
        f.write("| Model / Configuration | Accuracy | Normal Prec / Rec / F1 | Attack Prec / Rec / F1 | Macro-F1 | Weighted-F1 | FPR | ROC-AUC | Train Time |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        # RF Baseline
        f.write(f"| **RF (Balanced Baseline)** | {rf_base['accuracy']*100:.4f}% | {rf_base['normal_class_metrics']['precision']:.4f} / {rf_base['normal_class_metrics']['recall']:.4f} / {rf_base['normal_class_metrics']['f1_score']:.4f} | {rf_base['attack_class_metrics']['precision']:.4f} / {rf_base['attack_class_metrics']['recall']:.4f} / {rf_base['attack_class_metrics']['f1_score']:.4f} | {rf_base['f1_macro']:.6f} | {rf_base['f1_weighted']:.6f} | {rf_base['false_positive_rate']*100:.4f}% | {rf_base['roc_auc']:.6f} | {report_data['baselines']['random_forest_balanced']['training_time_seconds']}s |\n")

        # Unweighted XGB Baseline
        f.write(f"| **XGB (Unweighted Baseline)** | {xgb_base['accuracy']*100:.4f}% | {xgb_base['normal_class_metrics']['precision']:.4f} / {xgb_base['normal_class_metrics']['recall']:.4f} / {xgb_base['normal_class_metrics']['f1_score']:.4f} | {xgb_base['attack_class_metrics']['precision']:.4f} / {xgb_base['attack_class_metrics']['recall']:.4f} / {xgb_base['attack_class_metrics']['f1_score']:.4f} | {xgb_base['f1_macro']:.6f} | {xgb_base['f1_weighted']:.6f} | {xgb_base['false_positive_rate']*100:.4f}% | {xgb_base['roc_auc']:.6f} | {report_data['baselines']['xgboost_unweighted']['training_time_seconds']}s |\n")

        # Candidates
        for cand_id, cand_data in cands.items():
            m = cand_data["metrics"]
            f.write(f"| **XGB ({cand_id})** | {m['accuracy']*100:.4f}% | {m['normal_class_metrics']['precision']:.4f} / {m['normal_class_metrics']['recall']:.4f} / {m['normal_class_metrics']['f1_score']:.4f} | {m['attack_class_metrics']['precision']:.4f} / {m['attack_class_metrics']['recall']:.4f} / {m['attack_class_metrics']['f1_score']:.4f} | {m['f1_macro']:.6f} | {m['f1_weighted']:.6f} | {m['false_positive_rate']*100:.4f}% | {m['roc_auc']:.6f} | {cand_data['training_time_seconds']}s |\n")

        f.write("\n## Confusion Matrices\n\n")
        f.write("Order: `[[TN (Normal), FP], [FN, TP (Attack)]]`\n\n")

        f.write(f"### Baseline: Random Forest (Balanced)\n")
        f.write(f"- TN: {rf_base['confusion_matrix']['true_negatives_normal']}, FP: {rf_base['confusion_matrix']['false_positives_normal_flagged_as_attack']}, FN: {rf_base['confusion_matrix']['false_negatives_attack_missed']}, TP: {rf_base['confusion_matrix']['true_positives_attack_detected']}\n\n")

        f.write(f"### Baseline: XGBoost (Unweighted)\n")
        f.write(f"- TN: {xgb_base['confusion_matrix']['true_negatives_normal']}, FP: {xgb_base['confusion_matrix']['false_positives_normal_flagged_as_attack']}, FN: {xgb_base['confusion_matrix']['false_negatives_attack_missed']}, TP: {xgb_base['confusion_matrix']['true_positives_attack_detected']}\n\n")

        for cand_id, cand_data in cands.items():
            cm = cand_data["metrics"]["confusion_matrix"]
            f.write(f"### Candidate: XGBoost ({cand_id})\n")
            f.write(f"- Description: {cand_data['description']}\n")
            f.write(f"- TN: {cm['true_negatives_normal']}, FP: {cm['false_positives_normal_flagged_as_attack']}, FN: {cm['false_negatives_attack_missed']}, TP: {cm['true_positives_attack_detected']}\n\n")


if __name__ == "__main__":
    run_binary_weighting_experiments()
