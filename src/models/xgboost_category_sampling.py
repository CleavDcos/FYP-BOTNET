"""
XGBoost Attack-Category Classification Experiments across Sampling Strategies
Stage 5B: Attack-Category Classification (DDoS, DoS, Reconnaissance, Theft)
Comparing:
  1. Clean data with sample weighting (balanced strategy)
  2. Random Over-Sampling (ROS) category dataset
  3. SMOTENC category dataset
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
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
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
    TRAIN_V2_ROS_CAT_PARQUET,
    TRAIN_V2_SMOTE_CAT_PARQUET,
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

EXPECTED_ATTACK_CATEGORIES: List[str] = ["DDoS", "DoS", "Reconnaissance", "Theft"]
MIN_SAMPLES_RELIABLE_EVAL: int = 50


def load_validation_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads validation clean dataset, filtering strictly for malicious records (attack == 1 and category != 'Normal').
    Ensures validation set remains completely identical and untouched across all experiments.
    """
    print(f"Loading held-out validation data from {VAL_V2_CLEAN_PARQUET}...")
    df_val_raw = pd.read_parquet(VAL_V2_CLEAN_PARQUET)
    malicious_mask = (df_val_raw["attack"] == 1) & (df_val_raw["category"] != "Normal")
    df_val_mal = df_val_raw[malicious_mask].copy()

    X_val = df_val_mal[CANONICAL_22_FEATURES].copy()
    y_val = df_val_mal["category"].astype(str).copy()

    assert list(X_val.columns) == CANONICAL_22_FEATURES, "X_val features mismatch canonical order"
    for target in TARGET_LABELS:
        assert target not in X_val.columns, f"Leakage detected: {target} found in X_val"

    assert len(X_val) == 550206, f"Validation row count mismatch: expected 550,206, got {len(X_val)}"
    assert set(y_val.unique()) == set(EXPECTED_ATTACK_CATEGORIES), "Validation category mismatch"
    assert "Normal" not in y_val.unique(), "Contamination: 'Normal' found in validation category data"
    return X_val, y_val


def load_training_variant(variant: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads training dataset variant, strictly filtering for malicious categories
    (excluding 'Normal') and extracting canonical 22 features.
    """
    if variant == "clean":
        fpath = TRAIN_V2_CLEAN_PARQUET
        print(f"Loading Clean training data from {fpath}...")
        df_raw = pd.read_parquet(fpath)
        mask = (df_raw["attack"] == 1) & (df_raw["category"] != "Normal")
        df_mal = df_raw[mask].copy()
    elif variant == "ros":
        fpath = TRAIN_V2_ROS_CAT_PARQUET
        print(f"Loading ROS training data from {fpath}...")
        df_raw = pd.read_parquet(fpath)
        mask = df_raw["category"] != "Normal"
        df_mal = df_raw[mask].copy()
    elif variant == "smote":
        fpath = TRAIN_V2_SMOTE_CAT_PARQUET
        print(f"Loading SMOTENC training data from {fpath}...")
        df_raw = pd.read_parquet(fpath)
        mask = df_raw["category"] != "Normal"
        df_mal = df_raw[mask].copy()
    else:
        raise ValueError(f"Unknown training variant: {variant}")

    X_train = df_mal[CANONICAL_22_FEATURES].copy()
    y_train = df_mal["category"].astype(str).copy()

    # Zero leakage assertions
    assert list(X_train.columns) == CANONICAL_22_FEATURES, f"X_train ({variant}) features mismatch canonical order"
    for target in TARGET_LABELS:
        assert target not in X_train.columns, f"Leakage detected: {target} found in X_train ({variant})"

    train_cats = set(y_train.unique())
    assert train_cats == set(EXPECTED_ATTACK_CATEGORIES), f"Categories mismatch in {variant}: {train_cats}"
    assert "Normal" not in train_cats, f"'Normal' leaked into malicious training data ({variant})"

    return X_train, y_train


def evaluate_category_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_names: List[str] = EXPECTED_ATTACK_CATEGORIES,
) -> Dict[str, Any]:
    """Computes full multiclass evaluation metrics on predictions."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=target_names).tolist()

    prec_per_class = precision_score(y_true, y_pred, labels=target_names, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_true, y_pred, labels=target_names, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, labels=target_names, average=None, zero_division=0).tolist()

    per_class_metrics = {}
    underrepresented_flags = {}

    for idx, cat in enumerate(target_names):
        support = int((y_true == cat).sum())
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
                f"Support of {support} samples is below the reliability threshold ({MIN_SAMPLES_RELIABLE_EVAL})."
            )

    clf_report = classification_report(
        y_true,
        y_pred,
        labels=target_names,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
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


def extract_feature_importances(
    preprocessor,
    classifier: xgb.XGBClassifier,
) -> Dict[str, float]:
    """Extracts feature importances mapped to preprocessed feature names."""
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    feat_imp = {
        name: float(imp) for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    }
    return feat_imp


def run_experiment(
    variant_name: str,
    description: str,
    use_sample_weight: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """Executes a single end-to-end XGBoost category experiment."""
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {variant_name.upper()} - {description}")
    print("=" * 70)

    train_counts = {cat: int((y_train == cat).sum()) for cat in EXPECTED_ATTACK_CATEGORIES}
    print(f"Training set sample counts: {train_counts} (Total: {len(X_train):,})")

    # Encode labels to [0, 1, 2, 3] strictly matching label_encoder classes
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Calculate sample weights if clean weighted strategy
    sample_weight = None
    if use_sample_weight:
        print("Computing balanced sample weights for training data...")
        sample_weight = compute_sample_weight("balanced", y_train_encoded)
        print(f"Sample weights computed. Shape: {sample_weight.shape}")

    # Build feature preprocessor and fit exclusively on training data
    print("Fitting feature preprocessor on training data...")
    preprocessor = build_feature_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    xgb_config = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
    }

    classifier = xgb.XGBClassifier(**xgb_config)

    print(f"Training XGBClassifier ({variant_name})...")
    train_start = time.time()
    if sample_weight is not None:
        classifier.fit(X_train_trans, y_train_encoded, sample_weight=sample_weight)
    else:
        classifier.fit(X_train_trans, y_train_encoded)
    train_duration = time.time() - train_start
    print(f"Training completed in {train_duration:.2f}s ({train_duration / 60:.2f} mins).")

    print(f"Evaluating XGBClassifier ({variant_name}) on held-out validation set...")
    eval_start = time.time()
    y_pred_encoded = classifier.predict(X_val_trans)
    eval_duration = time.time() - eval_start
    print(f"Validation evaluation completed in {eval_duration:.2f}s.")

    # Decode predictions back to original string labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    metrics = evaluate_category_predictions(y_val, y_pred, target_names=EXPECTED_ATTACK_CATEGORIES)
    feature_importances = extract_feature_importances(preprocessor, classifier)

    # Save model pipeline bundle (preprocessor, classifier, label_encoder)
    model_save_filename = f"xgb_category_{variant_name}_pipeline.joblib"
    model_save_path = MODELS_DIR / model_save_filename
    pipeline_bundle = {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "label_encoder": label_encoder,
        "target_names": EXPECTED_ATTACK_CATEGORIES,
    }
    joblib.dump(pipeline_bundle, model_save_path)
    model_size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"Saved pipeline bundle to {model_save_path} ({model_size_mb:.2f} MB).")

    res = {
        "strategy": variant_name,
        "description": description,
        "training_samples": len(X_train),
        "training_counts": train_counts,
        "hyperparameters": xgb_config,
        "used_sample_weight": use_sample_weight,
        "training_time_seconds": round(train_duration, 2),
        "evaluation_time_seconds": round(eval_duration, 2),
        "model_path": str(model_save_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "feature_importances": feature_importances,
    }
    return res


def run_all_xgboost_category_experiments():
    """Runs all 3 XGBoost category experiments and produces JSON & Markdown reports."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 5B: XGBOOST ATTACK-CATEGORY EXPERIMENTS (CLEAN, ROS, SMOTE)")
    print("=" * 70)

    # Common validation dataset - loaded once, untouched across all experiments
    X_val, y_val = load_validation_data()
    val_counts = {cat: int((y_val == cat).sum()) for cat in EXPECTED_ATTACK_CATEGORIES}
    print(f"Validation dataset loaded ({len(X_val):,} samples): {val_counts}")

    # Establish canonical label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(EXPECTED_ATTACK_CATEGORIES)
    print(f"Label encoder classes: {list(label_encoder.classes_)}")

    results = {}

    # 1. Clean Weighted Experiment
    X_train_clean, y_train_clean = load_training_variant("clean")
    results["clean_weighted"] = run_experiment(
        variant_name="clean_weighted",
        description="Clean training data with compute_sample_weight('balanced')",
        use_sample_weight=True,
        X_train=X_train_clean,
        y_train=y_train_clean,
        X_val=X_val,
        y_val=y_val,
        label_encoder=label_encoder,
    )
    del X_train_clean, y_train_clean

    # 2. ROS Experiment
    X_train_ros, y_train_ros = load_training_variant("ros")
    results["ros"] = run_experiment(
        variant_name="ros",
        description="Random Over-Sampling (ROS) category dataset (Normal excluded)",
        use_sample_weight=False,
        X_train=X_train_ros,
        y_train=y_train_ros,
        X_val=X_val,
        y_val=y_val,
        label_encoder=label_encoder,
    )
    del X_train_ros, y_train_ros

    # 3. SMOTENC Experiment
    X_train_smote, y_train_smote = load_training_variant("smote")
    results["smote"] = run_experiment(
        variant_name="smote",
        description="SMOTENC synthetic oversampling category dataset (Normal excluded)",
        use_sample_weight=False,
        X_train=X_train_smote,
        y_train=y_train_smote,
        X_val=X_val,
        y_val=y_val,
        label_encoder=label_encoder,
    )
    del X_train_smote, y_train_smote

    # Generate Output JSON
    full_report = {
        "experiment": "Stage 5B XGBoost Attack-Category Experiments (Clean Weighted, ROS, SMOTE)",
        "model_architecture": "XGBClassifier (hist tree_method)",
        "validation_samples": len(X_val),
        "validation_counts": val_counts,
        "target_classes": EXPECTED_ATTACK_CATEGORIES,
        "experiments": results,
    }

    json_path = EXPERIMENTS_REPORTS_DIR / "xgb_category_sampling_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nExperiment comparison JSON saved to {json_path}.")

    # Generate Output Markdown
    md_path = EXPERIMENTS_REPORTS_DIR / "xgb_category_sampling_comparison.md"
    generate_markdown_report(full_report, md_path)
    print(f"Experiment comparison Markdown saved to {md_path}.")

    print("\n" + "=" * 70)
    print("STAGE 5B XGBOOST CATEGORY EXPERIMENTS COMPLETED.")
    print("=" * 70)


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path):
    """Formats and writes a comprehensive comparison Markdown report."""
    results = report_data["experiments"]
    val_counts = report_data["validation_counts"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Stage 5B: XGBoost Attack-Category Multiclass Experiments\n\n")
        f.write("Evaluation of XGBoost multiclass classifiers across three training data strategies:\n")
        f.write("1. **Clean Weighted**: Clean training data with balanced sample weighting.\n")
        f.write("2. **ROS**: Random Over-Sampling category training set (Normal excluded).\n")
        f.write("3. **SMOTENC**: SMOTENC synthetic oversampling category training set (Normal excluded).\n\n")
        f.write(f"- **Validation Set**: Held-out clean validation set ({report_data['validation_samples']:,} samples, strictly malicious flows)\n")
        f.write(f"- **Validation Distribution**: {val_counts}\n\n")

        f.write("## Overall Validation Performance Comparison\n\n")
        f.write("| Metric | Clean Weighted | ROS | SMOTENC |\n")
        f.write("|---|---|---|---|\n")

        f.write(f"| **Overall Accuracy** | {results['clean_weighted']['metrics']['overall_accuracy']*100:.4f}% | {results['ros']['metrics']['overall_accuracy']*100:.4f}% | {results['smote']['metrics']['overall_accuracy']*100:.4f}% |\n")
        f.write(f"| **Macro Precision** | {results['clean_weighted']['metrics']['precision_macro']:.6f} | {results['ros']['metrics']['precision_macro']:.6f} | {results['smote']['metrics']['precision_macro']:.6f} |\n")
        f.write(f"| **Macro Recall** | {results['clean_weighted']['metrics']['recall_macro']:.6f} | {results['ros']['metrics']['recall_macro']:.6f} | {results['smote']['metrics']['recall_macro']:.6f} |\n")
        f.write(f"| **Macro F1-Score** | {results['clean_weighted']['metrics']['f1_macro']:.6f} | {results['ros']['metrics']['f1_macro']:.6f} | {results['smote']['metrics']['f1_macro']:.6f} |\n")
        f.write(f"| **Weighted F1-Score** | {results['clean_weighted']['metrics']['f1_weighted']:.6f} | {results['ros']['metrics']['f1_weighted']:.6f} | {results['smote']['metrics']['f1_weighted']:.6f} |\n")
        f.write(f"| **Training Time** | {results['clean_weighted']['training_time_seconds']}s | {results['ros']['training_time_seconds']}s | {results['smote']['training_time_seconds']}s |\n")
        f.write(f"| **Evaluation Time** | {results['clean_weighted']['evaluation_time_seconds']}s | {results['ros']['evaluation_time_seconds']}s | {results['smote']['evaluation_time_seconds']}s |\n\n")

        f.write("## Per-Class F1-Score and Recall\n\n")
        f.write("| Category | Support | Clean Weighted (F1 / Rec) | ROS (F1 / Rec) | SMOTENC (F1 / Rec) |\n")
        f.write("|---|---|---|---|---|\n")

        for cat in EXPECTED_ATTACK_CATEGORIES:
            sup = val_counts[cat]
            c_f1 = results['clean_weighted']['metrics']['per_class_metrics'][cat]['f1_score']
            c_rec = results['clean_weighted']['metrics']['per_class_metrics'][cat]['recall']
            r_f1 = results['ros']['metrics']['per_class_metrics'][cat]['f1_score']
            r_rec = results['ros']['metrics']['per_class_metrics'][cat]['recall']
            s_f1 = results['smote']['metrics']['per_class_metrics'][cat]['f1_score']
            s_rec = results['smote']['metrics']['per_class_metrics'][cat]['recall']
            flag = " *(Underrepresented)*" if sup < MIN_SAMPLES_RELIABLE_EVAL else ""
            f.write(f"| **{cat}**{flag} | {sup:,} | {c_f1:.6f} / {c_rec:.6f} | {r_f1:.6f} / {r_rec:.6f} | {s_f1:.6f} / {s_rec:.6f} |\n")

        f.write("\n## Confusion Matrices\n\n")
        labels = report_data["target_classes"]

        for var_key, title in [("clean_weighted", "Clean Weighted"), ("ros", "ROS"), ("smote", "SMOTENC")]:
            cm = results[var_key]['metrics']['confusion_matrix']['matrix']
            f.write(f"### {title} Confusion Matrix (Order: {', '.join(labels)})\n\n")
            f.write(f"| True \\ Pred | " + " | ".join([f"Pred {lbl}" for lbl in labels]) + " |\n")
            f.write("|---" * (len(labels) + 1) + "|\n")
            for idx, row in enumerate(cm):
                row_str = " | ".join([f"{val:,}" for val in row])
                f.write(f"| **{labels[idx]}** | {row_str} |\n")
            f.write("\n")

        f.write("## Top Feature Importances (Clean Weighted Model)\n\n")
        f.write("| Feature | Importance |\n")
        f.write("|---|---|\n")
        for feat, score in list(results['clean_weighted']['feature_importances'].items())[:10]:
            f.write(f"| `{feat}` | {score:.6f} |\n")
        f.write("\n")


if __name__ == "__main__":
    run_all_xgboost_category_experiments()
