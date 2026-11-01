"""Evaluate saved category models once on the held-out test split."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config.settings import (
    CANONICAL_22_FEATURES,
    PROJECT_ROOT,
    TEST_V2_CLEAN_PARQUET,
)

MODEL_PATHS = {
    "decision_tree": PROJECT_ROOT / "models" / "decision_tree_category.joblib",
    "lightgbm": PROJECT_ROOT / "models" / "lightgbm_category.joblib",
}


def evaluate_model(name: str, model, X: pd.DataFrame, y_true: pd.Series) -> dict:
    """Compute threshold and probability metrics for one fitted classifier."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    labels = model.classes_.tolist()
    y_true_binarized = label_binarize(y_true, classes=labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    total = int(matrix.sum())
    class_metrics = {}
    per_class_auc = {}
    per_class_average_precision = {}
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        )
        class_metrics[str(label)] = {
            "precision": float(precision[index]),
            "recall_sensitivity": float(recall[index]),
            "f1_score": float(f1[index]),
            "specificity": float(specificity),
            "false_positive_rate": float(1.0 - specificity),
            "false_negative_rate": float(1.0 - recall[index]),
            "support": int(support[index]),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": int(true_negative),
            "false_negative": false_negative,
        }
        per_class_auc[str(label)] = float(
            roc_auc_score(y_true_binarized[:, index], y_proba[:, index])
        )
        per_class_average_precision[str(label)] = float(
            average_precision_score(y_true_binarized[:, index], y_proba[:, index])
        )

    metrics = {
        "model": name,
        "target": "category",
        "evaluation_split": str(TEST_V2_CLEAN_PARQUET),
        "test_rows": int(len(y_true)),
        "features": CANONICAL_22_FEATURES,
        "labels": [str(label) for label in labels],
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
            "roc_auc_ovr_macro": float(roc_auc_score(y_true_binarized, y_proba, average="macro", multi_class="ovr")),
            "roc_auc_ovr_weighted": float(roc_auc_score(y_true_binarized, y_proba, average="weighted", multi_class="ovr")),
            "average_precision_macro": float(average_precision_score(y_true_binarized, y_proba, average="macro")),
            "average_precision_weighted": float(average_precision_score(y_true_binarized, y_proba, average="weighted")),
        },
        "per_class_roc_auc_ovr": per_class_auc,
        "per_class_average_precision": per_class_average_precision,
        "per_class": class_metrics,
        "confusion_matrix": {
            "labels_in_order": [str(label) for label in labels],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score saved category models on the held-out test set; does not train."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PATHS),
        default=list(MODEL_PATHS),
        help="Models to evaluate (default: both).",
    )
    args = parser.parse_args()

    missing = [str(MODEL_PATHS[name]) for name in args.models if not MODEL_PATHS[name].is_file()]
    if missing:
        parser.error("Missing trained model file(s): " + ", ".join(missing))
    if not TEST_V2_CLEAN_PARQUET.is_file():
        parser.error(f"Test split not found: {TEST_V2_CLEAN_PARQUET}")

    required_columns = CANONICAL_22_FEATURES + ["category"]
    print(f"Loading held-out test split: {TEST_V2_CLEAN_PARQUET}", flush=True)
    test = pd.read_parquet(TEST_V2_CLEAN_PARQUET, columns=required_columns)
    X_test = test[CANONICAL_22_FEATURES]
    y_test = test["category"]
    report_dir = PROJECT_ROOT / "reports" / "experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []

    for name in args.models:
        print(f"Evaluating {name} on {len(test):,} test rows...", flush=True)
        model = joblib.load(MODEL_PATHS[name])
        metrics = evaluate_model(name, model, X_test, y_test)
        report_path = report_dir / f"{name}_category_test_metrics.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        comparison_rows.append({"model": name, **metrics["overall"]})
        print(f"  Accuracy: {metrics['overall']['accuracy']:.6f}")
        print(f"  Balanced accuracy: {metrics['overall']['balanced_accuracy']:.6f}")
        print(f"  Macro F1: {metrics['overall']['f1_macro']:.6f}")
        print(f"  Saved: {report_path}")

    comparison_path = report_dir / "category_model_test_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"Saved model comparison: {comparison_path}")
    print("No model was fitted or tuned; the test split was used only for evaluation.")


if __name__ == "__main__":
    main()
