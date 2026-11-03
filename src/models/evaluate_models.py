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
from src.models.cnn_utils import CNNCategoryPredictor

MODEL_PATHS = {
    "decision_tree": PROJECT_ROOT / "models" / "decision_tree_category.joblib",
    "lightgbm": PROJECT_ROOT / "models" / "lightgbm_category.joblib",
    "cnn": PROJECT_ROOT / "models" / "cnn_category.keras",
    "isolation_forest": PROJECT_ROOT / "models" / "isolation_forest_binary.joblib",
}


def evaluate_model(name: str, model, X: pd.DataFrame, y_true: pd.Series) -> dict:
    """Compute threshold and probability metrics for one fitted classifier."""
    y_proba = model.predict_proba(X)
    labels = model.classes_.tolist()
    y_pred = np.asarray(labels)[np.argmax(y_proba, axis=1)]
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


def evaluate_isolation_forest(model, X: pd.DataFrame, categories: pd.Series) -> dict:
    """Evaluate Isolation Forest with Normal=0 and Attack=1 ground truth."""
    y_true = categories.ne("Normal").astype(np.int8).to_numpy()
    raw_predictions = model.predict(X)
    y_pred = (raw_predictions == -1).astype(np.int8)
    anomaly_scores = -model.decision_function(X)
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    total = int(matrix.sum())
    per_class = {}
    for index, class_name in enumerate(("Normal", "Attack")):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        specificity = tn / (tn + fp) if tn + fp else 0.0
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall_sensitivity": float(recall[index]),
            "f1_score": float(f1[index]),
            "specificity": float(specificity),
            "false_positive_rate": float(1.0 - specificity),
            "false_negative_rate": float(1.0 - recall[index]),
            "support": int(support[index]),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        }

    roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    average_precision = float(average_precision_score(y_true, anomaly_scores))
    overall = {
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
        "roc_auc_from_anomaly_score": roc_auc,
        "average_precision_from_anomaly_score": average_precision,
    }
    return {
        "model": "isolation_forest",
        "target": "binary_normal_vs_attack",
        "evaluation_split": str(TEST_V2_CLEAN_PARQUET),
        "label_mapping": {"Normal": 0, "Attack": 1},
        "test_rows": int(len(categories)),
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {
            "labels_in_order": ["Normal (0)", "Attack (1)"],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["Normal", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score saved category models on the held-out test set; does not train."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PATHS),
        default=list(MODEL_PATHS),
        help="Models to evaluate (default: all four models).",
    )
    args = parser.parse_args()

    missing = [
        str(MODEL_PATHS[name])
        for name in args.models
        if not MODEL_PATHS[name].is_file()
    ]
    if "cnn" in args.models:
        cnn_preprocessors = PROJECT_ROOT / "models" / "cnn_category_preprocessors.joblib"
        if not cnn_preprocessors.is_file():
            missing.append(str(cnn_preprocessors))
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
        if name == "isolation_forest":
            model = joblib.load(MODEL_PATHS[name])
            metrics = evaluate_isolation_forest(model, X_test, test["category"])
        elif name == "cnn":
            import tensorflow as tf

            network = tf.keras.models.load_model(MODEL_PATHS[name])
            preprocessors = joblib.load(cnn_preprocessors)
            model = CNNCategoryPredictor(
                network,
                preprocessors["numeric_transformer"],
                preprocessors["categorical_encoder"],
                preprocessors["label_encoder"],
            )
        else:
            model = joblib.load(MODEL_PATHS[name])
            metrics = evaluate_model(name, model, X_test, y_test)
        report_target = "binary" if name == "isolation_forest" else "category"
        report_path = report_dir / f"{name}_{report_target}_test_metrics.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        comparison_rows.append(
            {"model": name, "target": metrics["target"], **metrics["overall"]}
        )
        print(f"  Accuracy: {metrics['overall']['accuracy']:.6f}")
        print(f"  Balanced accuracy: {metrics['overall']['balanced_accuracy']:.6f}")
        print(f"  Macro F1: {metrics['overall']['f1_macro']:.6f}")
        print(f"  Saved: {report_path}")

    comparison_path = report_dir / "model_test_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"Saved model comparison: {comparison_path}")
    print("No model was fitted or tuned; the test split was used only for evaluation.")


if __name__ == "__main__":
    main()
