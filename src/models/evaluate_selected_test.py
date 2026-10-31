"""One-time final evaluation of validation-selected models on the held-out test set."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    CANONICAL_22_FEATURES,
    EXPERIMENTS_REPORTS_DIR,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    TEST_V2_CLEAN_PARQUET,
)

MODELS_DIR = CONFIG_PROJECT_ROOT / "models"
CATEGORY_ORDER = ["DDoS", "DoS", "Reconnaissance", "Theft"]

# Selected using validation performance before looking at test results.
SELECTED_MODELS = {
    "binary": {
        "model_name": "LinearSVC",
        "path": MODELS_DIR / "svm_binary_pipeline.joblib",
        "target": "attack",
        "selection_basis": "Highest validation macro-F1 among the SVM and MLP binary candidates (0.825757).",
    },
    "category": {
        "model_name": "MLPClassifier",
        "path": MODELS_DIR / "mlp_category_pipeline.joblib",
        "target": "category",
        "selection_basis": "Highest validation macro-F1 among the SVM and MLP category candidates (0.910396).",
    },
}


def calculate_metrics(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "labels": [int(v) if isinstance(v, (np.integer, int)) else str(v) for v in labels],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    if labels == [0, 1]:
        tn, fp, fn, tp = matrix.ravel()
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if fp + tn else 0.0
        metrics["binary_confusion_counts"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    return metrics


def evaluate_task(task, test_frame):
    chosen = SELECTED_MODELS[task]
    if not chosen["path"].exists():
        raise FileNotFoundError(
            f"Model artifact is missing: {chosen['path']}. Run the SVM/MLP experiment script first."
        )

    frame = test_frame
    if task == "category":
        frame = frame[frame["attack"] == 1]
        labels = CATEGORY_ORDER
    else:
        labels = [0, 1]

    X_test = frame[CANONICAL_22_FEATURES]
    y_test = frame[chosen["target"]]
    pipeline = joblib.load(chosen["path"])
    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - start

    return {
        "model": chosen["model_name"],
        "model_path": str(chosen["path"].relative_to(CONFIG_PROJECT_ROOT)),
        "target": chosen["target"],
        "selection_basis": chosen["selection_basis"],
        "test_rows": int(len(frame)),
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "prediction_time_seconds": round(predict_seconds, 3),
        "metrics": calculate_metrics(y_test, y_pred, labels),
    }


def write_markdown(record, path):
    lines = [
        "# Final Test Evaluation: SVM and MLP",
        "",
        "These results use the held-out test split exactly once after candidate selection on validation. No model was fitted or tuned on test data.",
        "",
        "## Selected models and results",
        "",
        "| Task | Selected model | Test rows | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("binary", "category"):
        item = record["results"][task]
        m = item["metrics"]
        lines.append(
            f"| {task.title()} | {item['model']} | {item['test_rows']:,} | {m['accuracy']:.6f} | "
            f"{m['precision_macro']:.6f} | {m['recall_macro']:.6f} | {m['f1_macro']:.6f} | {m['f1_weighted']:.6f} |"
        )
    lines.extend(["", "## Class supports and selection basis", ""])
    for task in ("binary", "category"):
        item = record["results"][task]
        lines.extend(
            [
                f"### {task.title()}",
                "",
                f"- **Test class counts:** `{item['test_class_counts']}`",
                f"- **Selection basis:** {item['selection_basis']}",
                f"- **Prediction time:** {item['prediction_time_seconds']:.3f}s",
                f"- **Confusion matrix label order:** `{item['metrics']['labels']}`",
                "",
            ]
        )
        if task == "binary":
            lines.append(
                f"- **False positive rate:** {item['metrics']['false_positive_rate']:.6f} "
                f"({item['metrics']['binary_confusion_counts']['false_positives']} false positives among "
                f"{item['test_class_counts'].get('0', 0)} Normal records)"
            )
            lines.append("")
    lines.extend(
        [
            "## Interpretation notes",
            "",
            "The test split contains very few Normal and Theft records. Their precision, recall, and F1 estimates can change sharply with just one or two errors. Read the per-class precision/recall/F1 and support values in the JSON alongside aggregate metrics. These test results estimate performance for the chosen validation-selected models; they do not replace evaluation on other networks or deployments.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not TEST_V2_CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"Test split is missing: {TEST_V2_CLEAN_PARQUET}")
    test = pd.read_parquet(TEST_V2_CLEAN_PARQUET)
    missing = set(CANONICAL_22_FEATURES + ["attack", "category"]) - set(test.columns)
    if missing:
        raise ValueError(f"Test split is missing required columns: {sorted(missing)}")

    record = {
        "evaluation": "final held-out test evaluation",
        "test_dataset": TEST_V2_CLEAN_PARQUET.name,
        "test_rows_total": int(len(test)),
        "selection_protocol": "One candidate selected per task using validation macro-F1 before test evaluation.",
        "results": {
            "binary": evaluate_task("binary", test),
            "category": evaluate_task("category", test),
        },
    }

    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPERIMENTS_REPORTS_DIR / "svm_mlp_test_evaluation.json"
    markdown_path = EXPERIMENTS_REPORTS_DIR / "svm_mlp_test_evaluation.md"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_markdown(record, markdown_path)
    for task, item in record["results"].items():
        print(
            f"{task}: {item['model']} test macro-F1="
            f"{item['metrics']['f1_macro']:.6f}; accuracy={item['metrics']['accuracy']:.6f}"
        )
    print(f"Saved: {markdown_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
