"""Train reproducible SVM and MLP binary/category BoT-IoT experiments.

The full training split contains millions of rows, so experiments use a
deterministic per-class cap. Validation remains the complete, untouched split.
"""

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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    CANONICAL_22_FEATURES,
    EXPERIMENTS_REPORTS_DIR,
    RANDOM_SEED,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
)
from src.models.model_preprocessor import build_feature_preprocessor  # noqa: E402

MODELS_DIR = CONFIG_PROJECT_ROOT / "models"
TARGETS = {
    "binary": {"target": "attack", "per_class_cap": 10000, "malicious_only": False},
    "category": {"target": "category", "per_class_cap": 10000, "malicious_only": True},
}
CATEGORY_ORDER = ["DDoS", "DoS", "Reconnaissance", "Theft"]


def load_task_frames(task: str):
    config = TARGETS[task]
    target = config["target"]
    train = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET)
    val = pd.read_parquet(VAL_V2_CLEAN_PARQUET)
    if config["malicious_only"]:
        train = train[train["attack"] == 1]
        val = val[val["attack"] == 1]
    return train, val, target


def cap_training_rows(frame: pd.DataFrame, target: str, cap: int) -> pd.DataFrame:
    """Keep up to cap rows per class, preserving every available rare example."""
    pieces = [
        group.sample(n=min(cap, len(group)), random_state=RANDOM_SEED)
        for _, group in frame.groupby(target, sort=True, observed=True)
    ]
    result = pd.concat(pieces).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return result


def metrics_for(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "labels": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in labels],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    if labels == [0, 1]:
        tn, fp, fn, tp = cm.ravel()
        result["false_positive_rate"] = float(fp / (fp + tn)) if fp + tn else 0.0
    return result


def make_estimators(model_name: str):
    if model_name == "svm":
        # LinearSVC scales to the high row count and one-hot expanded schema.
        # Class weighting counters the extreme binary imbalance.
        return LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=3000)
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=RANDOM_SEED,
        verbose=False,
    )


def run_one(model_name: str, task: str):
    train, val, target = load_task_frames(task)
    sampled = cap_training_rows(train, target, TARGETS[task]["per_class_cap"])
    features = sampled[CANONICAL_22_FEATURES]
    y_train = sampled[target]
    X_val = val[CANONICAL_22_FEATURES]
    y_val = val[target]
    labels = [0, 1] if task == "binary" else CATEGORY_ORDER

    pipeline = Pipeline(
        [
            ("preprocessor", build_feature_preprocessor()),
            ("classifier", make_estimators(model_name)),
        ]
    )
    started = time.perf_counter()
    pipeline.fit(features, y_train)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = pipeline.predict(X_val)
    evaluation_seconds = time.perf_counter() - started
    metrics = metrics_for(y_val, predictions, labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}_{task}_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    record = {
        "model": "LinearSVC" if model_name == "svm" else "MLPClassifier",
        "task": task,
        "target": target,
        "training_dataset": TRAIN_V2_CLEAN_PARQUET.name,
        "validation_dataset": VAL_V2_CLEAN_PARQUET.name,
        "features": CANONICAL_22_FEATURES,
        "feature_count": len(CANONICAL_22_FEATURES),
        "seed": RANDOM_SEED,
        "sampling": {
            "method": "deterministic random sampling capped per class",
            "maximum_samples_per_class": TARGETS[task]["per_class_cap"],
            "training_rows_before_sampling": int(len(train)),
            "training_rows_used": int(len(sampled)),
            "training_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "validation_rows": int(len(val)),
            "validation_class_counts": {str(k): int(v) for k, v in y_val.value_counts().items()},
        },
        "preprocessing": "OneHotEncoder and StandardScaler fitted inside the pipeline on sampled training data only",
        "hyperparameters": pipeline.named_steps["classifier"].get_params(),
        "training_time_seconds": round(train_seconds, 3),
        "evaluation_time_seconds": round(evaluation_seconds, 3),
        "model_path": str(model_path.relative_to(CONFIG_PROJECT_ROOT)),
        "metrics": metrics,
    }
    base = EXPERIMENTS_REPORTS_DIR / f"{model_name}_{task}_experiment"
    with base.with_suffix(".json").open("w", encoding="utf-8") as out:
        json.dump(record, out, indent=2, default=str)
    write_markdown(record, base.with_suffix(".md"))
    print(f"Completed {model_name.upper()} {task}: macro-F1={metrics['f1_macro']:.6f}; reports at {base}")
    return record


def write_markdown(record, path: Path):
    metrics = record["metrics"]
    lines = [
        f"# {record['model']} {record['task'].title()} Experiment",
        "",
        f"- **Target:** `{record['target']}`",
        f"- **Features:** {record['feature_count']} canonical features",
        f"- **Training rows used:** {record['sampling']['training_rows_used']:,} of {record['sampling']['training_rows_before_sampling']:,}",
        f"- **Per-class cap:** {record['sampling']['maximum_samples_per_class']:,}",
        f"- **Training class counts:** `{record['sampling']['training_class_counts']}`",
        f"- **Validation rows:** {record['sampling']['validation_rows']:,} (full validation split)",
        f"- **Training time:** {record['training_time_seconds']:.3f}s",
        f"- **Model artifact:** `{record['model_path']}`",
        "",
        "## Validation metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| Macro precision | {metrics['precision_macro']:.6f} |",
        f"| Macro recall | {metrics['recall_macro']:.6f} |",
        f"| Macro F1 | {metrics['f1_macro']:.6f} |",
        f"| Weighted F1 | {metrics['f1_weighted']:.6f} |",
    ]
    if "false_positive_rate" in metrics:
        lines.append(f"| False positive rate | {metrics['false_positive_rate']:.6f} |")
    lines.extend([
        "",
        f"Confusion matrix label order: `{metrics['labels']}`.",
        "",
        "## Evaluation notes",
        "",
        "The validation set was not sampled or resampled. Training used a fixed-seed per-class cap to make these algorithms practical on the multi-million-row dataset. Compare per-class results and support, especially for Normal and Theft, rather than relying on accuracy alone. The test split remains reserved for final evaluation after model selection.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    for model in ("svm", "mlp"):
        for task in ("binary", "category"):
            run_one(model, task)


if __name__ == "__main__":
    main()
