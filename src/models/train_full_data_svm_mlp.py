"""Train separate full-data linear-SVM and MLP models for binary/category tasks.

All eligible training rows are seen in every epoch. Minibatches are only an
optimization strategy; they do not subsample the training dataset. The test
split is deliberately not read. Validation macro-F1 controls early stopping.
"""

import copy
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from config.settings import (
    CANONICAL_22_FEATURES,
    EXPERIMENTS_REPORTS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
)

CATEGORICAL = ["proto", "flgs", "state"]
NUMERICAL = [name for name in CANONICAL_22_FEATURES if name not in CATEGORICAL]
CATEGORY_LABELS = ["DDoS", "DoS", "Reconnaissance", "Theft"]
BATCH_SIZE = 65_536
TASKS = {
    "binary": {"target": "attack", "malicious_only": False, "labels": [0, 1]},
    "category": {
        "target": "category",
        "malicious_only": True,
        "labels": CATEGORY_LABELS,
    },
}


def fit_transformers(training_frame):
    """Fit encoders/scaler on the complete task training partition."""
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float32
    )
    encoder.fit(training_frame[CATEGORICAL])

    scaler = StandardScaler(with_mean=False)
    for start in range(0, len(training_frame), BATCH_SIZE):
        numeric = training_frame.iloc[start : start + BATCH_SIZE][NUMERICAL]
        scaler.partial_fit(numeric.to_numpy(dtype=np.float64, copy=False))
    return encoder, scaler


def transform_frame(frame, encoder, scaler):
    categorical = encoder.transform(frame[CATEGORICAL])
    numerical = scaler.transform(frame[NUMERICAL].to_numpy(dtype=np.float64, copy=False))
    numerical = csr_matrix(numerical.astype(np.float32, copy=False))
    return hstack([categorical, numerical], format="csr", dtype=np.float32)


def class_sample_weights(y, labels, counts):
    """Return balanced per-row weights without adding or removing rows."""
    total = len(y)
    n_classes = len(labels)
    weights = {label: total / (n_classes * counts[label]) for label in labels}
    return np.fromiter((weights[value] for value in y), dtype=np.float32, count=total)


def make_model(kind):
    if kind == "svm":
        return LinearSVC(
            C=0.1,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=RANDOM_SEED,
            tol=1e-2,
        )
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=16_384,
        learning_rate_init=1e-3,
        max_iter=1,
        shuffle=False,
        early_stopping=False,
        random_state=RANDOM_SEED,
        verbose=False,
    )


def predict(model, X, kind):
    return model.predict(X)


def evaluate(model, X_val, y_val, labels, kind):
    y_pred = predict(model, X_val, kind)
    matrix = confusion_matrix(y_val, y_pred, labels=labels)
    result = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy": float(
            recall_score(y_val, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_val, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_val, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_val, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_val, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "labels": [
            int(label) if isinstance(label, (int, np.integer)) else str(label)
            for label in labels
        ],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_val,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    if labels == [0, 1]:
        tn, fp, fn, tp = matrix.ravel()
        result["false_positive_rate"] = float(fp / (fp + tn)) if fp + tn else 0.0
        result["binary_counts"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    return result


def fit_mlp_with_early_stopping(model, X_train, y_train, sample_weights, X_val, y_val, labels):
    """Stream all rows each epoch and retain the validation-best MLP checkpoint."""
    rng = np.random.default_rng(RANDOM_SEED)
    max_epochs, patience = 20, 4
    best_score = -1.0
    best_epoch = 0
    best_model = None
    stale_epochs = 0
    classes_array = np.asarray(labels)
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            X_batch = X_train[indices]
            y_batch = y_train.iloc[indices].to_numpy()
            weights_batch = sample_weights[indices]
            model.partial_fit(
                X_batch,
                y_batch,
                classes=classes_array if epoch == 1 and start == 0 else None,
                sample_weight=weights_batch,
            )

        epoch_metrics = evaluate(model, X_val, y_val, labels, "mlp")
        score = epoch_metrics["f1_macro"]
        print(
            f"  MLP epoch {epoch}: validation macro-F1={score:.6f}",
            flush=True,
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - started
    return best_model, best_epoch, epoch, elapsed


def run_task(task, kind):
    task_options = TASKS[task]
    target = task_options["target"]
    cols = CANONICAL_22_FEATURES + ["attack", "category"]
    print(f"Loading full training data for {kind}/{task}...", flush=True)
    train = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET, columns=cols)
    if task_options["malicious_only"]:
        train = train.loc[train["attack"] == 1].copy()
    y_train = train[target].reset_index(drop=True)
    X_train_df = train[CANONICAL_22_FEATURES].reset_index(drop=True)
    train_counts = {label: int((y_train == label).sum()) for label in task_options["labels"]}

    print(f"Loading full validation data for {kind}/{task}...", flush=True)
    validation = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=cols)
    if task_options["malicious_only"]:
        validation = validation.loc[validation["attack"] == 1].copy()
    X_val_df = validation[CANONICAL_22_FEATURES]
    y_val = validation[target].reset_index(drop=True)
    val_counts = {label: int((y_val == label).sum()) for label in task_options["labels"]}

    print("Fitting full-data encoders/scaler...", flush=True)
    encoder, scaler = fit_transformers(X_train_df)
    X_train = transform_frame(X_train_df, encoder, scaler)
    X_val = transform_frame(X_val_df, encoder, scaler)
    sample_weights = class_sample_weights(y_train.to_numpy(), task_options["labels"], train_counts)

    model = make_model(kind)
    print(f"Training {kind.upper()} on all {len(y_train):,} rows...", flush=True)
    if kind == "svm":
        fit_started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started
        best_model = model
        best_epoch = None
        epochs_completed = None
    else:
        best_model, best_epoch, epochs_completed, fit_seconds = fit_mlp_with_early_stopping(
            model,
            X_train,
            y_train,
            sample_weights,
            X_val,
            y_val,
            task_options["labels"],
        )
        if best_model is None:
            raise RuntimeError(f"No model checkpoint was produced for {kind}/{task}")

    started = time.perf_counter()
    metrics = evaluate(best_model, X_val, y_val, task_options["labels"], kind)
    prediction_seconds = time.perf_counter() - started
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{kind}_{task}_full_data.joblib"
    bundle = {
        "encoder": encoder,
        "scaler": scaler,
        "classifier": best_model,
        "features": CANONICAL_22_FEATURES,
        "categorical_features": CATEGORICAL,
        "numerical_features": NUMERICAL,
        "task": task,
        "model_kind": kind,
    }
    joblib.dump(bundle, model_path)

    record = {
        "model": "LinearSVC" if kind == "svm" else "MLPClassifier",
        "task": task,
        "target": target,
        "training_dataset": TRAIN_V2_CLEAN_PARQUET.name,
        "validation_dataset": VAL_V2_CLEAN_PARQUET.name,
        "features": CANONICAL_22_FEATURES,
        "feature_count": len(CANONICAL_22_FEATURES),
        "seed": RANDOM_SEED,
        "training_rows": int(len(y_train)),
        "training_class_counts": {str(k): v for k, v in train_counts.items()},
        "sampling_or_resampling": "None. Every eligible training row is visited once in every completed epoch.",
        "optimization_batch_size": BATCH_SIZE if kind == "mlp" else None,
        "validation_rows": int(len(y_val)),
        "validation_class_counts": {str(k): v for k, v in val_counts.items()},
        "class_weighting": (
            "LinearSVC class_weight='balanced'."
            if kind == "svm"
            else "Balanced per-row sample weights computed from the complete task training split."
        ),
        "early_stopping": (
            {
                "used": False,
                "regularization": "L2 with C=0.1; solver stops by tolerance or max_iter.",
                "iterations_used": int(best_model.n_iter_),
                "converged_within_iteration_limit": int(best_model.n_iter_) < best_model.get_params()["max_iter"],
            }
            if kind == "svm"
            else {
                "used": True,
                "monitored_metric": "validation macro-F1",
                "best_epoch": best_epoch,
                "epochs_completed": epochs_completed,
                "maximum_epochs": 20,
                "patience": 4,
            }
        ),
        "hyperparameters": best_model.get_params(),
        "training_time_seconds": round(fit_seconds, 3),
        "validation_prediction_time_seconds": round(prediction_seconds, 3),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "test_split_used": False,
        "metrics": metrics,
    }
    base = EXPERIMENTS_REPORTS_DIR / f"{kind}_{task}_full_data_experiment"
    base.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    write_markdown(record, base.with_suffix(".md"))
    print(
        f"{kind.upper()} {task}: macro-F1={metrics['f1_macro']:.6f}, "
        f"best epoch={best_epoch}/{epochs_completed}",
        flush=True,
    )
    return record


def write_markdown(record, path):
    metrics = record["metrics"]
    if record["model"] == "LinearSVC":
        training_details = [
            f"- **Training rows:** {record['training_rows']:,} (all eligible rows; no sampling or resampling)",
            f"- **Training class counts:** `{record['training_class_counts']}`",
            f"- **Validation rows:** {record['validation_rows']:,} (full validation task split)",
            f"- **Validation class counts:** `{record['validation_class_counts']}`",
            "- **Optimization:** LinearSVC fit on the complete sparse training matrix.",
            f"- **Class weighting:** {record['class_weighting']}",
            f"- **Regularization / solver:** {record['early_stopping']['regularization']} "
            f"Iterations used: {record['early_stopping']['iterations_used']}; "
            f"converged: {record['early_stopping']['converged_within_iteration_limit']}",
        ]
        overfit_note = (
            "L2 regularization (C=0.1) limits model complexity. The solver convergence flag above "
            "matters: the binary fit reached its iteration limit, so treat it as a provisional "
            "candidate. This experiment did not use test data."
        )
    else:
        training_details = [
            f"- **Training rows:** {record['training_rows']:,} (all eligible rows; no sampling or resampling)",
            f"- **Training class counts:** `{record['training_class_counts']}`",
            f"- **Rows processed per epoch:** {record['training_rows']:,}",
            f"- **Optimization batch size:** {record['optimization_batch_size']:,}; batches cover the full training set each epoch",
            f"- **Validation rows:** {record['validation_rows']:,} (full validation task split)",
            f"- **Validation class counts:** `{record['validation_class_counts']}`",
            f"- **Class weighting:** {record['class_weighting']}",
            f"- **Early stopping:** monitor validation macro-F1; best epoch {record['early_stopping']['best_epoch']} of {record['early_stopping']['epochs_completed']} completed",
        ]
        overfit_note = (
            "L2 regularization is used; the best validation macro-F1 checkpoint is retained and "
            "training stops after the configured patience or at the maximum epoch limit. Balanced "
            "sample weights change each row's loss contribution but do not change which rows are "
            "used. This experiment did not use test data."
        )
    lines = [
        f"# {record['model']}: {record['task'].title()}",
        "",
        "## Training and validation protocol",
        "",
        *training_details,
        f"- **Training time:** {record['training_time_seconds']:.2f}s",
        f"- **Model artifact:** `{record['model_path']}`",
        "- **Test split used:** No",
        "",
        "## Validation metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.6f} |",
        f"| Macro precision | {metrics['precision_macro']:.6f} |",
        f"| Macro recall | {metrics['recall_macro']:.6f} |",
        f"| Macro F1 | {metrics['f1_macro']:.6f} |",
        f"| Weighted F1 | {metrics['f1_weighted']:.6f} |",
    ]
    if "false_positive_rate" in metrics:
        lines.append(f"| False positive rate | {metrics['false_positive_rate']:.6f} |")
        counts = metrics["binary_counts"]
        lines.extend(
            [
                "",
                f"Confusion matrix `[[TN, FP], [FN, TP]]`: `{metrics['confusion_matrix']}`",
                f"Counts: TN={counts['true_negatives']}, FP={counts['false_positives']}, "
                f"FN={counts['false_negatives']}, TP={counts['true_positives']}.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"Confusion matrix label order: `{metrics['labels']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Overfitting and class-balance notes",
            "",
            overfit_note,
            (
                "The validation set contains only 72 Normal samples; Normal-class metrics are sensitive to a few records."
                if record["task"] == "binary"
                else "The validation set contains only 12 Theft samples; Theft-class metrics are highly uncertain."
            ),
            "Validation metrics are for model comparison; the test set was not used for these full-data runs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_report():
    names = {
        ("svm", "binary"): "svm_binary_full_data_experiment.json",
        ("mlp", "binary"): "mlp_binary_full_data_experiment.json",
        ("svm", "category"): "svm_category_full_data_experiment.json",
        ("mlp", "category"): "mlp_category_full_data_experiment.json",
    }
    records = {}
    for key, filename in names.items():
        report_path = EXPERIMENTS_REPORTS_DIR / filename
        if not report_path.exists():
            return
        records[key] = json.loads(report_path.read_text(encoding="utf-8"))
    lines = [
        "# Full-Data SVM and MLP Validation Comparison",
        "",
        "Each model is trained independently. All eligible rows from the clean training split are used; there is no sampling or resampling. Results are measured on the complete validation split.",
        "",
        "| Task | Model | Training rows | Validation macro-F1 | Accuracy | Training time |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("binary", "category"):
        for kind in ("svm", "mlp"):
            record = records[(kind, task)]
            metric = record["metrics"]
            lines.append(
                f"| {task.title()} | {record['model']} | {record['training_rows']:,} | "
                f"{metric['f1_macro']:.6f} | {metric['accuracy']:.6f} | "
                f"{record['training_time_seconds']:.2f}s |"
            )
    lines.extend(
        [
            "",
            "## Method notes",
            "",
            "- SVM and MLP are separate candidates; neither feeds into or combines with the other.",
            "- Class weights address imbalance without changing the training rows.",
            "- MLP retains its best validation macro-F1 checkpoint. LinearSVC uses L2 regularization; its convergence status is reported in its task reports.",
            "- The held-out test split was not used for these full-data runs. Earlier test results in the repository refer to the older capped-sample model artifacts and do not describe these retrained models.",
            "- Normal has 72 validation rows in binary detection; Theft has 12 in category classification. Interpret those class metrics cautiously.",
            "",
        ]
    )
    (EXPERIMENTS_REPORTS_DIR / "svm_mlp_full_data_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "category", "all"], default="all")
    parser.add_argument("--model", choices=["svm", "mlp", "all"], default="all")
    args = parser.parse_args(argv)
    tasks = TASKS if args.task == "all" else [args.task]
    models = ("svm", "mlp") if args.model == "all" else (args.model,)
    for task in tasks:
        for kind in models:
            run_task(task, kind)
    write_comparison_report()


if __name__ == "__main__":
    main()
