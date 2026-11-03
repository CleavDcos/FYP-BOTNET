"""Train a normal-only reconstruction autoencoder for binary anomaly detection.

Every Normal record from the clean training split is used. The test split is
not loaded. Validation chooses the reconstruction-error threshold and is used
for reporting, so validation scores are model-selection metrics.
"""

import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
MAX_EPOCHS = 300
PATIENCE = 20
NOISE_SCALE = 0.1


def fit_preprocessing(normal_train):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )
    encoder.fit(normal_train[CATEGORICAL])
    scaler = StandardScaler()
    scaler.fit(normal_train[NUMERICAL].to_numpy(dtype=np.float64))
    return encoder, scaler


def transform(frame, encoder, scaler):
    categorical = encoder.transform(frame[CATEGORICAL])
    numerical = scaler.transform(frame[NUMERICAL].to_numpy(dtype=np.float64))
    return np.concatenate([categorical, numerical], axis=1).astype(np.float32)


def reconstruction_errors(model, values):
    reconstruction = model.predict(values)
    return np.mean(np.square(values - reconstruction), axis=1)


def choose_thresholds(y_true, anomaly_scores):
    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        y_true, anomaly_scores, pos_label=1
    )
    balanced_scores = (true_positive_rates + (1.0 - false_positive_rates)) / 2.0
    normal_count = int(np.sum(np.asarray(y_true) == 0))
    attack_count = int(np.sum(np.asarray(y_true) == 1))
    true_positive = true_positive_rates * attack_count
    false_positive = false_positive_rates * normal_count
    true_negative = normal_count - false_positive
    false_negative = attack_count - true_positive
    attack_f1 = (2.0 * true_positive) / (
        2.0 * true_positive + false_positive + false_negative + 1e-12
    )
    normal_f1 = (2.0 * true_negative) / (
        2.0 * true_negative + false_positive + false_negative + 1e-12
    )
    macro_f1_scores = (attack_f1 + normal_f1) / 2.0
    macro_index = int(np.nanargmax(macro_f1_scores))
    balanced_index = int(np.nanargmax(balanced_scores))
    return {
        "macro_f1": {
            "threshold": float(thresholds[macro_index]),
            "validation_macro_f1": float(macro_f1_scores[macro_index]),
            "validation_balanced_accuracy": float(balanced_scores[macro_index]),
        },
        "balanced_accuracy": {
            "threshold": float(thresholds[balanced_index]),
            "validation_balanced_accuracy": float(balanced_scores[balanced_index]),
            "validation_macro_f1": float(macro_f1_scores[balanced_index]),
        },
    }


def compute_metrics(y_true, y_pred, anomaly_scores):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "normal_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "attack_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc_from_reconstruction_error": float(roc_auc_score(y_true, anomaly_scores)),
        "average_precision_attack": float(average_precision_score(y_true, anomaly_scores)),
        "confusion_matrix_labels_in_order": ["Normal (0)", "Attack (1)"],
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "confusion_counts": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
    }


def fit_autoencoder(X_normal_train, X_normal_validation):
    model = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation="relu",
        solver="adam",
        alpha=5e-2,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=1,
        shuffle=True,
        tol=0.0,
        warm_start=True,
        random_state=RANDOM_SEED,
    )
    best_model = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)

    for epoch in range(1, MAX_EPOCHS + 1):
        noisy_inputs = X_normal_train + rng.normal(
            loc=0.0,
            scale=NOISE_SCALE,
            size=X_normal_train.shape,
        ).astype(np.float32)
        model.partial_fit(noisy_inputs, X_normal_train)
        validation_errors = reconstruction_errors(model, X_normal_validation)
        validation_normal_loss = float(np.mean(validation_errors))
        if validation_normal_loss < best_loss - 1e-7:
            best_loss = validation_normal_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
        if epoch % 10 == 0:
            print(
                f"Autoencoder epoch {epoch}: validation Normal reconstruction MSE="
                f"{validation_normal_loss:.6f}",
                flush=True,
            )

    fit_seconds = time.perf_counter() - started
    return best_model, best_epoch, epoch, best_loss, fit_seconds


def run_experiment():
    columns = CANONICAL_22_FEATURES + ["attack"]
    print(f"Loading training split: {TRAIN_V2_CLEAN_PARQUET}", flush=True)
    train = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET, columns=columns)
    normal_train = train.loc[train["attack"] == 0, CANONICAL_22_FEATURES].copy()
    if len(normal_train) < 20:
        raise ValueError(f"Too few Normal training rows for an autoencoder: {len(normal_train)}")

    print(f"Loading validation split: {VAL_V2_CLEAN_PARQUET}", flush=True)
    validation = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=columns)
    y_validation = validation["attack"].astype(np.int8).to_numpy()
    normal_validation = validation.loc[
        validation["attack"] == 0, CANONICAL_22_FEATURES
    ].copy()

    encoder, scaler = fit_preprocessing(normal_train)
    X_normal_train = transform(normal_train, encoder, scaler)
    X_normal_validation = transform(normal_validation, encoder, scaler)
    X_validation = transform(validation, encoder, scaler)

    print(
        f"Fitting bottleneck autoencoder on all {len(X_normal_train):,} Normal training rows...",
        flush=True,
    )
    model, best_epoch, epochs_completed, normal_val_loss, fit_seconds = fit_autoencoder(
        X_normal_train,
        X_normal_validation,
    )
    anomaly_scores = reconstruction_errors(model, X_validation)
    thresholds = choose_thresholds(
        y_validation, anomaly_scores
    )
    threshold = thresholds["macro_f1"]["threshold"]
    predictions = (anomaly_scores >= threshold).astype(np.int8)
    metrics = compute_metrics(y_validation, predictions, anomaly_scores)

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "autoencoder_binary_full_data.joblib"
    bundle = {
        "encoder": encoder,
        "scaler": scaler,
        "autoencoder": model,
        "features": CANONICAL_22_FEATURES,
        "categorical_features": CATEGORICAL,
        "numerical_features": NUMERICAL,
        "threshold": threshold,
        "normal_label": 0,
        "attack_label": 1,
    }
    joblib.dump(bundle, model_path)

    record = {
        "model": "Normal-only bottleneck autoencoder",
        "task": "binary_anomaly_detection",
        "target": "attack",
        "training_dataset": TRAIN_V2_CLEAN_PARQUET.name,
        "validation_dataset": VAL_V2_CLEAN_PARQUET.name,
        "features": CANONICAL_22_FEATURES,
        "feature_count": len(CANONICAL_22_FEATURES),
        "seed": RANDOM_SEED,
        "training_strategy": (
            "Fit on every Normal (attack=0) row in the training split. Attacks are not used to fit the reconstruction model."
        ),
        "normal_training_rows": int(len(normal_train)),
        "validation_rows": int(len(validation)),
        "validation_class_counts": {
            str(k): int(v) for k, v in validation["attack"].value_counts().items()
        },
        "preprocessing": (
            "One-hot categorical encoding and numeric standardization fitted on Normal training rows only."
        ),
        "architecture": {
            "input_features_after_encoding": int(X_normal_train.shape[1]),
            "hidden_layers": [16, 8, 16],
            "bottleneck_width": 8,
            "activation": "relu",
            "optimizer": "adam",
            "loss": "mean squared reconstruction error",
            "l2_alpha": 5e-2,
            "denoising_input_noise_standard_deviation": NOISE_SCALE,
        },
        "overfitting_controls": {
            "early_stopping_signal": "validation Normal reconstruction MSE",
            "best_epoch": best_epoch,
            "epochs_completed": epochs_completed,
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "maximum_epoch_limit_reached": epochs_completed == MAX_EPOCHS,
            "best_normal_validation_reconstruction_mse": normal_val_loss,
        },
        "threshold_selection": {
            "method": "Choose threshold that maximizes macro-F1 on the full validation split.",
            "threshold": threshold,
            "validation_macro_f1_at_threshold": thresholds["macro_f1"]["validation_macro_f1"],
            "validation_balanced_accuracy_at_threshold": thresholds["macro_f1"]["validation_balanced_accuracy"],
            "alternative_balanced_accuracy_threshold": thresholds["balanced_accuracy"],
        },
        "training_time_seconds": round(fit_seconds, 3),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "test_split_used": False,
        "metrics": metrics,
    }
    json_path = EXPERIMENTS_REPORTS_DIR / "autoencoder_binary_full_data_experiment.json"
    markdown_path = EXPERIMENTS_REPORTS_DIR / "autoencoder_binary_full_data_experiment.md"
    json_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    write_markdown(record, markdown_path)
    print(
        f"Autoencoder binary validation: macro-F1={metrics['macro_f1']:.6f}; "
        f"balanced accuracy={metrics['balanced_accuracy']:.6f}; threshold={threshold:.6g}",
        flush=True,
    )
    print(f"Saved model: {model_path}")
    print(f"Saved report: {markdown_path}")
    return record


def write_markdown(record, path):
    metrics = record["metrics"]
    counts = metrics["confusion_counts"]
    lines = [
        "# Normal-Only Autoencoder: Binary Botnet Anomaly Detection",
        "",
        "## Training and validation protocol",
        "",
        f"- **Training rows used:** {record['normal_training_rows']:,} Normal records (all Normal rows in the training split)",
        "- **Attacks used to fit the autoencoder:** No",
        f"- **Validation rows:** {record['validation_rows']:,} (full validation split)",
        f"- **Validation class counts:** `{record['validation_class_counts']}`",
        f"- **Features:** {record['feature_count']} canonical input features; {record['architecture']['input_features_after_encoding']} after encoding",
        f"- **Architecture:** input -> 16 -> 8 bottleneck -> 16 -> reconstruction output",
        f"- **Regularization:** L2 alpha={record['architecture']['l2_alpha']} plus denoising input noise std={record['architecture']['denoising_input_noise_standard_deviation']}; best checkpoint by Normal validation reconstruction MSE",
        f"- **Best epoch:** {record['overfitting_controls']['best_epoch']} of {record['overfitting_controls']['epochs_completed']} completed",
        f"- **Maximum epoch limit reached:** {record['overfitting_controls']['maximum_epoch_limit_reached']}",
        f"- **Validation-selected anomaly threshold:** {record['threshold_selection']['threshold']:.8g}",
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
        f"| Macro F1 | {metrics['macro_f1']:.6f} |",
        f"| Normal precision | {metrics['normal_precision']:.6f} |",
        f"| Normal recall | {metrics['normal_recall']:.6f} |",
        f"| Attack precision | {metrics['attack_precision']:.6f} |",
        f"| Attack recall | {metrics['attack_recall']:.6f} |",
        f"| False positive rate | {metrics['false_positive_rate']:.6f} |",
        f"| ROC-AUC from reconstruction error | {metrics['roc_auc_from_reconstruction_error']:.6f} |",
        "",
        f"Confusion matrix `[[TN, FP], [FN, TP]]`: `{metrics['confusion_matrix_rows_actual_columns_predicted']}`.",
        f"Counts: TN={counts['true_negatives']}, FP={counts['false_positives']}, FN={counts['false_negatives']}, TP={counts['true_positives']}.",
        "",
        "## Interpretation and limitations",
        "",
        "The autoencoder is trained only on Normal traffic. A record is flagged as an attack when its reconstruction error is at least the selected threshold. The primary threshold was selected on validation by macro-F1; an alternative threshold maximizing balanced accuracy is recorded in the JSON. The same validation split is reported, so these scores are for model comparison rather than an unbiased final estimate. The test split was not used.",
        "",
        "There are only 334 Normal training records and 72 Normal validation records. That is very little data for learning benign behavior; per-class metrics and the chosen threshold can be unstable. This autoencoder is a binary detector only and does not predict DDoS/DoS/Reconnaissance/Theft categories.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_experiment()
