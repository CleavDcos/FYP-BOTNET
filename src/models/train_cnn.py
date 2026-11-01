"""Train and validate a 1D CNN for Normal / attack-category prediction."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from config.settings import (
    CANONICAL_22_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_V2_ROS_CAT_PARQUET,
    VAL_V2_CLEAN_PARQUET,
)
from src.models.cnn_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_cnn,
    fit_transform_features,
    make_preprocessors,
    transform_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 1D CNN to predict Normal or an attack category."
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must all be positive")

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    required_columns = CANONICAL_22_FEATURES + ["category"]
    print(f"Loading resampled training data: {TRAIN_V2_ROS_CAT_PARQUET}", flush=True)
    train = pd.read_parquet(TRAIN_V2_ROS_CAT_PARQUET, columns=required_columns)
    print(f"Loading untouched validation data: {VAL_V2_CLEAN_PARQUET}", flush=True)
    validation = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=required_columns)

    numeric_transformer, categorical_encoder, label_encoder = make_preprocessors()
    print("Fitting feature encoders on training data only...", flush=True)
    X_train = fit_transform_features(
        train[CANONICAL_22_FEATURES], numeric_transformer, categorical_encoder
    )
    X_validation = transform_features(
        validation[CANONICAL_22_FEATURES], numeric_transformer, categorical_encoder
    )
    y_train = label_encoder.fit_transform(train["category"])
    y_validation = label_encoder.transform(validation["category"])

    cardinalities = [len(categories) for categories in categorical_encoder.categories_]
    model = build_cnn(cardinalities, len(label_encoder.classes_))
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}

    print(
        f"Training CNN on {len(train):,} rows; numeric feature order: {NUMERIC_FEATURES}",
        flush=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    probabilities = model.predict(X_validation, batch_size=args.batch_size, verbose=0)
    predictions = label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
    y_validation_labels = label_encoder.inverse_transform(y_validation)
    labels = label_encoder.classes_.tolist()
    metrics = {
        "model": "1D CNN",
        "target": "category",
        "training_data": str(TRAIN_V2_ROS_CAT_PARQUET),
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "numeric_feature_sequence_order": NUMERIC_FEATURES,
        "categorical_features_embedded_separately": CATEGORICAL_FEATURES,
        "labels": [str(label) for label in labels],
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "batch_size": args.batch_size,
        "accuracy": float(accuracy_score(y_validation_labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_validation_labels, predictions)
        ),
        "macro_f1": float(f1_score(y_validation_labels, predictions, average="macro")),
        "weighted_f1": float(
            f1_score(y_validation_labels, predictions, average="weighted")
        ),
        "confusion_matrix": confusion_matrix(
            y_validation_labels, predictions, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_validation_labels,
            predictions,
            labels=labels,
            target_names=[str(label) for label in labels],
            output_dict=True,
            zero_division=0,
        ),
    }

    model_dir = PROJECT_ROOT / "models"
    report_dir = PROJECT_ROOT / "reports" / "experiments"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cnn_category.keras"
    preprocessor_path = model_dir / "cnn_category_preprocessors.joblib"
    report_path = report_dir / "cnn_category_metrics.json"

    model.save(model_path)
    joblib.dump(
        {
            "numeric_transformer": numeric_transformer,
            "categorical_encoder": categorical_encoder,
            "label_encoder": label_encoder,
        },
        preprocessor_path,
    )
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Validation accuracy:          {metrics['accuracy']:.6f}")
    print(f"Validation balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"Validation macro F1:          {metrics['macro_f1']:.6f}")
    print(f"Saved model: {model_path}")
    print(f"Saved preprocessors: {preprocessor_path}")
    print(f"Saved validation metrics: {report_path}")
    print("The test split was not loaded or used.")


if __name__ == "__main__":
    main()
