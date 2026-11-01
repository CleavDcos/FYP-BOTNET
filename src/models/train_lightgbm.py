"""Train and validate a LightGBM category classifier on the v2 data."""

import argparse
import json

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config.settings import (
    CANONICAL_22_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_V2_ROS_CAT_PARQUET,
    VAL_V2_CLEAN_PARQUET,
)

CATEGORICAL_FEATURES = ["proto", "flgs", "state"]
NUMERIC_FEATURES = [
    name for name in CANONICAL_22_FEATURES if name not in CATEGORICAL_FEATURES
]


def build_pipeline(
    n_estimators: int, learning_rate: float, num_leaves: int
) -> Pipeline:
    """Build sparse feature preprocessing and a multiclass LightGBM model."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        # Keep the combined representation sparse for the large training set.
        sparse_threshold=1.0,
    )

    classifier = LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LightGBM model to predict Normal or an attack category, "
            "then report metrics on the untouched validation split."
        )
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()
    if args.n_estimators < 1 or args.learning_rate <= 0 or args.num_leaves < 2:
        parser.error("n-estimators and learning-rate must be positive; num-leaves must be at least 2")

    required_columns = CANONICAL_22_FEATURES + ["category"]
    print(f"Loading training data: {TRAIN_V2_ROS_CAT_PARQUET}", flush=True)
    train = pd.read_parquet(TRAIN_V2_ROS_CAT_PARQUET, columns=required_columns)
    print(f"Loading validation data: {VAL_V2_CLEAN_PARQUET}", flush=True)
    validation = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=required_columns)

    X_train = train[CANONICAL_22_FEATURES]
    y_train = train["category"]
    X_validation = validation[CANONICAL_22_FEATURES]
    y_validation = validation["category"]

    model = build_pipeline(args.n_estimators, args.learning_rate, args.num_leaves)
    print(f"Fitting LightGBM on {len(train):,} resampled training rows...", flush=True)
    model.fit(X_train, y_train)

    print("Scoring validation split...", flush=True)
    predictions = model.predict(X_validation)
    labels = model.named_steps["classifier"].classes_.tolist()
    metrics = {
        "model": "LightGBM",
        "target": "category",
        "training_data": str(TRAIN_V2_ROS_CAT_PARQUET),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "features": CANONICAL_22_FEATURES,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "class_weight": "balanced",
        "accuracy": accuracy_score(y_validation, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_validation, predictions),
        "macro_f1": f1_score(y_validation, predictions, average="macro"),
        "weighted_f1": f1_score(y_validation, predictions, average="weighted"),
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(
            y_validation, predictions, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_validation,
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
    model_path = model_dir / "lightgbm_category.joblib"
    report_path = report_dir / "lightgbm_category_metrics.json"
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Validation accuracy:          {metrics['accuracy']:.6f}")
    print(f"Validation balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"Validation macro F1:          {metrics['macro_f1']:.6f}")
    print(f"Saved model: {model_path}")
    print(f"Saved validation metrics: {report_path}")
    print("The test split was not loaded or used.")


if __name__ == "__main__":
    main()
