"""Train and validate a decision-tree classifier on the clean v2 data."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
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
from sklearn.tree import DecisionTreeClassifier

from config.settings import (
    CANONICAL_22_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_V2_CLEAN_PARQUET,
    TRAIN_V2_ROS_CAT_PARQUET,
    VAL_V2_CLEAN_PARQUET,
)

CATEGORICAL_FEATURES = ["proto", "flgs", "state"]
NUMERIC_FEATURES = [
    name for name in CANONICAL_22_FEATURES if name not in CATEGORICAL_FEATURES
]


def build_pipeline(max_depth: int | None, min_samples_leaf: int) -> Pipeline:
    """Build preprocessing and classifier as one serializable pipeline."""
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
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a decision tree and report held-out validation metrics."
    )
    parser.add_argument(
        "--target",
        choices=("attack", "category", "subcategory"),
        default="category",
        help="Label to predict (default: category, including Normal).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum tree depth; use 0 for an unlimited-depth tree (default: 20).",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    args = parser.parse_args()

    if args.max_depth < 0 or args.min_samples_leaf < 1:
        parser.error("max-depth must be nonnegative and min-samples-leaf must be positive")

    max_depth = args.max_depth or None
    required_columns = CANONICAL_22_FEATURES + [args.target]
    # The category ROS split is training-only resampling. It gives the very
    # underrepresented Normal and Theft classes more examples; validation stays
    # on its original, untouched distribution.
    train_path = (
        TRAIN_V2_ROS_CAT_PARQUET
        if args.target == "category"
        else TRAIN_V2_CLEAN_PARQUET
    )
    print(f"Loading training data: {train_path}", flush=True)
    train = pd.read_parquet(train_path, columns=required_columns)
    print(f"Loading validation data: {VAL_V2_CLEAN_PARQUET}", flush=True)
    validation = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=required_columns)

    X_train = train[CANONICAL_22_FEATURES]
    y_train = train[args.target]
    X_validation = validation[CANONICAL_22_FEATURES]
    y_validation = validation[args.target]

    model = build_pipeline(max_depth, args.min_samples_leaf)
    print(
        f"Fitting decision tree for '{args.target}' on {len(train):,} rows...",
        flush=True,
    )
    model.fit(X_train, y_train)

    print("Scoring validation split...", flush=True)
    predictions = model.predict(X_validation)
    labels = model.named_steps["classifier"].classes_.tolist()
    metrics = {
        "target": args.target,
        "training_data": str(train_path),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "features": CANONICAL_22_FEATURES,
        "max_depth": max_depth,
        "min_samples_leaf": args.min_samples_leaf,
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
    model_path = model_dir / f"decision_tree_{args.target}.joblib"
    report_path = report_dir / f"decision_tree_{args.target}_metrics.json"
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
