"""Train and validate an Isolation Forest for normal-vs-attack detection."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from config.settings import (
    CANONICAL_22_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
)

CATEGORICAL_FEATURES = ["proto", "flgs", "state"]
NUMERIC_FEATURES = [
    feature for feature in CANONICAL_22_FEATURES if feature not in CATEGORICAL_FEATURES
]


def build_pipeline(n_estimators: int, contamination: float) -> Pipeline:
    """Build feature preprocessing and a detector trained on normal examples."""
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
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    detector = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("detector", detector)])


def ground_truth(category: pd.Series) -> np.ndarray:
    """Map Normal to 0 and every attack category to 1."""
    return category.ne("Normal").astype(np.int8).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Isolation Forest on normal training traffic; evaluate the "
            "Normal=0 / Attack=1 mapping on validation data."
        )
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomalous fraction within normal training data (default: 0.05).",
    )
    args = parser.parse_args()
    if args.n_estimators < 1:
        parser.error("n-estimators must be positive")
    if not 0.0 < args.contamination <= 0.5:
        parser.error("contamination must be greater than 0 and at most 0.5")

    columns = CANONICAL_22_FEATURES + ["category"]
    print(f"Loading original training split: {TRAIN_V2_CLEAN_PARQUET}", flush=True)
    train = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET, columns=columns)
    normal_train = train.loc[train["category"].eq("Normal"), CANONICAL_22_FEATURES]
    if normal_train.empty:
        parser.error("No Normal records found in the clean training split")

    print(f"Loading untouched validation split: {VAL_V2_CLEAN_PARQUET}", flush=True)
    validation = pd.read_parquet(VAL_V2_CLEAN_PARQUET, columns=columns)
    X_validation = validation[CANONICAL_22_FEATURES]
    y_validation = ground_truth(validation["category"])

    model = build_pipeline(args.n_estimators, args.contamination)
    print(
        f"Fitting Isolation Forest on {len(normal_train):,} Normal training records...",
        flush=True,
    )
    model.fit(normal_train)

    print("Scoring validation split...", flush=True)
    raw_predictions = model.predict(X_validation)
    y_predictions = (raw_predictions == -1).astype(np.int8)  # anomaly -> Attack (1)
    anomaly_scores = -model.decision_function(X_validation)
    report = classification_report(
        y_validation,
        y_predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "Isolation Forest",
        "target": "binary_normal_vs_attack",
        "label_mapping": {"Normal": 0, "Attack": 1},
        "training_data": str(TRAIN_V2_CLEAN_PARQUET),
        "training_strategy": "fit on Normal records only; attack records excluded from fit",
        "normal_training_rows": int(len(normal_train)),
        "validation_rows": int(len(validation)),
        "features": CANONICAL_22_FEATURES,
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "accuracy": float(accuracy_score(y_validation, y_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_validation, y_predictions)),
        "attack_precision": float(precision_score(y_validation, y_predictions, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(y_validation, y_predictions, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(y_validation, y_predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_validation, y_predictions, average="macro", zero_division=0)),
        "roc_auc_from_anomaly_score": float(roc_auc_score(y_validation, anomaly_scores)),
        "average_precision_from_anomaly_score": float(
            average_precision_score(y_validation, anomaly_scores)
        ),
        "confusion_matrix_labels_in_order": ["Normal (0)", "Attack (1)"],
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(
            y_validation, y_predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
    }

    model_dir = PROJECT_ROOT / "models"
    report_dir = PROJECT_ROOT / "reports" / "experiments"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "isolation_forest_binary.joblib"
    report_path = report_dir / "isolation_forest_binary_metrics.json"
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Validation accuracy:          {metrics['accuracy']:.6f}")
    print(f"Validation balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"Attack precision:             {metrics['attack_precision']:.6f}")
    print(f"Attack recall:                {metrics['attack_recall']:.6f}")
    print(f"Attack F1:                    {metrics['attack_f1']:.6f}")
    print(f"Saved model: {model_path}")
    print(f"Saved validation metrics: {report_path}")
    print("The test split was not loaded or used.")


if __name__ == "__main__":
    main()
