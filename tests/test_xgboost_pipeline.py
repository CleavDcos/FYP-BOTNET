"""
Unit Tests for XGBoost Models (Binary and Multiclass Category Classification)
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import (
    CANONICAL_22_FEATURES,
    TARGET_LABELS,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
    PROJECT_ROOT,
)
from src.preprocessing.model_preprocessor import build_feature_preprocessor
from src.models.xgboost_category_sampling import (
    EXPECTED_ATTACK_CATEGORIES,
    MIN_SAMPLES_RELIABLE_EVAL,
    load_validation_data,
    load_training_variant,
    evaluate_category_predictions,
)
from src.models.xgboost_binary import evaluate_binary_model


class TestXGBoostPipeline(unittest.TestCase):
    """Test suite verifying XGBoost data pipelines, label encoding, and metrics computation."""

    def test_binary_label_and_feature_isolation(self):
        """Verify binary dataset features are isolated and labels are [0, 1]."""
        df_train_sample = pd.read_parquet(TRAIN_V2_CLEAN_PARQUET, columns=CANONICAL_22_FEATURES + ["attack"]).head(200)
        X = df_train_sample[CANONICAL_22_FEATURES]
        y = df_train_sample["attack"]

        self.assertEqual(list(X.columns), CANONICAL_22_FEATURES)
        for target in TARGET_LABELS:
            self.assertNotIn(target, X.columns)

        unique_labels = set(y.unique())
        self.assertTrue(unique_labels.issubset({0, 1}))

    def test_category_label_encoding_and_inversion(self):
        """Verify label encoder maps the 4 expected categories to integers and back losslessly."""
        le = LabelEncoder()
        le.fit(EXPECTED_ATTACK_CATEGORIES)

        self.assertEqual(len(le.classes_), 4)
        self.assertEqual(set(le.classes_), {"DDoS", "DoS", "Reconnaissance", "Theft"})

        encoded = le.transform(EXPECTED_ATTACK_CATEGORIES)
        self.assertEqual(list(encoded), [0, 1, 2, 3])

        decoded = le.inverse_transform(encoded)
        self.assertEqual(list(decoded), EXPECTED_ATTACK_CATEGORIES)

    def test_category_validation_data_integrity(self):
        """Verify load_validation_data filters out Normal and isolates malicious categories."""
        X_val, y_val = load_validation_data()

        self.assertEqual(len(X_val), 550206)
        self.assertEqual(list(X_val.columns), CANONICAL_22_FEATURES)
        self.assertNotIn("Normal", y_val.unique())
        self.assertEqual(set(y_val.unique()), set(EXPECTED_ATTACK_CATEGORIES))
        self.assertEqual(int((y_val == "Theft").sum()), 12)

    def test_category_training_variants_exclusion_of_normal(self):
        """Verify all three training variants exclude 'Normal' records."""
        for variant in ["clean", "ros", "smote"]:
            X_train, y_train = load_training_variant(variant)
            self.assertNotIn("Normal", y_train.unique(), f"'Normal' found in variant: {variant}")
            self.assertEqual(set(y_train.unique()), set(EXPECTED_ATTACK_CATEGORIES))
            self.assertEqual(list(X_train.columns), CANONICAL_22_FEATURES)

    def test_sample_weight_computation(self):
        """Verify compute_sample_weight produces positive weights matching class frequencies."""
        y_synthetic = np.array([0] * 1000 + [1] * 500 + [2] * 50 + [3] * 5)
        weights = compute_sample_weight("balanced", y_synthetic)

        self.assertEqual(len(weights), len(y_synthetic))
        self.assertTrue((weights > 0).all())
        # Class 3 (5 samples) must have higher sample weight than Class 0 (1000 samples)
        self.assertGreater(weights[1550], weights[0])

    def test_evaluate_category_predictions_calculation(self):
        """Verify evaluate_category_predictions accurately flags underrepresented classes and calculates metrics."""
        y_true = pd.Series(["DDoS"] * 200 + ["DoS"] * 150 + ["Reconnaissance"] * 80 + ["Theft"] * 10)
        y_pred = np.array(["DDoS"] * 200 + ["DoS"] * 148 + ["DDoS"] * 2 + ["Reconnaissance"] * 80 + ["Theft"] * 10)

        metrics = evaluate_category_predictions(y_true, y_pred, target_names=EXPECTED_ATTACK_CATEGORIES)

        self.assertIn("overall_accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("Theft", metrics["underrepresented_classes_flags"])
        self.assertNotIn("DDoS", metrics["underrepresented_classes_flags"])
        self.assertEqual(metrics["per_class_metrics"]["Theft"]["support"], 10)
        self.assertEqual(metrics["per_class_metrics"]["Theft"]["f1_score"], 1.0)

    def test_binary_imbalance_evaluator_metrics_calculation(self):
        """Verify evaluate_binary_predictions correctly calculates FPR, PR-AUC, and per-class metrics."""
        from src.models.xgboost_binary_imbalance import evaluate_binary_predictions

        # 70 TN, 2 FP, 1 FN, 927 TP
        y_true = pd.Series([0] * 72 + [1] * 928)
        y_pred = np.array([0] * 70 + [1] * 2 + [0] * 1 + [1] * 927)
        y_proba = np.array([0.05] * 70 + [0.8] * 2 + [0.4] * 1 + [0.95] * 927)

        metrics = evaluate_binary_predictions(y_true, y_pred, y_proba)

        self.assertEqual(metrics["confusion_matrix"]["true_negatives_normal"], 70)
        self.assertEqual(metrics["confusion_matrix"]["false_positives_normal_flagged_as_attack"], 2)
        self.assertEqual(metrics["confusion_matrix"]["false_negatives_attack_missed"], 1)
        self.assertEqual(metrics["confusion_matrix"]["true_positives_attack_detected"], 927)

        # FPR = FP / (FP + TN) = 2 / 72
        expected_fpr = 2.0 / 72.0
        self.assertAlmostEqual(metrics["false_positive_rate"], expected_fpr, places=6)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("roc_auc", metrics)


if __name__ == "__main__":
    unittest.main()
