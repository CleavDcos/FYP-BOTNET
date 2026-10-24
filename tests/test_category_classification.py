"""
Unit Tests for Multiclass Attack-Category Classification Module
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from config.settings import (
    CANONICAL_22_FEATURES,
    TARGET_LABELS,
    TRAIN_V2_CLEAN_PARQUET,
    VAL_V2_CLEAN_PARQUET,
)
from src.preprocessing.model_preprocessor import build_feature_preprocessor
from src.models.random_forest_category import (
    EXPECTED_ATTACK_CATEGORIES,
    MIN_SAMPLES_RELIABLE_EVAL,
    load_malicious_category_datasets,
    evaluate_category_model,
)


class TestCategoryClassification(unittest.TestCase):
    """Test suite verifying malicious category filtering, representation, label consistency and evaluation."""

    def test_expected_categories_definition(self):
        """Verify expected attack categories list exactly contains 4 attack classes and excludes Normal."""
        self.assertEqual(len(EXPECTED_ATTACK_CATEGORIES), 4)
        self.assertEqual(set(EXPECTED_ATTACK_CATEGORIES), {"DDoS", "DoS", "Reconnaissance", "Theft"})
        self.assertNotIn("Normal", EXPECTED_ATTACK_CATEGORIES)

    def test_malicious_dataset_loading_and_filtering(self):
        """Verify load_malicious_category_datasets filters out Normal and keeps strictly attack==1."""
        X_train, y_train, X_val, y_val = load_malicious_category_datasets()

        # Check shapes match known row counts
        self.assertEqual(len(X_train), 2567631)
        self.assertEqual(len(X_val), 550206)

        # Confirm 22 canonical features
        self.assertEqual(list(X_train.columns), CANONICAL_22_FEATURES)
        self.assertEqual(list(X_val.columns), CANONICAL_22_FEATURES)

        # Confirm zero leakage
        for target in TARGET_LABELS:
            self.assertNotIn(target, X_train.columns)
            self.assertNotIn(target, X_val.columns)

        # Confirm target category contains all 4 expected categories and zero Normal
        self.assertEqual(set(y_train.unique()), set(EXPECTED_ATTACK_CATEGORIES))
        self.assertEqual(set(y_val.unique()), set(EXPECTED_ATTACK_CATEGORIES))
        self.assertNotIn("Normal", y_train.unique())
        self.assertNotIn("Normal", y_val.unique())

        # Exact class counts match known clean dataset malicious distribution
        self.assertEqual(int((y_train == "Theft").sum()), 55)
        self.assertEqual(int((y_val == "Theft").sum()), 12)
        self.assertEqual(int((y_train == "DDoS").sum()), 1348005)
        self.assertEqual(int((y_val == "DDoS").sum()), 289275)

    def test_evaluate_category_model_metrics_computation(self):
        """Verify evaluate_category_model accurately computes per-class and aggregated metrics."""
        # Create a synthetic validation ground truth and dummy predictions
        y_true = pd.Series(["DDoS"] * 100 + ["DoS"] * 80 + ["Reconnaissance"] * 20 + ["Theft"] * 5)
        X_dummy = pd.DataFrame(np.zeros((len(y_true), len(CANONICAL_22_FEATURES))), columns=CANONICAL_22_FEATURES)

        # Dummy classifier that predicts actual classes to test calculation correctness
        dummy_model = DummyClassifier(strategy="constant", constant="DDoS")
        dummy_model.fit(X_dummy, y_true)

        pipeline = Pipeline([("dummy", dummy_model)])
        metrics = evaluate_category_model(pipeline, X_dummy, y_true, target_names=EXPECTED_ATTACK_CATEGORIES)

        self.assertIn("overall_accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("per_class_metrics", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Theft has 5 samples (< 50 threshold), so it must be flagged
        theft_metrics = metrics["per_class_metrics"]["Theft"]
        self.assertEqual(theft_metrics["support"], 5)
        self.assertFalse(theft_metrics["reliable_evaluation"])
        self.assertIn("Theft", metrics["underrepresented_classes_flags"])

        # DDoS has 100 samples (>= 50 threshold), so reliable_evaluation should be True
        ddos_metrics = metrics["per_class_metrics"]["DDoS"]
        self.assertEqual(ddos_metrics["support"], 100)
        self.assertTrue(ddos_metrics["reliable_evaluation"])

    def test_sampling_datasets_loading_and_counts(self):
        """Verify load_training_variant handles Clean, ROS, and SMOTE correctly with exact counts."""
        from src.models.random_forest_category_sampling import load_training_variant, load_validation_data

        # Test validation data isolation
        X_val, y_val = load_validation_data()
        self.assertEqual(len(X_val), 550206)
        self.assertNotIn("Normal", y_val.unique())
        self.assertEqual(int((y_val == "Theft").sum()), 12)

        # Test Clean variant
        X_clean, y_clean = load_training_variant("clean")
        self.assertEqual(len(X_clean), 2567631)
        self.assertEqual(int((y_clean == "Theft").sum()), 55)

        # Test ROS variant
        X_ros, y_ros = load_training_variant("ros")
        self.assertEqual(len(X_ros), 2572576)
        self.assertEqual(int((y_ros == "Theft").sum()), 5000)
        self.assertNotIn("Normal", y_ros.unique())

        # Test SMOTENC variant
        X_smote, y_smote = load_training_variant("smote")
        self.assertEqual(len(X_smote), 2572576)
        self.assertEqual(int((y_smote == "Theft").sum()), 5000)
        self.assertNotIn("Normal", y_smote.unique())


if __name__ == "__main__":
    unittest.main()

