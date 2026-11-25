"""
Predictor Wrappers for XGBoost Binary and LightGBM Multiclass Models.

Abstracts model execution, handles pipeline differences, extracts probability
estimates, and maps class labels dynamically from model attributes.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class XGBoostBinaryPredictor:
    """
    Predictor wrapper for Stage 1 XGBoost Binary Classification.

    Maps:
      0 -> 'Normal'
      1 -> 'Malicious'
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._is_dict_bundle = isinstance(model, dict) and "classifier" in model
        if self._is_dict_bundle:
            self.preprocessor = model["preprocessor"]
            self.classifier = model["classifier"]
        else:
            self.preprocessor = None
            self.classifier = model

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes binary prediction on preprocessed flow features.

        Parameters
        ----------
        X : pd.DataFrame
            Canonical 22 features dataframe.

        Returns
        -------
        raw_preds : np.ndarray
            Binary integers (0 or 1).
        labels : np.ndarray
            Mapped string labels ('Normal' or 'Malicious').
        confidences : np.ndarray
            Confidence scores for the predicted class.
        probabilities : np.ndarray
            2D array of [P(Normal), P(Malicious)].
        """
        if self._is_dict_bundle:
            X_trans = self.preprocessor.transform(X)
            raw_preds = self.classifier.predict(X_trans)
            probabilities = self.classifier.predict_proba(X_trans)
        else:
            raw_preds = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

        raw_preds = np.asarray(raw_preds).astype(int)
        probabilities = np.asarray(probabilities)

        # Map labels: 0 -> Normal, 1 -> Malicious
        label_map = {0: "Normal", 1: "Malicious"}
        labels = np.array([label_map.get(p, "Malicious") for p in raw_preds])

        # Confidence is the probability of the predicted class
        confidences = np.array([
            probabilities[i, raw_preds[i]] if probabilities.shape[1] > raw_preds[i] else 1.0
            for i in range(len(raw_preds))
        ])

        return raw_preds, labels, confidences, probabilities


class LightGBMCategoryPredictor:
    """
    Predictor wrapper for Stage 2 LightGBM Attack Category Classification.

    Determines categories dynamically from model classes without hardcoding.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.classes_ = self._extract_classes(model)

    def _extract_classes(self, model: Any) -> List[str]:
        """Dynamically retrieves class labels from the fitted classifier."""
        if hasattr(model, "classes_"):
            return [str(c) for c in model.classes_]
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "classes_"):
                return [str(c) for c in classifier.classes_]
        return []

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes attack category prediction on malicious records.

        Parameters
        ----------
        X : pd.DataFrame
            Canonical 22 features dataframe for records flagged as Malicious.

        Returns
        -------
        categories : np.ndarray
            Predicted attack category string for each record.
        confidences : np.ndarray
            Confidence scores (max probability) for predicted category.
        probabilities : np.ndarray
            2D array of class probabilities across all learned categories.
        """
        if len(X) == 0:
            return np.array([]), np.array([]), np.empty((0, len(self.classes_)))

        raw_preds = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        probabilities = np.asarray(probabilities)
        categories = np.asarray(raw_preds).astype(str)

        # Extract confidence for the predicted class
        confidences = np.max(probabilities, axis=1)

        return categories, confidences, probabilities

    def get_classes(self) -> List[str]:
        """Returns the list of learned attack categories."""
        return list(self.classes_)
