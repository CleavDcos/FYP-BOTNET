"""
Cascaded Inference Pipeline Orchestrator.

Integrates preprocessing, XGBoost binary detection (Stage 1), and LightGBM
attack category classification (Stage 2) into a cohesive, production-grade
inference engine.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.inference.preprocessing import (
    InferencePreprocessor,
    InferenceValidationError,
)
from src.inference.model_loader import ModelLoader
from src.inference.predictor import (
    XGBoostBinaryPredictor,
    LightGBMCategoryPredictor,
)


class CascadedInferencePipeline:
    """
    End-to-End Cascaded Inference Pipeline for IoT Botnet Detection.

    Cascaded Workflow:
    1. Preprocessing: Validates input, selects canonical 22 features, enforces types.
    2. Stage 1 (XGBoost): Classifies traffic as Normal vs Malicious.
    3. Conditional Branch:
       - If Normal: Returns result immediately. LightGBM is SKIPPED.
       - If Malicious: Passes flow to Stage 2 (LightGBM) to identify attack category.
    4. Output: Formats unified structured dictionary.
    """

    def __init__(
        self,
        xgb_predictor: XGBoostBinaryPredictor,
        lgb_predictor: LightGBMCategoryPredictor,
        preprocessor: Optional[InferencePreprocessor] = None,
    ) -> None:
        self.xgb_predictor = xgb_predictor
        self.lgb_predictor = lgb_predictor
        self.preprocessor = preprocessor or InferencePreprocessor()

    @classmethod
    def from_models(
        cls,
        xgb_model: Any,
        lgb_model: Any,
        preprocessor: Optional[InferencePreprocessor] = None,
    ) -> "CascadedInferencePipeline":
        """Instantiates pipeline directly from loaded model objects."""
        xgb_predictor = XGBoostBinaryPredictor(xgb_model)
        lgb_predictor = LightGBMCategoryPredictor(lgb_model)
        return cls(
            xgb_predictor=xgb_predictor,
            lgb_predictor=lgb_predictor,
            preprocessor=preprocessor,
        )

    @classmethod
    def from_loader(
        cls,
        loader: Optional[ModelLoader] = None,
        preprocessor: Optional[InferencePreprocessor] = None,
    ) -> "CascadedInferencePipeline":
        """Instantiates pipeline using ModelLoader to retrieve artifacts."""
        model_loader = loader or ModelLoader()
        xgb_model, lgb_model = model_loader.load_all()
        return cls.from_models(
            xgb_model=xgb_model,
            lgb_model=lgb_model,
            preprocessor=preprocessor,
        )

    def predict_single(
        self,
        record: Union[Dict[str, Any], pd.Series],
    ) -> Dict[str, Any]:
        """
        Executes cascaded inference on a single network traffic record.

        Parameters
        ----------
        record : dict or pd.Series
            Single flow feature dictionary or Series.

        Returns
        -------
        dict
            Unified detection output dictionary.
        """
        results = self.predict_batch([record if isinstance(record, dict) else record.to_dict()])
        return results[0]

    def predict_batch(
        self,
        records: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Executes cascaded inference over a batch of records.

        Optimized cascaded execution:
        - All records pass Stage 1 (XGBoost).
        - ONLY records predicted as Malicious are sliced and passed to Stage 2 (LightGBM).
        - LightGBM is completely skipped if all records are Normal.

        Parameters
        ----------
        records : list of dicts or pd.DataFrame
            Batch of network traffic flow records.

        Returns
        -------
        list of dict
            Structured detection results for each record.
        """
        # Step 1: Preprocess and validate input
        X_canonical = self.preprocessor.validate_and_transform(records)
        num_samples = len(X_canonical)

        # Step 2: Stage 1 - XGBoost Binary Classification
        raw_preds, labels, confidences, probabilities = self.xgb_predictor.predict(X_canonical)

        results: List[Dict[str, Any]] = [None] * num_samples  # type: ignore
        malicious_indices: List[int] = []

        # Populate Normal results immediately; collect malicious indices
        for i in range(num_samples):
            is_attack = bool(raw_preds[i] == 1)
            bin_label = str(labels[i])
            bin_conf = float(confidences[i])

            if not is_attack:
                # Normal: LightGBM is SKIPPED
                results[i] = {
                    "status": "success",
                    "binary_prediction": bin_label,
                    "attack_detected": False,
                    "attack_category": None,
                    "binary_confidence": round(bin_conf, 4),
                    "category_confidence": None,
                }
            else:
                malicious_indices.append(i)

        # Step 3: Stage 2 - LightGBM Multiclass Classification (Malicious ONLY)
        if malicious_indices:
            X_malicious = X_canonical.iloc[malicious_indices]
            categories, cat_confidences, _ = self.lgb_predictor.predict(X_malicious)

            for idx_in_mal, orig_idx in enumerate(malicious_indices):
                pred_cat = str(categories[idx_in_mal])
                cat_conf = float(cat_confidences[idx_in_mal])
                bin_conf = float(confidences[orig_idx])

                results[orig_idx] = {
                    "status": "success",
                    "binary_prediction": "Malicious",
                    "attack_detected": True,
                    "attack_category": pred_cat,
                    "binary_confidence": round(bin_conf, 4),
                    "category_confidence": round(cat_conf, 4),
                }

        return results

    def predict(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.Series, pd.DataFrame],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Universal entrypoint for cascaded inference.

        Automatically dispatches between single-record and batch predictions.
        """
        if isinstance(data, (dict, pd.Series)):
            return self.predict_single(data)
        return self.predict_batch(data)

    def get_supported_categories(self) -> List[str]:
        """Returns the learned attack categories supported by Stage 2 LightGBM."""
        return self.lgb_predictor.get_classes()
