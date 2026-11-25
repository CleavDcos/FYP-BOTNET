"""
Inference Preprocessing and Input Validation Module.

Validates incoming network traffic records, isolates the canonical 22 features,
strictly preserves canonical column ordering, ensures data type compatibility,
and prevents target leakage without fitting new transformations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from config.settings import CANONICAL_22_FEATURES, TARGET_LABELS

CATEGORICAL_FEATURES: List[str] = ["proto", "flgs", "state"]
NUMERIC_FEATURES: List[str] = [
    f for f in CANONICAL_22_FEATURES if f not in CATEGORICAL_FEATURES
]


class InferenceError(Exception):
    """Base exception for all inference pipeline operations."""
    pass


class InferenceValidationError(InferenceError):
    """Exception raised when input data fails validation."""
    pass


class InferencePreprocessor:
    """
    Validates and formats raw network flow records for model consumption.

    Guarantees:
    - Exactly the canonical 22 features are selected.
    - Canonical feature ordering is strictly preserved.
    - Zero target leakage (strips target columns if passed).
    - Safe data type conversion for numerical and categorical features.
    - Zero transformation fitting during inference.
    """

    def __init__(
        self,
        canonical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        strict_numeric: bool = True,
    ) -> None:
        self.canonical_features = canonical_features or list(CANONICAL_22_FEATURES)
        self.categorical_features = categorical_features or list(CATEGORICAL_FEATURES)
        self.numeric_features = numeric_features or list(NUMERIC_FEATURES)
        self.strict_numeric = strict_numeric

    def validate_and_transform(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Validates input records and formats them as a clean pandas DataFrame.

        Parameters
        ----------
        data : dict, list of dicts, pd.Series, or pd.DataFrame
            Incoming network traffic flow record(s).

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame containing strictly the canonical 22 features
            in canonical order with validated data types.

        Raises
        ------
        InferenceValidationError
            If input type is invalid, empty, or missing required canonical features.
        """
        df = self._to_dataframe(data)

        if df.empty:
            raise InferenceValidationError("Input dataset is empty (zero records provided).")

        # Strip target columns if mistakenly passed in raw feed to prevent leakage
        cols_to_drop = [c for c in TARGET_LABELS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Check for missing canonical features
        missing_features = [f for f in self.canonical_features if f not in df.columns]
        if missing_features:
            raise InferenceValidationError(
                f"Missing required canonical feature(s): {missing_features}. "
                f"Expected all {len(self.canonical_features)} canonical features."
            )

        # Select strictly the canonical features in exact canonical ordering
        df_canonical = df[self.canonical_features].copy()

        # Sanitize and type-cast features
        df_canonical = self._sanitize_types(df_canonical)

        return df_canonical

    def _to_dataframe(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        """Converts diverse input types into a standardized pandas DataFrame."""
        if data is None:
            raise InferenceValidationError("Input data cannot be None.")

        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, pd.Series):
            return pd.DataFrame([data])

        if isinstance(data, dict):
            if not data:
                raise InferenceValidationError("Input dictionary is empty.")
            return pd.DataFrame([data])

        if isinstance(data, list):
            if not data:
                raise InferenceValidationError("Input list is empty.")
            if not all(isinstance(item, dict) for item in data):
                raise InferenceValidationError(
                    "All items in the input list must be dictionaries representing records."
                )
            return pd.DataFrame(data)

        raise InferenceValidationError(
            f"Unsupported input type '{type(data).__name__}'. "
            "Expected dict, list of dicts, pandas Series, or pandas DataFrame."
        )

    def _sanitize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures correct types for numerical and categorical features."""
        # 1. Categoricals: cast to string, handle None/NaN safely
        for col in self.categorical_features:
            # Replace NaNs or None with a string placeholder if needed
            df[col] = df[col].apply(lambda x: "" if pd.isna(x) else str(x))

        # 2. Numerics: safely convert to numeric floats
        for col in self.numeric_features:
            original_series = df[col]
            numeric_series = pd.to_numeric(original_series, errors="coerce")

            if self.strict_numeric:
                # If non-NaN values became NaN after conversion, non-numeric input was supplied
                invalid_mask = original_series.notna() & numeric_series.isna()
                if invalid_mask.any():
                    invalid_val = original_series[invalid_mask].iloc[0]
                    raise InferenceValidationError(
                        f"Non-numeric value '{invalid_val}' encountered in numeric feature '{col}'."
                    )

            df[col] = numeric_series

        return df
