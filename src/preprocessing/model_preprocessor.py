"""
Feature Transformation and Pipeline Preprocessor for ML Models
"""

from typing import List, Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config.settings import CANONICAL_22_FEATURES

# Categorical and Numerical feature partition within the canonical 22
CATEGORICAL_FEATURES: List[str] = ["proto", "flgs", "state"]
NUMERICAL_FEATURES: List[str] = [
    col for col in CANONICAL_22_FEATURES if col not in CATEGORICAL_FEATURES
]


def build_feature_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that applies One-Hot Encoding
    to categorical features and passes numerical features through untouched.
    Fitted strictly on training data only with handle_unknown='ignore'.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
