"""Shared feature transformations for the SVM and MLP experiments."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import CANONICAL_22_FEATURES

CATEGORICAL_FEATURES = ["proto", "flgs", "state"]
NUMERICAL_FEATURES = [name for name in CANONICAL_22_FEATURES if name not in CATEGORICAL_FEATURES]


def build_feature_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals and standardize numeric features."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
