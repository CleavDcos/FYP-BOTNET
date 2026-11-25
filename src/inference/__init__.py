"""
Modular Cascaded Inference Pipeline Package for IoT Botnet Detection.

Provides an end-to-end, cascaded inference architecture:
  Stage 1: XGBoost Binary Classification (Normal vs Malicious)
  Stage 2: LightGBM Attack Category Classification (Executed strictly for Malicious traffic)
"""

from src.inference.preprocessing import (
    InferencePreprocessor,
    InferenceError,
    InferenceValidationError,
)
from src.inference.model_loader import (
    ModelLoader,
    ModelArtifactNotFoundError,
    ModelArtifactLoadError,
)
from src.inference.predictor import (
    XGBoostBinaryPredictor,
    LightGBMCategoryPredictor,
)
from src.inference.inference_pipeline import (
    CascadedInferencePipeline,
)

__all__ = [
    "InferencePreprocessor",
    "InferenceError",
    "InferenceValidationError",
    "ModelLoader",
    "ModelArtifactNotFoundError",
    "ModelArtifactLoadError",
    "XGBoostBinaryPredictor",
    "LightGBMCategoryPredictor",
    "CascadedInferencePipeline",
]
