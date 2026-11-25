"""
Model Loader and Artifact Management Module.

Provides cached loading of trained XGBoost binary classification and
LightGBM multiclass classification pipelines, ensuring artifacts are loaded
once and reused across inference invocations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib

from config.settings import PROJECT_ROOT
from src.inference.preprocessing import InferenceError

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_XGB_MODEL_PATH = MODELS_DIR / "xgb_binary_pipeline.joblib"
DEFAULT_LGB_MODEL_PATH = MODELS_DIR / "lightgbm_category.joblib"


class ModelArtifactNotFoundError(InferenceError):
    """Exception raised when a required model artifact cannot be found on disk."""
    pass


class ModelArtifactLoadError(InferenceError):
    """Exception raised when a model artifact fails to deserialize or load."""
    pass


class ModelLoader:
    """
    Manages loading and caching of serialized machine learning models.

    Avoids reloading heavy pipeline artifacts repeatedly on each request.
    """

    def __init__(
        self,
        xgb_path: Optional[Union[str, Path]] = None,
        lgb_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.xgb_path = Path(xgb_path) if xgb_path else DEFAULT_XGB_MODEL_PATH
        self.lgb_path = Path(lgb_path) if lgb_path else DEFAULT_LGB_MODEL_PATH
        self._cache: Dict[str, Any] = {}

    def get_xgb_model(self, force_reload: bool = False) -> Any:
        """
        Loads and caches the XGBoost binary classification pipeline.

        Returns
        -------
        Any
            Trained pipeline or model object.

        Raises
        ------
        ModelArtifactNotFoundError
            If model file is not present on disk.
        ModelArtifactLoadError
            If model deserialization fails.
        """
        key = f"xgb:{self.xgb_path.resolve()}"
        if not force_reload and key in self._cache:
            return self._cache[key]

        if not self.xgb_path.exists():
            # Check fallback if default wasn't found
            fallback = MODELS_DIR / "xgb_binary_sample_weight_balanced_pipeline.joblib"
            if self.xgb_path == DEFAULT_XGB_MODEL_PATH and fallback.exists():
                self.xgb_path = fallback
                key = f"xgb:{self.xgb_path.resolve()}"
            else:
                raise ModelArtifactNotFoundError(
                    f"XGBoost model artifact not found at: '{self.xgb_path}'. "
                    f"Please ensure models are trained and saved in {MODELS_DIR}."
                )

        try:
            model = joblib.load(self.xgb_path)
            self._cache[key] = model
            return model
        except Exception as exc:
            raise ModelArtifactLoadError(
                f"Failed to load XGBoost model from '{self.xgb_path}': {exc}"
            ) from exc

    def get_lgb_model(self, force_reload: bool = False) -> Any:
        """
        Loads and caches the LightGBM category classification pipeline.

        Returns
        -------
        Any
            Trained LightGBM pipeline or model object.

        Raises
        ------
        ModelArtifactNotFoundError
            If model file is not present on disk.
        ModelArtifactLoadError
            If model deserialization fails.
        """
        key = f"lgb:{self.lgb_path.resolve()}"
        if not force_reload and key in self._cache:
            return self._cache[key]

        if not self.lgb_path.exists():
            raise ModelArtifactNotFoundError(
                f"LightGBM model artifact not found at: '{self.lgb_path}'. "
                f"Please ensure train_lightgbm.py has been run and saved in {MODELS_DIR}."
            )

        try:
            model = joblib.load(self.lgb_path)
            self._cache[key] = model
            return model
        except Exception as exc:
            raise ModelArtifactLoadError(
                f"Failed to load LightGBM model from '{self.lgb_path}': {exc}"
            ) from exc

    def load_all(self, force_reload: bool = False) -> Tuple[Any, Any]:
        """Loads both XGBoost and LightGBM models."""
        xgb_model = self.get_xgb_model(force_reload=force_reload)
        lgb_model = self.get_lgb_model(force_reload=force_reload)
        return xgb_model, lgb_model

    def clear_cache(self) -> None:
        """Clears all cached model artifacts in memory."""
        self._cache.clear()
