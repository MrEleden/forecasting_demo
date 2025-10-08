"""
Stacking Ensemble for time series forecasting.

Combines multiple base models with a meta-learner.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ...data.datasets import TimeSeriesDataset
from ...data.loaders import SimpleDataLoader
from ..base import StatisticalForecaster

logger = logging.getLogger(__name__)


class StackingForecaster(StatisticalForecaster):
    """
    Stacking ensemble forecaster with sklearn-like interface.

    Stacking combines multiple base models by:
    - Training base models on the training data
    - Using their predictions as features for a meta-learner
    - Meta-learner learns optimal combination weights
    - Better than simple averaging when models have different strengths

    Great for combining diverse model types (statistical + ML + DL).

    Args:
        base_models: List of (name, model) tuples for base learners
        meta_model: Meta-learner model (default: Ridge regression)
        use_features: Include original features in meta-learner
        cv_folds: Number of cross-validation folds for meta-features

    Example:
        >>> from ml_portfolio.models.ensemble.stacking import StackingForecaster
        >>> from ml_portfolio.models.statistical.prophet import ProphetForecaster
        >>> from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
        >>>
        >>> base_models = [
        ...     ("prophet", ProphetForecaster()),
        ...     ("lgbm", LightGBMForecaster())
        ... ]
        >>> model = StackingForecaster(base_models=base_models)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        meta_model: Optional[Any] = None,
        use_features: bool = False,
        cv_folds: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge()
        self.use_features = use_features
        self.cv_folds = cv_folds
        self.kwargs = kwargs

        # Will be set during fit
        self.fitted_base_models_ = []
        self.feature_names_ = None
        self.n_features_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Internal fit implementation called by StatisticalForecaster."""

        X_df = self._ensure_dataframe(X, fit=True)
        y_array = self._ensure_array(y)

        dataset = TimeSeriesDataset(X_df.values, y_array, feature_names=self.feature_names_)
        loader = SimpleDataLoader(dataset, shuffle=False)

        base_predictions = []
        self.fitted_base_models_ = []

        for name, model in self.base_models:
            logger.info(f"Training base model: {name}")
            model.fit(train_loader=loader)
            self.fitted_base_models_.append((name, model))

            preds = np.asarray(model.predict(X_df.values)).reshape(-1, 1)
            base_predictions.append(preds)

        if not base_predictions:
            raise ValueError("No base models were provided to the stacking ensemble")

        meta_features = np.hstack(base_predictions)

        if self.use_features:
            meta_features = np.hstack([meta_features, X_df.values])

        logger.info("Training meta-learner")
        self.meta_model.fit(meta_features, y_array)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features dataframe
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Predicted values
        """
        if not self.fitted_base_models_:
            raise ValueError("Model must be fit before making predictions")

        X_df = self._ensure_dataframe(X, fit=False)

        # Get predictions from base models
        base_predictions = []
        for name, model in self.fitted_base_models_:
            preds = np.asarray(model.predict(X_df.values)).reshape(-1, 1)
            base_predictions.append(preds)

        # Stack predictions
        meta_features = np.hstack(base_predictions)

        # Include original features if specified
        if self.use_features:
            meta_features = np.hstack([meta_features, X_df.values])

        # Meta-learner prediction
        predictions = self.meta_model.predict(meta_features)

        return predictions

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "base_models": self.base_models,
            "meta_model": self.meta_model,
            "use_features": self.use_features,
            "cv_folds": self.cv_folds,
            **self.kwargs,
        }

    def set_params(self, **params) -> "StackingForecaster":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        info = {
            "num_base_models": len(self.fitted_base_models_),
            "base_model_names": [name for name, _ in self.fitted_base_models_],
            "meta_model": type(self.meta_model).__name__,
            "use_features": self.use_features,
        }
        return info

    def _ensure_dataframe(self, X: Any, fit: bool) -> pd.DataFrame:
        """Ensure input is a DataFrame and track feature names."""

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            if fit or self.feature_names_ is None:
                self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self.feature_names_)

        if fit:
            self.feature_names_ = df.columns.tolist()
            self.n_features_ = df.shape[1]
        else:
            if self.feature_names_ is not None and list(df.columns) != list(self.feature_names_):
                # Reorder or regenerate columns to match training layout
                if df.shape[1] != self.n_features_:
                    raise ValueError("Input feature dimension does not match training data")
                df.columns = self.feature_names_

        return df

    @staticmethod
    def _ensure_array(y: Any) -> np.ndarray:
        """Convert target to 1D numpy array."""

        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        elif isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.asarray(y)

        if y.ndim > 1:
            y = y.ravel()

        return y
