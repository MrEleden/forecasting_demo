"""
Stacking Ensemble for time series forecasting.

Combines multiple base models with a meta-learner.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..base import StatisticalForecaster


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
        base_models: List[tuple],
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

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fit stacking ensemble.

        Args:
            X: Features dataframe
            y: Target values
            **kwargs: Additional fitting parameters

        Returns:
            self: Fitted model
        """
        self.feature_names_ = list(X.columns)

        # Train base models and collect predictions
        base_predictions = []
        self.fitted_base_models_ = []

        for name, model in self.base_models:
            print(f"Training base model: {name}")

            # Fit model
            model.fit(X, y)
            self.fitted_base_models_.append((name, model))

            # Get out-of-fold predictions for meta-learner
            # Simple approach: use same data (can improve with CV)
            preds = model.predict(X)
            base_predictions.append(preds)

        # Stack base predictions
        meta_features = np.column_stack(base_predictions)

        # Optionally include original features
        if self.use_features:
            meta_features = np.hstack([meta_features, X.values])

        # Train meta-learner
        print("Training meta-learner")
        self.meta_model.fit(meta_features, y)

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

        # Get predictions from base models
        base_predictions = []
        for name, model in self.fitted_base_models_:
            preds = model.predict(X)
            base_predictions.append(preds)

        # Stack predictions
        meta_features = np.column_stack(base_predictions)

        # Include original features if specified
        if self.use_features:
            meta_features = np.hstack([meta_features, X.values])

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
