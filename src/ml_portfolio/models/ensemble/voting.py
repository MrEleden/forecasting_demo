"""
Voting Ensemble for time series forecasting.

Combines predictions from multiple models via averaging or weighted voting.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster


class VotingForecaster(StatisticalForecaster):
    """
    Voting ensemble forecaster with sklearn-like interface.

    Voting combines multiple models by:
    - Training each model independently
    - Averaging their predictions (uniform or weighted)
    - Simple but effective for diverse models
    - More robust than individual models

    Great for quick ensembles without meta-learning complexity.

    Args:
        models: List of (name, model) tuples
        weights: Optional weights for each model (default: uniform)
        voting_type: 'mean' or 'median'

    Example:
        >>> from ml_portfolio.models.ensemble.voting import VotingForecaster
        >>> from ml_portfolio.models.statistical.prophet import ProphetForecaster
        >>> from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
        >>> from ml_portfolio.models.statistical.xgboost import XGBoostForecaster
        >>>
        >>> models = [
        ...     ("prophet", ProphetForecaster()),
        ...     ("lgbm", LightGBMForecaster()),
        ...     ("xgb", XGBoostForecaster())
        ... ]
        >>> # Equal weights
        >>> model = VotingForecaster(models=models)
        >>> # Or custom weights
        >>> model = VotingForecaster(models=models, weights=[0.5, 0.3, 0.2])
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        models: List[tuple],
        weights: Optional[List[float]] = None,
        voting_type: str = "mean",
        **kwargs,
    ):
        super().__init__()

        self.models = models
        self.weights = weights
        self.voting_type = voting_type
        self.kwargs = kwargs

        # Validate weights
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(self.weights), 1.0):
                raise ValueError("Weights must sum to 1.0")

        # Will be set during fit
        self.fitted_models_ = []
        self.feature_names_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Internal fit method for voting ensemble.

        Args:
            X: Features array
            y: Target values
        """
        # Store feature names if available
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)

        # Train each model
        self.fitted_models_ = []
        for name, model in self.models:
            print(f"Training model: {name}")
            # Models expect arrays, not dataloaders at this level
            model._fit(X, y) if hasattr(model, "_fit") else model.fit(X, y)
            self.fitted_models_.append((name, model))

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features dataframe
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Predicted values
        """
        if not self.fitted_models_:
            raise ValueError("Model must be fit before making predictions")

        # Collect predictions from all models
        all_predictions = []
        for name, model in self.fitted_models_:
            preds = model.predict(X)
            all_predictions.append(preds)

        # Stack predictions (n_models, n_samples)
        all_predictions = np.array(all_predictions)

        # Combine predictions
        if self.voting_type == "mean":
            if self.weights is None:
                # Uniform average
                predictions = np.mean(all_predictions, axis=0)
            else:
                # Weighted average
                weights = np.array(self.weights).reshape(-1, 1)
                predictions = np.sum(all_predictions * weights, axis=0)
        elif self.voting_type == "median":
            predictions = np.median(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown voting_type: {self.voting_type}")

        return predictions

    def predict_with_individual(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from ensemble and individual models.

        Args:
            X: Features dataframe

        Returns:
            predictions: Dictionary with 'ensemble' and individual model predictions
        """
        results = {}

        # Individual predictions
        for name, model in self.fitted_models_:
            results[name] = model.predict(X)

        # Ensemble prediction
        results["ensemble"] = self.predict(X)

        return results

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "models": self.models,
            "weights": self.weights,
            "voting_type": self.voting_type,
            **self.kwargs,
        }

    def set_params(self, **params) -> "VotingForecaster":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        info = {
            "num_models": len(self.fitted_models_),
            "model_names": [name for name, _ in self.fitted_models_],
            "weights": self.weights if self.weights else "uniform",
            "voting_type": self.voting_type,
        }
        return info
