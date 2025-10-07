"""
LightGBM forecasting model wrapper.

Provides sklearn-compatible interface for LightGBM regressor.
Optimized for time series forecasting with gradient boosting.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMForecaster(StatisticalForecaster):
    """
    LightGBM forecasting model with sklearn interface.

    Gradient boosting machine optimized for:
    - Fast training speed
    - Memory efficiency
    - Handling missing values
    - Feature importance analysis

    Popular choice for Kaggle competitions and production systems.

    Args:
        n_estimators: Number of boosting iterations
        learning_rate: Boosting learning rate
        max_depth: Maximum tree depth for base learners
        num_leaves: Maximum number of leaves in one tree
        min_child_samples: Minimum number of data needed in a child
        min_child_weight: Minimum sum of instance weight needed in a child
        subsample: Subsample ratio of the training instance
        subsample_freq: Frequency of subsample
        colsample_bytree: Subsample ratio of columns when constructing each tree
        reg_alpha: L1 regularization term on weights
        reg_lambda: L2 regularization term on weights
        n_jobs: Number of parallel threads (-1 for all cores)
        random_state: Random seed for reproducibility
        verbose: Controls verbosity of training
        early_stopping_rounds: Early stopping rounds (requires validation set)
        **kwargs: Additional LightGBM parameters

    Example:
        >>> from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
        >>> model = LightGBMForecaster(n_estimators=500, learning_rate=0.05)
        >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        min_child_weight: float = 0.001,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = -1,
        early_stopping_rounds: Optional[int] = 50,
        **kwargs,
    ):
        # Initialize base class
        super().__init__()

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs

        # Will be initialized in fit()
        self.model = None
        self.feature_names_ = None
        self.feature_importances_ = None

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        eval_names: Optional[list] = None,
        **fit_kwargs,
    ):
        """
        Internal fit method for LightGBM model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            eval_set: List of (X, y) tuples for validation
            eval_names: Names for evaluation sets
            **fit_kwargs: Additional fit parameters
        """
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]

        # Convert y to 1D array
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if y.ndim > 1:
            y = y.ravel()

        # Initialize model
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.kwargs,
        )

        # Prepare callbacks for early stopping
        callbacks = []
        if eval_set is not None and self.early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))

        # Fit model
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks if callbacks else None,
            **fit_kwargs,
        )

        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Mark as fitted
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the fitted model.

        Args:
            X: Features to predict on (n_samples, n_features)

        Returns:
            predictions: Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R^2 score on test data.

        Args:
            X: Test features
            y: True target values

        Returns:
            r2_score: R^2 coefficient of determination
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if y.ndim > 1:
            y = y.ravel()

        return self.model.score(X, y)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('split' or 'gain')

        Returns:
            feature_importance: Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if importance_type == "gain":
            importances = self.model.feature_importances_
        elif importance_type == "split":
            # Get split-based importance from booster
            importances = self.model.booster_.feature_importance(importance_type="split")
        else:
            raise ValueError(f"Unknown importance_type: {importance_type}")

        return dict(zip(self.feature_names_, importances))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get model parameters (sklearn compatibility).

        Args:
            deep: If True, return parameters for nested objects

        Returns:
            params: Model parameters
        """
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params) -> "LightGBMForecaster":
        """
        Set model parameters (sklearn compatibility).

        Args:
            **params: Parameters to set

        Returns:
            self: Updated model
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"LightGBMForecaster(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})"
        )
