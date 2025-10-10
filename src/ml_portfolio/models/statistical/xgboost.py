"""
XGBoost forecasting model wrapper.

Provides sklearn-compatible interface for XGBoost regressor.
Popular choice for Kaggle competitions with robust performance.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostForecaster(StatisticalForecaster):
    """
    XGBoost forecasting model with sklearn interface.

    Extreme Gradient Boosting optimized for:
    - Robust predictions on structured data
    - Regularization to prevent overfitting
    - Feature importance analysis
    - Parallel processing

    Popular in Kaggle competitions and production environments.

    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate (eta)
        max_depth: Maximum tree depth for base learners
        min_child_weight: Minimum sum of instance weight needed in a child
        subsample: Subsample ratio of the training instance
        colsample_bytree: Subsample ratio of columns when constructing each tree
        gamma: Minimum loss reduction required to make a split
        reg_alpha: L1 regularization term on weights
        reg_lambda: L2 regularization term on weights
        tree_method: Tree construction algorithm
        n_jobs: Number of parallel threads (-1 for all cores)
        random_state: Random seed for reproducibility
        verbosity: Verbosity of printing messages (0=silent, 1=warning, 2=info, 3=debug)
        early_stopping_rounds: Early stopping rounds (requires validation set)
        **kwargs: Additional XGBoost parameters

    Example:
        >>> from ml_portfolio.models.statistical.xgboost import XGBoostForecaster
        >>> model = XGBoostForecaster(n_estimators=500, learning_rate=0.05)
        >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        min_child_weight: float = 3.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        n_jobs: int = -1,
        random_state: int = 42,
        verbosity: int = 0,
        early_stopping_rounds: Optional[int] = 50,
        **kwargs,
    ):
        # Initialize base class
        super().__init__()

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity
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
        eval_metric: Optional[str] = None,
        verbose: bool = False,
        **fit_kwargs,
    ):
        """
        Internal fit method for XGBoost model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            eval_set: List of (X, y) tuples for validation
            eval_metric: Metric to use for validation ('rmse', 'mae', 'mape', etc.)
            verbose: Whether to print training progress
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

        # Initialize model with early_stopping_rounds in constructor (XGBoost 1.6+)
        model_params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            **self.kwargs,
        }

        # Add early_stopping_rounds to constructor if eval_set provided
        if eval_set is not None and self.early_stopping_rounds is not None:
            model_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBRegressor(**model_params)

        # Fit model
        fit_params = fit_kwargs.copy()
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            if eval_metric is not None:
                fit_params["eval_metric"] = eval_metric
            fit_params["verbose"] = verbose

        self.model.fit(X, y, **fit_params)

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

        # Convert to DataFrame with feature names to avoid sklearn warning
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_"):
                X = pd.DataFrame(X, columns=self.feature_names_)

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

    def get_feature_importance(self, importance_type: str = "weight") -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')

        Returns:
            feature_importance: Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        # Get importance from model
        booster = self.model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)

        # Map feature IDs to names
        result = {}
        for i, name in enumerate(self.feature_names_):
            feat_id = f"f{i}"
            result[name] = importance_dict.get(feat_id, 0.0)

        return result

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
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params) -> "XGBoostForecaster":
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
            f"XGBoostForecaster(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})"
        )
