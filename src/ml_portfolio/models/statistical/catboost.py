"""
CatBoost forecasting model wrapper.

Provides sklearn-compatible interface for CatBoost regressor.
Excellent for handling categorical features with minimal tuning.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostForecaster(StatisticalForecaster):
    """
    CatBoost forecasting model with sklearn interface.

    Gradient boosting with automatic categorical feature handling:
    - Ordered boosting for better generalization
    - Native categorical feature support
    - Less hyperparameter tuning required
    - Robust to overfitting

    Often outperforms XGBoost/LightGBM on retail data with many categories.

    Args:
        iterations: Number of boosting iterations
        learning_rate: Boosting learning rate
        depth: Depth of the tree
        l2_leaf_reg: L2 regularization coefficient
        bagging_temperature: Randomness parameter for bagging
        random_strength: Randomness parameter for score calculation
        subsample: Sample rate for bagging
        thread_count: Number of threads (-1 for all cores)
        random_seed: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=info)
        early_stopping_rounds: Early stopping rounds (requires validation set)
        loss_function: Loss function to minimize
        **kwargs: Additional CatBoost parameters

    Example:
        >>> from ml_portfolio.models.statistical.catboost import CatBoostForecaster
        >>> model = CatBoostForecaster(iterations=500, learning_rate=0.05)
        >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 8,
        l2_leaf_reg: float = 3.0,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
        subsample: float = 0.8,
        thread_count: int = -1,
        random_seed: int = 42,
        verbose: int = 0,
        early_stopping_rounds: Optional[int] = 50,
        loss_function: str = "RMSE",
        **kwargs,
    ):
        # Initialize base class
        super().__init__()

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.random_strength = random_strength
        self.subsample = subsample
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.loss_function = loss_function
        self.kwargs = kwargs

        # Will be initialized in fit()
        self.model = None
        self.feature_names_ = None
        self.feature_importances_ = None

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        cat_features: Optional[list] = None,
        **fit_kwargs,
    ):
        """
        Internal fit method for CatBoost model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            eval_set: Tuple of (X_val, y_val) for validation
            cat_features: List of categorical feature indices or names
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
        self.model = cb.CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature,
            random_strength=self.random_strength,
            subsample=self.subsample,
            thread_count=self.thread_count,
            random_seed=self.random_seed,
            verbose=self.verbose,
            loss_function=self.loss_function,
            **self.kwargs,
        )

        # Prepare fit parameters
        fit_params = fit_kwargs.copy()

        if eval_set is not None:
            # CatBoost expects Pool objects or eval_set as tuple
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, (pd.DataFrame, pd.Series)):
                y_val = y_val.values
            if y_val.ndim > 1:
                y_val = y_val.ravel()

            fit_params["eval_set"] = (X_val, y_val)

            if self.early_stopping_rounds is not None:
                fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        if cat_features is not None:
            fit_params["cat_features"] = cat_features

        # Fit model
        self.model.fit(X, y, **fit_params)

        # Store feature importances
        self.feature_importances_ = self.model.get_feature_importance()

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

    def get_feature_importance(self, importance_type: str = "FeatureImportance") -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('FeatureImportance', 'PredictionValuesChange', 'LossFunctionChange')

        Returns:
            feature_importance: Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importances = self.model.get_feature_importance(type=importance_type)

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
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "bagging_temperature": self.bagging_temperature,
            "random_strength": self.random_strength,
            "subsample": self.subsample,
            "thread_count": self.thread_count,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "early_stopping_rounds": self.early_stopping_rounds,
            "loss_function": self.loss_function,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params) -> "CatBoostForecaster":
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
            f"CatBoostForecaster(iterations={self.iterations}, "
            f"learning_rate={self.learning_rate}, depth={self.depth})"
        )
