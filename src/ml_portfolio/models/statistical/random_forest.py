"""
Random Forest forecasting model wrapper.

Provides consistent interface with other statistical models.
Random Forest is an ensemble of decision trees.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor

from ..base import StatisticalForecaster


class RandomForestForecaster(StatisticalForecaster):
    """
    Random Forest forecasting model with consistent interface.

    Random Forest is an ensemble method that:
    - Builds multiple decision trees
    - Uses bootstrap aggregating (bagging)
    - Reduces overfitting compared to single trees
    - Provides feature importance

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the tree (None for unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at leaf node
        max_features: Number of features to consider for best split
        bootstrap: Whether to use bootstrap samples
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed for reproducibility
        verbose: Verbosity level
        **kwargs: Additional RandomForest parameters

    Example:
        >>> from ml_portfolio.models.statistical.random_forest import RandomForestForecaster
        >>> model = RandomForestForecaster(n_estimators=200, max_depth=10)
        >>> model.fit(train_loader, val_loader)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 0,
        **kwargs,
    ):
        # Initialize base class
        super().__init__()

        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        # Initialize sklearn model
        self.model = SKRandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.kwargs,
        )

    def _fit(self, X, y):
        """
        Internal fit method (implements abstract method from StatisticalForecaster).

        Args:
            X: Training features (numpy array)
            y: Training targets (numpy array)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Generate predictions.

        Args:
            X: Features (numpy array or dataloader)

        Returns:
            numpy array of predictions
        """
        # Handle dataloader input
        if hasattr(X, "__iter__") and not isinstance(X, np.ndarray):
            try:
                X, _ = next(iter(X))
            except (TypeError, ValueError):
                pass

        return self.model.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (sklearn compatibility)."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            **self.kwargs,
        }

    def set_params(self, **params) -> "RandomForestForecaster":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Reinitialize sklearn model with new params
        self.model = SKRandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.kwargs,
        )

        return self

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from trained model."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_
