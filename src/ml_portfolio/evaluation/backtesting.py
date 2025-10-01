"""
Backtesting and time series cross-validation utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator
import warnings


class TimeSeriesBacktester:
    """
    Backtesting framework for time series forecasting models.
    """

    def __init__(
        self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0, max_train_size: Optional[int] = None
    ):
        """
        Initialize TimeSeriesBacktester.

        Args:
            n_splits: Number of splits
            test_size: Size of test set for each split
            gap: Gap between train and test sets
            max_train_size: Maximum size of training set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap, max_train_size=max_train_size)

    def backtest(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: List[callable],
        refit: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Perform backtesting on a time series model.

        Args:
            model: Forecasting model
            X: Features
            y: Target values
            metrics: List of metric functions
            refit: Whether to refit model on each fold

        Returns:
            Dictionary of metric scores for each fold
        """
        results = {metric.__name__: [] for metric in metrics}

        for train_idx, test_idx in self.tscv.split(X):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit model
            if refit:
                model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            for metric in metrics:
                score = metric(y_test, y_pred)
                results[metric.__name__].append(score)

        return results

    def rolling_origin_backtest(
        self,
        model: BaseEstimator,
        data: pd.DataFrame,
        target_col: str,
        horizon: int = 1,
        window_size: Optional[int] = None,
        step_size: int = 1,
        metrics: List[callable] = None,
    ) -> Dict[str, Any]:
        """
        Perform rolling origin backtesting.

        Args:
            model: Forecasting model
            data: Time series data
            target_col: Name of target column
            horizon: Forecast horizon
            window_size: Size of training window (None for expanding)
            step_size: Step size for rolling window
            metrics: List of metric functions

        Returns:
            Dictionary containing predictions and metric scores
        """
        if metrics is None:
            from ..models.metrics import rmse, mae, mape

            metrics = [rmse, mae, mape]

        predictions = []
        actuals = []
        dates = []

        n = len(data)
        start_idx = window_size if window_size else int(0.3 * n)  # Default to 30% for initial training

        for i in range(start_idx, n - horizon + 1, step_size):
            # Define training window
            train_start = max(0, i - window_size) if window_size else 0
            train_end = i

            # Get training data
            train_data = data.iloc[train_start:train_end]
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]

            # Get test data
            test_idx = i + horizon - 1
            if test_idx >= n:
                break

            X_test = data.iloc[[test_idx]].drop(columns=[target_col])
            y_test = data.iloc[test_idx][target_col]

            # Fit and predict
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)[0]

                predictions.append(y_pred)
                actuals.append(y_test)
                dates.append(data.index[test_idx])
            except Exception as e:
                warnings.warn(f"Error in fold {i}: {e}")
                continue

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        metric_scores = {}
        for metric in metrics:
            try:
                score = metric(actuals, predictions)
                metric_scores[metric.__name__] = score
            except Exception as e:
                warnings.warn(f"Error calculating {metric.__name__}: {e}")
                metric_scores[metric.__name__] = np.nan

        return {"predictions": predictions, "actuals": actuals, "dates": dates, "metrics": metric_scores}


class WalkForwardAnalysis:
    """
    Walk-forward analysis for time series models.
    """

    def __init__(self, train_size: int, test_size: int, step_size: int = 1, anchored: bool = False):
        """
        Initialize WalkForwardAnalysis.

        Args:
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for walking forward
            anchored: Whether to use anchored analysis (expanding window)
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for walk-forward analysis.

        Args:
            X: Data array

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(X)
        splits = []

        start = 0
        while start + self.train_size + self.test_size <= n:
            if self.anchored:
                train_start = 0
            else:
                train_start = start

            train_end = start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))
            start += self.step_size

        return splits

    def analyze(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: List[callable],
    ) -> Dict[str, List[float]]:
        """
        Perform walk-forward analysis.

        Args:
            model: Forecasting model
            X: Features
            y: Target values
            metrics: List of metric functions

        Returns:
            Dictionary of metric scores for each window
        """
        results = {metric.__name__: [] for metric in metrics}
        splits = self.split(X)

        for train_idx, test_idx in splits:
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            for metric in metrics:
                score = metric(y_test, y_pred)
                results[metric.__name__].append(score)

        return results


def cross_validate_time_series(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: Union[int, TimeSeriesSplit] = 5,
    scoring: Union[str, callable, List[callable]] = None,
    return_train_score: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Cross-validate a time series model.

    Args:
        model: Forecasting model
        X: Features
        y: Target values
        cv: Cross-validation strategy
        scoring: Scoring function(s)
        return_train_score: Whether to return training scores

    Returns:
        Dictionary of cross-validation scores
    """
    if isinstance(cv, int):
        cv = TimeSeriesSplit(n_splits=cv)

    if scoring is None:
        from ..models.metrics import rmse

        scoring = [rmse]
    elif callable(scoring):
        scoring = [scoring]

    test_scores = {metric.__name__: [] for metric in scoring}
    if return_train_score:
        train_scores = {metric.__name__: [] for metric in scoring}

    for train_idx, test_idx in cv.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        if return_train_score:
            y_pred_train = model.predict(X_train)

        # Calculate metrics
        for metric in scoring:
            test_score = metric(y_test, y_pred_test)
            test_scores[metric.__name__].append(test_score)

            if return_train_score:
                train_score = metric(y_train, y_pred_train)
                train_scores[metric.__name__].append(train_score)

    # Convert to arrays
    for key in test_scores:
        test_scores[f"test_{key}"] = np.array(test_scores.pop(key))

    if return_train_score:
        for key in train_scores:
            test_scores[f"train_{key}"] = np.array(train_scores[key])

    return test_scores
