"""
Evaluation metrics for time series forecasting.

This module provides comprehensive metrics for evaluating forecasting performance
including accuracy metrics, directional accuracy, and time series specific measures.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.inf

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        SMAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon

    return np.mean(numerator / denominator) * 100


def mase(y_true, y_pred, y_train, seasonal_period=1):
    """
    Mean Absolute Scaled Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonal_period: Seasonal period for naive forecast

    Returns:
        MASE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)

    # Calculate naive forecast error on training data
    if seasonal_period == 1:
        # Non-seasonal naive forecast (lag-1)
        naive_errors = np.abs(y_train[1:] - y_train[:-1])
    else:
        # Seasonal naive forecast
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])

    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf

    return mae(y_true, y_pred) / mae_naive


def directional_accuracy(y_true, y_pred):
    """
    Directional accuracy (percentage of correct directional predictions).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Directional accuracy as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) < 2:
        return np.nan

    # Calculate directional changes
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0

    # Count correct predictions
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)

    return (correct / total) * 100


def coverage_probability(y_true, y_lower, y_upper):
    """
    Coverage probability for prediction intervals.

    Args:
        y_true: True values
        y_lower: Lower prediction bounds
        y_upper: Upper prediction bounds

    Returns:
        Coverage probability as percentage
    """
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)

    # Check if true values fall within prediction intervals
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)

    return np.mean(within_interval) * 100


def mean_interval_width(y_lower, y_upper):
    """
    Mean width of prediction intervals.

    Args:
        y_lower: Lower prediction bounds
        y_upper: Upper prediction bounds

    Returns:
        Mean interval width
    """
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)

    return np.mean(y_upper - y_lower)


def evaluate_forecast(y_true, y_pred, y_train=None, seasonal_period=1, metrics=None) -> Dict[str, float]:
    """
    Comprehensive forecast evaluation with multiple metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data (for MASE calculation)
        seasonal_period: Seasonal period for MASE
        metrics: List of metrics to compute (if None, compute all)

    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape", "smape", "directional_accuracy"]

    results = {}

    # Basic accuracy metrics
    if "mae" in metrics:
        results["mae"] = mae(y_true, y_pred)

    if "rmse" in metrics:
        results["rmse"] = rmse(y_true, y_pred)

    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_true, y_pred)

    # Percentage errors
    if "mape" in metrics:
        results["mape"] = mape(y_true, y_pred)

    if "smape" in metrics:
        results["smape"] = smape(y_true, y_pred)

    # Scaled errors
    if "mase" in metrics and y_train is not None:
        results["mase"] = mase(y_true, y_pred, y_train, seasonal_period)

    # Directional accuracy
    if "directional_accuracy" in metrics:
        results["directional_accuracy"] = directional_accuracy(y_true, y_pred)

    return results


class ForecastEvaluator:
    """
    Class for comprehensive forecast evaluation and comparison.
    """

    def __init__(self, seasonal_period=1):
        """
        Initialize ForecastEvaluator.

        Args:
            seasonal_period: Seasonal period for MASE calculation
        """
        self.seasonal_period = seasonal_period
        self.results = {}

    def add_forecast(self, name: str, y_true, y_pred, y_train=None):
        """
        Add a forecast to evaluate.

        Args:
            name: Name of the forecast method
            y_true: True values
            y_pred: Predicted values
            y_train: Training data
        """
        metrics = evaluate_forecast(y_true, y_pred, y_train, self.seasonal_period)
        self.results[name] = metrics

    def get_results(self) -> pd.DataFrame:
        """
        Get evaluation results as DataFrame.

        Returns:
            DataFrame with methods as rows and metrics as columns
        """
        return pd.DataFrame(self.results).T

    def get_best_method(self, metric="mae") -> str:
        """
        Get the best performing method for a given metric.

        Args:
            metric: Metric name to optimize

        Returns:
            Name of the best method
        """
        df = self.get_results()

        if metric in df.columns:
            # For most metrics, lower is better
            if metric in ["mae", "rmse", "mse", "mape", "smape", "mase"]:
                return df[metric].idxmin()
            else:
                # For directional accuracy, higher is better
                return df[metric].idxmax()
        else:
            raise ValueError(f"Metric '{metric}' not found in results")

    def compare_methods(self, baseline_method: str) -> pd.DataFrame:
        """
        Compare all methods against a baseline.

        Args:
            baseline_method: Name of the baseline method

        Returns:
            DataFrame showing relative performance vs baseline
        """
        df = self.get_results()
        baseline = df.loc[baseline_method]

        # Calculate relative performance
        comparison = df.div(baseline) - 1
        comparison = comparison * 100  # Convert to percentage

        return comparison
