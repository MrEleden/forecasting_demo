"""
Evaluation metrics for time series forecasting.

This module provides comprehensive metrics for evaluating forecasting performance
including accuracy metrics, directional accuracy, and time series specific measures.
"""

from typing import Dict

import numpy as np

# ============================================================================
# Core Metric Functions (used by classes and directly importable)
# ============================================================================


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (as percentage)
    """
    mask = np.abs(y_true) > epsilon
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        SMAPE value (as percentage)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > epsilon
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional accuracy - percentage of correct direction predictions.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Directional accuracy (0-100%)
    """
    if len(y_true) < 2:
        return np.nan

    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return float(np.mean(true_direction == pred_direction) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 1) -> float:
    """
    Mean Absolute Scaled Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonal_period: Seasonal period for naive forecast

    Returns:
        MASE value
    """
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(np.diff(y_train, n=seasonal_period)))

    if mae_naive == 0:
        return np.nan

    return float(mae_forecast / mae_naive)


# Hydra-compatible metric classes
class BaseMetric:
    """Base class for Hydra-instantiable metrics."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__.lower()

    def __call__(self, y_true, y_pred, **kwargs):
        raise NotImplementedError


class MAEMetric(BaseMetric):
    """Mean Absolute Error metric."""

    def __call__(self, y_true, y_pred, **kwargs):
        return mae(y_true, y_pred)


class RMSEMetric(BaseMetric):
    """Root Mean Squared Error metric."""

    def __call__(self, y_true, y_pred, **kwargs):
        return rmse(y_true, y_pred)


class MAPEMetric(BaseMetric):
    """Mean Absolute Percentage Error metric."""

    def __init__(self, name: str = None, epsilon: float = 1e-8):
        super().__init__(name)
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred, **kwargs):
        return mape(y_true, y_pred, self.epsilon)


class SMAPEMetric(BaseMetric):
    """Symmetric Mean Absolute Percentage Error metric."""

    def __init__(self, name: str = None, epsilon: float = 1e-8):
        super().__init__(name)
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred, **kwargs):
        return smape(y_true, y_pred, self.epsilon)


class DirectionalAccuracyMetric(BaseMetric):
    """Directional accuracy metric."""

    def __call__(self, y_true, y_pred, **kwargs):
        return directional_accuracy(y_true, y_pred)


class MASEMetric(BaseMetric):
    """Mean Absolute Scaled Error metric."""

    def __init__(self, name: str = None, seasonal_period: int = 1):
        super().__init__(name)
        self.seasonal_period = seasonal_period

    def __call__(self, y_true, y_pred, y_train=None, **kwargs):
        if y_train is None:
            raise ValueError("MASE requires training data (y_train)")
        return mase(y_true, y_pred, y_train, self.seasonal_period)


class MetricCollection:
    """Collection of metrics for comprehensive evaluation."""

    def __init__(self, metrics: Dict[str, BaseMetric]):
        """
        Initialize metric collection.

        Args:
            metrics: Dictionary of metric name to metric instance
        """
        self.metrics = metrics

    def compute(self, y_true, y_pred, **kwargs) -> Dict[str, float]:
        """
        Compute all metrics in the collection.

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments passed to metrics

        Returns:
            Dictionary of metric names to values
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric(y_true, y_pred, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to compute {name}: {e}")
                results[name] = np.nan
        return results

    def get_primary_metric(self, metric_name: str = "mape") -> float:
        """Get value of primary metric for optimization."""
        if hasattr(self, "_last_results") and metric_name in self._last_results:
            return self._last_results[metric_name]
        return np.nan

    def compute_and_store(self, y_true, y_pred, **kwargs) -> Dict[str, float]:
        """Compute metrics and store for later access."""
        self._last_results = self.compute(y_true, y_pred, **kwargs)
        return self._last_results
