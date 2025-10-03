"""
Evaluation metrics for time series forecasting.

This module provides comprehensive metrics for evaluating forecasting performance
including accuracy metrics, directional accuracy, and time series specific measures.
"""

from typing import Dict

import numpy as np


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
