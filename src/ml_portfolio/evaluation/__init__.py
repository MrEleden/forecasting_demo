"""
Evaluation module for model assessment and visualization.
"""

from .backtesting import TimeSeriesBacktester, WalkForwardAnalysis, cross_validate_time_series

from .plots import (
    plot_forecast,
    plot_residuals,
    plot_learning_curves,
    plot_feature_importance,
    plot_cross_validation_scores,
    plot_forecast_decomposition,
    plot_correlation_matrix,
    plot_prediction_intervals,
)

__all__ = [
    # Backtesting
    "TimeSeriesBacktester",
    "WalkForwardAnalysis",
    "cross_validate_time_series",
    # Plotting
    "plot_forecast",
    "plot_residuals",
    "plot_learning_curves",
    "plot_feature_importance",
    "plot_cross_validation_scores",
    "plot_forecast_decomposition",
    "plot_correlation_matrix",
    "plot_prediction_intervals",
]
