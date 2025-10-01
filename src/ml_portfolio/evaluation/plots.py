"""
Plotting utilities for model evaluation and forecast visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional, Dict, Any, Tuple
import warnings


def plot_forecast(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    train_size: Optional[int] = None,
    confidence_intervals: Optional[np.ndarray] = None,
    title: str = "Forecast vs Actual",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot forecast against actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Date index (optional)
        train_size: Size of training data to mark split
        confidence_intervals: Confidence intervals for predictions
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to pandas Series if needed
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    # Use dates if provided
    if dates is not None:
        y_true.index = dates[: len(y_true)]
        y_pred.index = dates[-len(y_pred) :]

    # Plot actual values
    ax.plot(y_true.index, y_true.values, label="Actual", color="blue", alpha=0.7)

    # Plot predictions
    ax.plot(y_pred.index, y_pred.values, label="Forecast", color="red", alpha=0.8)

    # Add confidence intervals if provided
    if confidence_intervals is not None:
        lower = confidence_intervals[:, 0]
        upper = confidence_intervals[:, 1]
        ax.fill_between(y_pred.index, lower, upper, alpha=0.2, color="red", label="Confidence Interval")

    # Mark train/test split
    if train_size is not None and dates is not None:
        split_date = dates[train_size]
        ax.axvline(x=split_date, color="green", linestyle="--", alpha=0.7, label="Train/Test Split")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot residual analysis.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Date index (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals over time
    if dates is not None:
        axes[0, 0].plot(dates[-len(residuals) :], residuals, alpha=0.7)
    else:
        axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("Residuals Distribution")
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")

    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # Residuals vs fitted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 1].set_title("Residuals vs Fitted")
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_learning_curves(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: Optional[List[int]] = None,
    metric_name: str = "Score",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot learning curves.

    Args:
        train_scores: Training scores
        val_scores: Validation scores
        train_sizes: Training set sizes (optional)
        metric_name: Name of the metric
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if train_sizes is None:
        train_sizes = range(1, len(train_scores) + 1)

    ax.plot(train_sizes, train_scores, "o-", label=f"Training {metric_name}", alpha=0.8)
    ax.plot(train_sizes, val_scores, "o-", label=f"Validation {metric_name}", alpha=0.8)

    ax.set_title("Learning Curves")
    ax.set_xlabel("Training Set Size" if train_sizes != range(1, len(train_scores) + 1) else "Epoch")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    top_n: Optional[int] = None,
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        feature_names: Names of features
        importances: Feature importance values
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show

    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]

    if top_n is not None:
        indices = indices[:top_n]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(sorted_features)), sorted_importances, alpha=0.8)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
        ax.text(
            bar.get_width() + 0.01 * max(sorted_importances),
            bar.get_y() + bar.get_height() / 2,
            f"{importance:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_cross_validation_scores(cv_results: Dict[str, np.ndarray], figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot cross-validation scores.

    Args:
        cv_results: Cross-validation results
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    metrics = [key.replace("test_", "").replace("train_", "") for key in cv_results.keys()]
    metrics = list(set(metrics))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics // 2, figsize[1]))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"

        if test_key in cv_results:
            axes[i].boxplot([cv_results[test_key]], labels=["Test"])

        if train_key in cv_results:
            data = [cv_results[train_key], cv_results[test_key]] if test_key in cv_results else [cv_results[train_key]]
            labels = ["Train", "Test"] if test_key in cv_results else ["Train"]
            axes[i].boxplot(data, labels=labels)

        axes[i].set_title(f"{metric.upper()} Scores")
        axes[i].set_ylabel("Score")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_timeseries_decomposition(
    data: pd.Series, model_type: str = "additive", figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot time series decomposition.

    Args:
        data: Time series data
        model_type: Decomposition model ('additive' or 'multiplicative')
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        decomposition = seasonal_decompose(data, model=model_type, extrapolate_trend="freq")

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Original
        axes[0].plot(data.index, data.values, alpha=0.8)
        axes[0].set_title("Original")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, alpha=0.8, color="orange")
        axes[1].set_title("Trend")
        axes[1].set_ylabel("Value")
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, alpha=0.8, color="green")
        axes[2].set_title("Seasonal")
        axes[2].set_ylabel("Value")
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, alpha=0.8, color="red")
        axes[3].set_title("Residual")
        axes[3].set_xlabel("Time")
        axes[3].set_ylabel("Value")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except ImportError:
        warnings.warn("statsmodels not available for decomposition plotting")
        return plt.figure()


def plot_correlation_matrix(data: pd.DataFrame, figsize: Tuple[int, int] = (10, 8), annot: bool = True) -> plt.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        data: DataFrame with features
        figsize: Figure size
        annot: Whether to annotate cells

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    correlation_matrix = data.corr()

    sns.heatmap(correlation_matrix, annot=annot, cmap="coolwarm", center=0, square=True, ax=ax)

    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_prediction_intervals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    lower_bound: Union[pd.Series, np.ndarray],
    upper_bound: Union[pd.Series, np.ndarray],
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    coverage_level: float = 0.95,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot prediction intervals.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        lower_bound: Lower prediction bound
        upper_bound: Upper prediction bound
        dates: Date index (optional)
        coverage_level: Expected coverage level
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if dates is None:
        dates = range(len(y_true))

    # Plot actual and predicted
    ax.plot(dates, y_true, label="Actual", color="blue", alpha=0.8)
    ax.plot(dates, y_pred, label="Predicted", color="red", alpha=0.8)

    # Plot prediction intervals
    ax.fill_between(
        dates, lower_bound, upper_bound, alpha=0.2, color="red", label=f"{coverage_level*100:.0f}% Prediction Interval"
    )

    # Calculate coverage
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

    ax.set_title(f"Prediction Intervals (Actual Coverage: {coverage*100:.1f}%)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
