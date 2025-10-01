"""
Time series utilities for windowing, lag features, and calendar features.

This module provides specialized functions for time series data preparation
including seasonal decomposition, lag creation, and temporal feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
from datetime import datetime, timedelta


def create_windows(
    data: np.ndarray, window_size: int, stride: int = 1, forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series forecasting.

    Args:
        data: Input time series data
        window_size: Size of the input window (lookback)
        stride: Step size between windows
        forecast_horizon: Number of steps to predict ahead

    Returns:
        Tuple of (X, y) where X is windowed input and y is targets
    """
    X, y = [], []

    for i in range(0, len(data) - window_size - forecast_horizon + 1, stride):
        # Input window
        window = data[i : i + window_size]
        # Target (next forecast_horizon values)
        target = data[i + window_size : i + window_size + forecast_horizon]

        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)


def create_lag_features(data: pd.DataFrame, target_column: str, lags: List[int], suffix: str = "_lag") -> pd.DataFrame:
    """
    Create lag features for time series data.

    Args:
        data: Input DataFrame
        target_column: Name of the target column
        lags: List of lag periods to create
        suffix: Suffix for lag column names

    Returns:
        DataFrame with added lag features
    """
    result = data.copy()

    for lag in lags:
        lag_col_name = f"{target_column}{suffix}_{lag}"
        result[lag_col_name] = result[target_column].shift(lag)

    return result


def create_rolling_features(
    data: pd.DataFrame, target_column: str, windows: List[int], functions: List[str] = ["mean", "std", "min", "max"]
) -> pd.DataFrame:
    """
    Create rolling window statistical features.

    Args:
        data: Input DataFrame
        target_column: Name of the target column
        windows: List of window sizes for rolling statistics
        functions: List of statistical functions to apply

    Returns:
        DataFrame with added rolling features
    """
    result = data.copy()

    for window in windows:
        for func in functions:
            col_name = f"{target_column}_rolling_{window}_{func}"

            if func == "mean":
                result[col_name] = result[target_column].rolling(window=window).mean()
            elif func == "std":
                result[col_name] = result[target_column].rolling(window=window).std()
            elif func == "min":
                result[col_name] = result[target_column].rolling(window=window).min()
            elif func == "max":
                result[col_name] = result[target_column].rolling(window=window).max()

    return result


def create_calendar_features(
    data: pd.DataFrame, date_column: str, features: List[str] = ["hour", "day", "month", "quarter", "weekday"]
) -> pd.DataFrame:
    """
    Create calendar-based features from datetime column.

    Args:
        data: Input DataFrame
        date_column: Name of the datetime column
        features: List of calendar features to create

    Returns:
        DataFrame with added calendar features
    """
    result = data.copy()

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column])

    dt = result[date_column].dt

    if "hour" in features:
        result["hour"] = dt.hour
    if "day" in features:
        result["day"] = dt.day
    if "month" in features:
        result["month"] = dt.month
    if "quarter" in features:
        result["quarter"] = dt.quarter
    if "weekday" in features:
        result["weekday"] = dt.weekday
    if "dayofyear" in features:
        result["dayofyear"] = dt.dayofyear
    if "week" in features:
        result["week"] = dt.isocalendar().week

    return result


def create_seasonal_features(
    data: pd.DataFrame, date_column: str, seasonal_periods: List[int] = [24, 168, 8760]  # hour, week, year in hours
) -> pd.DataFrame:
    """
    Create seasonal/cyclical features using sine and cosine transformations.

    Args:
        data: Input DataFrame
        date_column: Name of the datetime column
        seasonal_periods: List of seasonal periods

    Returns:
        DataFrame with added seasonal features
    """
    result = data.copy()

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
        result[date_column] = pd.to_datetime(result[date_column])

    # Create time index (hours since start)
    start_time = result[date_column].min()
    result["hours_since_start"] = (result[date_column] - start_time).dt.total_seconds() / 3600

    for period in seasonal_periods:
        # Create sine and cosine features for each seasonal period
        result[f"sin_{period}"] = np.sin(2 * np.pi * result["hours_since_start"] / period)
        result[f"cos_{period}"] = np.cos(2 * np.pi * result["hours_since_start"] / period)

    # Remove helper column
    result = result.drop(columns=["hours_since_start"])

    return result


def train_test_split_time_series(
    data: pd.DataFrame, test_size: float = 0.2, date_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data maintaining temporal order.

    Args:
        data: Input DataFrame
        test_size: Fraction of data to use for testing
        date_column: Name of date column (for sorting)

    Returns:
        Tuple of (train_data, test_data)
    """
    # Sort by date if date column is provided
    if date_column and date_column in data.columns:
        data = data.sort_values(date_column)

    # Calculate split point
    split_point = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()

    return train_data, test_data


class TimeSeriesCV:
    """
    Time series cross-validation with expanding or sliding window.
    """

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0, expanding_window: bool = True):
        """
        Initialize TimeSeriesCV.

        Args:
            n_splits: Number of splits
            test_size: Size of test set for each split
            gap: Gap between train and test sets
            expanding_window: If True, use expanding window; if False, use sliding window
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for time series cross-validation.

        Args:
            X: Input data

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        splits = []

        for i in range(self.n_splits):
            # Calculate test indices
            test_end = n_samples - i * test_size
            test_start = test_end - test_size

            # Calculate train indices
            train_end = test_start - self.gap

            if self.expanding_window:
                train_start = 0
            else:
                # Sliding window: use same size as test set
                train_start = max(0, train_end - test_size)

            if train_start >= train_end or test_start >= test_end:
                break

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

        return splits[::-1]  # Return in chronological order
