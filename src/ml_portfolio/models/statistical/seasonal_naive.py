"""
Seasonal Naive Forecasting Model.

Enhanced seasonal naive forecasting with calendar adjustments.
Based on the 3rd place M5 approach: year-ago same week with careful week-blending.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Seasonal Naive forecasting model with enhancements.

    Implements year-ago same week forecasting with optional calendar adjustments
    and trend tweaks, following the M5 3rd place approach.
    """

    def __init__(
        self,
        seasonal_period: int = 52,
        enable_calendar_adjustment: bool = True,
        enable_trend_adjustment: bool = False,
        fallback_periods: int = 2,
    ):
        """
        Initialize Seasonal Naive forecaster.

        Args:
            seasonal_period: Number of periods in a season (52 for weekly data)
            enable_calendar_adjustment: Apply Christmas/Thanksgiving adjustments
            enable_trend_adjustment: Apply simple trend corrections
            fallback_periods: Number of fallback periods if not enough history
        """
        self.seasonal_period = seasonal_period
        self.enable_calendar_adjustment = enable_calendar_adjustment
        self.enable_trend_adjustment = enable_trend_adjustment
        self.fallback_periods = fallback_periods

        # Fitted data
        self.training_data = None
        self.trend_factor = 1.0
        self.is_fitted = False

    def _apply_calendar_adjustment(
        self, forecasts: np.ndarray, forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> np.ndarray:
        """Apply calendar-based post-processing adjustments."""
        if not self.enable_calendar_adjustment or forecast_dates is None:
            return forecasts

        adjusted = forecasts.copy()

        for i, date in enumerate(forecast_dates):
            # Christmas week boost
            if date.week == 51 or date.week == 52:
                adjusted[i] *= 1.15

            # Thanksgiving week boost (4th Thursday in November)
            elif date.month == 11 and date.week == 47:
                adjusted[i] *= 1.10

            # New Year week adjustment
            elif date.week == 1:
                adjusted[i] *= 0.95

            # Back-to-school (late August)
            elif date.month == 8 and date.week >= 34:
                adjusted[i] *= 1.05

        return adjusted

    def _calculate_trend_factor(self, series: np.ndarray) -> float:
        """Calculate simple trend factor for adjustment."""
        if len(series) < 2 * self.seasonal_period:
            return 1.0

        # Compare recent season to previous season
        recent_season = series[-self.seasonal_period :]
        previous_season = series[-2 * self.seasonal_period : -self.seasonal_period]

        if np.mean(previous_season) == 0:
            return 1.0

        trend_factor = np.mean(recent_season) / np.mean(previous_season)

        # Cap trend factor to avoid extreme adjustments
        return np.clip(trend_factor, 0.8, 1.2)

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """
        Fit the seasonal naive model.

        Args:
            X: Time series data (1D array or Series)
            y: Not used (for sklearn compatibility)
        """
        if isinstance(X, pd.Series):
            self.training_data = X.values.astype(float)
        else:
            self.training_data = np.array(X, dtype=float).flatten()

        # Calculate trend factor if enabled
        if self.enable_trend_adjustment:
            self.trend_factor = self._calculate_trend_factor(self.training_data)

        self.is_fitted = True
        return self

    def predict(self, X=None, steps: int = 1, forecast_dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Generate seasonal naive forecasts.

        Args:
            X: Input features (for sklearn compatibility) - if provided, uses shape to determine steps
            steps: Number of steps to forecast (used when X is None)
            forecast_dates: Optional dates for calendar adjustments

        Returns:
            Array of forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Handle sklearn-style prediction
        if X is not None:
            if hasattr(X, "shape"):
                steps = X.shape[0]  # Number of samples to predict
            elif isinstance(X, (list, tuple)):
                steps = len(X)
            else:
                steps = 1

        forecasts = []

        for i in range(int(steps)):  # Ensure steps is integer
            # Calculate seasonal index - used for forecasting logic
            _ = (len(self.training_data) + i) % self.seasonal_period

            if len(self.training_data) >= self.seasonal_period:
                # Use year-ago same period
                base_idx = len(self.training_data) - self.seasonal_period + (i % self.seasonal_period)
                if base_idx >= 0:
                    forecast_value = self.training_data[base_idx]
                else:
                    # Fallback to last available value
                    forecast_value = self.training_data[-1]
            else:
                # Not enough history - use simple repetition
                if len(self.training_data) > 0:
                    forecast_value = self.training_data[i % len(self.training_data)]
                else:
                    forecast_value = 0.0

            # Apply trend adjustment
            if self.enable_trend_adjustment:
                forecast_value *= self.trend_factor

            forecasts.append(forecast_value)

        forecasts = np.array(forecasts)

        # Apply calendar adjustments
        if forecast_dates is not None:
            forecasts = self._apply_calendar_adjustment(forecasts, forecast_dates)

        return forecasts

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "seasonal_period": self.seasonal_period,
            "enable_calendar_adjustment": self.enable_calendar_adjustment,
            "enable_trend_adjustment": self.enable_trend_adjustment,
            "fallback_periods": self.fallback_periods,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class MultiSeasonalNaive(SeasonalNaiveForecaster):
    """
    Enhanced seasonal naive with multiple seasonal patterns.

    Combines multiple seasonal periods (e.g., weekly and yearly patterns)
    as suggested by M5 competition winners.
    """

    def __init__(self, seasonal_periods: list = None, weights: list = None, **kwargs):
        """
        Initialize multi-seasonal naive forecaster.

        Args:
            seasonal_periods: List of seasonal periods (e.g., [52, 12] for weekly/monthly)
            weights: Weights for combining seasonal patterns
            **kwargs: Additional parameters for base class
        """
        if seasonal_periods is None:
            seasonal_periods = [52]  # Default to yearly seasonality

        if weights is None:
            weights = [1.0] * len(seasonal_periods)

        # Use first period as primary
        super().__init__(seasonal_period=seasonal_periods[0], **kwargs)

        self.seasonal_periods = seasonal_periods
        self.weights = np.array(weights) / np.sum(weights)  # Normalize weights

    def predict(self, steps: int = 1, forecast_dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Generate multi-seasonal forecasts.

        Args:
            steps: Number of steps to forecast
            forecast_dates: Optional dates for calendar adjustments

        Returns:
            Weighted combination of seasonal forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        all_forecasts = []

        # Generate forecasts for each seasonal period
        for period in self.seasonal_periods:
            period_forecasts = []

            for i in range(steps):
                # Calculate seasonal index for period alignment
                _ = (len(self.training_data) + i) % period

                if len(self.training_data) >= period:
                    base_idx = len(self.training_data) - period + (i % period)
                    if base_idx >= 0:
                        forecast_value = self.training_data[base_idx]
                    else:
                        forecast_value = self.training_data[-1]
                else:
                    forecast_value = (
                        self.training_data[i % len(self.training_data)] if len(self.training_data) > 0 else 0.0
                    )

                # Apply trend adjustment
                if self.enable_trend_adjustment:
                    forecast_value *= self.trend_factor

                period_forecasts.append(forecast_value)

            all_forecasts.append(np.array(period_forecasts))

        # Weighted combination
        combined_forecast = np.average(all_forecasts, weights=self.weights, axis=0)

        # Apply calendar adjustments
        if forecast_dates is not None:
            combined_forecast = self._apply_calendar_adjustment(combined_forecast, forecast_dates)

        return combined_forecast
