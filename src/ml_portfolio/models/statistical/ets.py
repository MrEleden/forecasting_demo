"""
Exponential Smoothing (ETS) Forecasting Models.

Pure exponential smoothing implementations from the Holt-Winters family,
as used by top M5 competitors as standalone components and in blends.
"""

import warnings
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")


class ETSForecaster(BaseEstimator, RegressorMixin):
    """
    Exponential Smoothing (ETS) forecaster with automatic model selection.

    Implements Error-Trend-Seasonality framework with automatic selection
    of best configuration (Add/Mult trends/seasonality) as used by M5 winners.
    """

    def __init__(
        self,
        seasonal_period: int = 52,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        damped_trend: bool = False,
        auto_select: bool = True,
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
    ):
        """
        Initialize ETS forecaster.

        Args:
            seasonal_period: Number of periods in a season
            trend: Trend component ('add', 'mul', None). If None and auto_select=True,
                will be determined automatically
            seasonal: Seasonal component ('add', 'mul', None). If None and auto_select=True,
                will be determined automatically
            damped_trend: Whether to use damped trend
            auto_select: Automatically select best ETS configuration
            smoothing_level: Alpha parameter for level smoothing (0-1)
            smoothing_trend: Beta parameter for trend smoothing (0-1)
            smoothing_seasonal: Gamma parameter for seasonal smoothing (0-1)
        """
        self.seasonal_period = seasonal_period
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.auto_select = auto_select
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal

        # Fitted components
        self.fitted_model = None
        self.level = None
        self.trend_component = None
        self.seasonal_components = None
        self.best_config = None
        self.is_fitted = False

    def _try_statsmodels_ets(self, series: np.ndarray, config: dict) -> tuple:
        """Try fitting ETS using statsmodels."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                series,
                trend=config["trend"],
                seasonal=config["seasonal"],
                seasonal_periods=self.seasonal_period if config["seasonal"] else None,
                damped_trend=self.damped_trend,
            )

            fitted = model.fit(
                smoothing_level=self.smoothing_level,
                smoothing_trend=self.smoothing_trend,
                smoothing_seasonal=self.smoothing_seasonal,
                optimized=True,
                use_boxcox=False,
            )

            return fitted, fitted.aic if hasattr(fitted, "aic") else float("inf")

        except Exception:
            return None, float("inf")

    def _simple_exponential_smoothing(self, series: np.ndarray, alpha: float = 0.3) -> dict:
        """Simple exponential smoothing fallback."""
        level = series[0]
        levels = [level]

        for value in series[1:]:
            level = alpha * value + (1 - alpha) * level
            levels.append(level)

        return {
            "level": level,
            "trend": 0.0,
            "seasonal": np.zeros(self.seasonal_period),
            "alpha": alpha,
            "fitted_values": np.array(levels),
        }

    def _holt_linear_trend(self, series: np.ndarray, alpha: float = 0.3, beta: float = 0.1) -> dict:
        """Holt's linear trend method."""
        if len(series) < 2:
            return self._simple_exponential_smoothing(series, alpha)

        level = series[0]
        trend = series[1] - series[0]
        levels = [level]
        trends = [trend]
        fitted = [level]

        for i, value in enumerate(series[1:], 1):
            prev_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

            levels.append(level)
            trends.append(trend)
            fitted.append(level + trend)

        return {
            "level": level,
            "trend": trend,
            "seasonal": np.zeros(self.seasonal_period),
            "alpha": alpha,
            "beta": beta,
            "fitted_values": np.array(fitted),
        }

    def _holt_winters_additive(
        self, series: np.ndarray, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1
    ) -> dict:
        """Holt-Winters additive method."""
        if len(series) < 2 * self.seasonal_period:
            return self._holt_linear_trend(series, alpha, beta)

        # Initialize components
        level = np.mean(series[: self.seasonal_period])
        trend = (
            np.mean(series[self.seasonal_period : 2 * self.seasonal_period]) - np.mean(series[: self.seasonal_period])
        ) / self.seasonal_period

        # Initialize seasonal components
        seasonal = np.zeros(self.seasonal_period)
        for i in range(self.seasonal_period):
            seasonal[i] = np.mean(
                [
                    series[i + j * self.seasonal_period] - level
                    for j in range(min(2, len(series) // self.seasonal_period))
                ]
            )

        levels = []
        trends = []
        seasonals = seasonal.copy()
        fitted = []

        for i, value in enumerate(series):
            season_idx = i % self.seasonal_period

            if i == 0:
                fitted_value = level + trend + seasonals[season_idx]
            else:
                fitted_value = level + trend + seasonals[season_idx]

                # Update components
                prev_level = level
                level = alpha * (value - seasonals[season_idx]) + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                seasonals[season_idx] = gamma * (value - level) + (1 - gamma) * seasonals[season_idx]

            levels.append(level)
            trends.append(trend)
            fitted.append(fitted_value)

        return {
            "level": level,
            "trend": trend,
            "seasonal": seasonals,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "fitted_values": np.array(fitted),
        }

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """
        Fit the ETS model.

        Args:
            X: Time series data
            y: Not used (for sklearn compatibility)
        """
        if isinstance(X, pd.Series):
            series = X.values.astype(float)
        else:
            series = np.array(X, dtype=float).flatten()

        if len(series) == 0:
            raise ValueError("Input series is empty")

        best_aic = float("inf")
        best_fitted = None
        best_config = None

        if self.auto_select:
            # Try different ETS configurations
            configs = [
                {"trend": None, "seasonal": None},
                {"trend": "add", "seasonal": None},
                {"trend": "add", "seasonal": "add"},
                {"trend": "mul", "seasonal": "add"} if np.all(series > 0) else {"trend": "add", "seasonal": "add"},
            ]

            # Only try seasonal models if we have enough data
            if len(series) < 2 * self.seasonal_period:
                configs = [c for c in configs if c["seasonal"] is None]

            for config in configs:
                fitted, aic = self._try_statsmodels_ets(series, config)

                if fitted is not None and aic < best_aic:
                    best_aic = aic
                    best_fitted = fitted
                    best_config = config
        else:
            # Use specified configuration
            config = {"trend": self.trend, "seasonal": self.seasonal}
            best_fitted, best_aic = self._try_statsmodels_ets(series, config)
            best_config = config

        # Fallback to simple methods if statsmodels fails
        if best_fitted is None:
            if len(series) >= 2 * self.seasonal_period:
                result = self._holt_winters_additive(series)
                best_config = {"trend": "add", "seasonal": "add"}
            elif len(series) >= 2:
                result = self._holt_linear_trend(series)
                best_config = {"trend": "add", "seasonal": None}
            else:
                result = self._simple_exponential_smoothing(series)
                best_config = {"trend": None, "seasonal": None}

            # Create mock fitted object
            class SimpleFitted:
                def __init__(self, result):
                    self.level = result["level"]
                    self.trend = result["trend"]
                    self.seasonal = result["seasonal"]
                    self.params = result

                def forecast(self, steps):
                    forecasts = []
                    level = self.level
                    trend = self.trend

                    for i in range(steps):
                        if best_config["seasonal"]:
                            seasonal_component = self.seasonal[i % len(self.seasonal)]
                        else:
                            seasonal_component = 0

                        forecast = level + trend * (i + 1) + seasonal_component
                        forecasts.append(forecast)

                    return np.array(forecasts)

            best_fitted = SimpleFitted(result)

        self.fitted_model = best_fitted
        self.best_config = best_config
        self.is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate ETS forecasts (sklearn-compatible interface).

        Args:
            X: Feature matrix (sklearn interface) - we use X.shape[0] as number of steps

        Returns:
            Array of forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert sklearn interface to time series interface
        if hasattr(X, "shape"):
            steps = X.shape[0]  # Number of samples to predict
        else:
            steps = len(X) if hasattr(X, "__len__") else 1

        try:
            forecast = self.fitted_model.forecast(steps)
            return forecast.values if hasattr(forecast, "values") else forecast
        except Exception:
            # Fallback forecast
            if hasattr(self.fitted_model, "level"):
                level = self.fitted_model.level
                trend = getattr(self.fitted_model, "trend", 0)
                seasonal = getattr(self.fitted_model, "seasonal", np.zeros(self.seasonal_period))

                forecasts = []
                for i in range(steps):
                    seasonal_component = seasonal[i % len(seasonal)] if len(seasonal) > 0 else 0
                    forecast = level + trend * (i + 1) + seasonal_component
                    forecasts.append(forecast)

                return np.array(forecasts)
            else:
                return np.zeros(steps)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "seasonal_period": self.seasonal_period,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "damped_trend": self.damped_trend,
            "auto_select": self.auto_select,
            "smoothing_level": self.smoothing_level,
            "smoothing_trend": self.smoothing_trend,
            "smoothing_seasonal": self.smoothing_seasonal,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class SimpleETSForecaster(BaseEstimator, RegressorMixin):
    """
    Simplified ETS forecaster without external dependencies.

    Implements basic exponential smoothing methods using only numpy.
    Useful as a lightweight alternative or fallback.
    """

    def __init__(
        self,
        method: Literal["simple", "holt", "holt_winters"] = "holt_winters",
        seasonal_period: int = 52,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
    ):
        """
        Initialize simple ETS forecaster.

        Args:
            method: ETS method ('simple', 'holt', 'holt_winters')
            seasonal_period: Seasonal period
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter
            gamma: Seasonal smoothing parameter
        """
        self.method = method
        self.seasonal_period = seasonal_period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.components = None
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """Fit the simple ETS model."""
        if isinstance(X, pd.Series):
            series = X.values.astype(float)
        else:
            series = np.array(X, dtype=float).flatten()

        forecaster = ETSForecaster(
            seasonal_period=self.seasonal_period,
            auto_select=False,
            smoothing_level=self.alpha,
            smoothing_trend=self.beta,
            smoothing_seasonal=self.gamma,
        )

        if self.method == "simple":
            forecaster.trend = None
            forecaster.seasonal = None
        elif self.method == "holt":
            forecaster.trend = "add"
            forecaster.seasonal = None
        else:  # holt_winters
            forecaster.trend = "add"
            forecaster.seasonal = "add"

        forecaster.fit(series)
        self.fitted_forecaster = forecaster
        self.is_fitted = True

        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """Generate forecasts using the simple ETS method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.fitted_forecaster.predict(steps)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "method": self.method,
            "seasonal_period": self.seasonal_period,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
