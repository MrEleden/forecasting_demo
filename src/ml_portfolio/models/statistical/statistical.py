"""
Statistical forecasting models (ARIMA, Prophet, etc.) with sklearn compatibility.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
import warnings


class ARIMAWrapper(BaseEstimator, RegressorMixin):
    """
    ARIMA model wrapper compatible with sklearn.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        concentrate_scale: bool = False,
    ):
        """
        Initialize ARIMA wrapper.

        Args:
            order: (p, d, q) order of the ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order
            trend: Trend component ('n', 'c', 't', 'ct')
            enforce_stationarity: Whether to enforce stationarity
            enforce_invertibility: Whether to enforce invertibility
            concentrate_scale: Whether to concentrate scale
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.concentrate_scale = concentrate_scale
        self.model_ = None
        self.fitted_model_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit ARIMA model.

        Args:
            X: Features (ignored for ARIMA, uses y as time series)
            y: Time series target values

        Returns:
            self
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels is required for ARIMA. Install with: pip install statsmodels")

        # Convert to pandas Series if needed
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        elif not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Store the time series
        self.y_ = y.copy()

        try:
            # Initialize ARIMA model
            self.model_ = ARIMA(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
                concentrate_scale=self.concentrate_scale,
            )

            # Fit the model
            self.fitted_model_ = self.model_.fit()
            self.is_fitted_ = True

            print(f"ARIMA{self.order} model fitted successfully")
            if hasattr(self.fitted_model_, "aic"):
                print(f"AIC: {self.fitted_model_.aic:.2f}")

        except Exception as e:
            warnings.warn(f"ARIMA fitting failed: {e}")
            raise

        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Number of steps to forecast (or array-like for sklearn compatibility)

        Returns:
            Forecasted values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Handle different input types
        if isinstance(X, (int, float)):
            steps = int(X)
        elif hasattr(X, "__len__"):
            steps = len(X)
        else:
            steps = 1

        try:
            # Make forecast
            forecast = self.fitted_model_.forecast(steps=steps)

            if isinstance(forecast, pd.Series):
                return forecast.values
            else:
                return np.array(forecast)
        except Exception as e:
            warnings.warn(f"ARIMA prediction failed: {e}")
            # Return last known value repeated as fallback
            return np.full(steps, self.y_.iloc[-1])

    def forecast(self, steps: int = 1, alpha: float = 0.05):
        """
        Make forecast with confidence intervals.

        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary with forecast, confidence intervals, and dates
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Get forecast with confidence intervals
            forecast_result = self.fitted_model_.get_forecast(steps=steps, alpha=alpha)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            return {
                "forecast": forecast.values if hasattr(forecast, "values") else forecast,
                "lower_ci": conf_int.iloc[:, 0].values if hasattr(conf_int, "iloc") else conf_int[:, 0],
                "upper_ci": conf_int.iloc[:, 1].values if hasattr(conf_int, "iloc") else conf_int[:, 1],
                "forecast_dates": forecast.index if hasattr(forecast, "index") else None,
            }
        except Exception as e:
            warnings.warn(f"ARIMA forecast with CI failed: {e}")
            # Fallback to simple forecast
            simple_forecast = self.predict(steps)
            return {
                "forecast": simple_forecast,
                "lower_ci": simple_forecast * 0.9,  # Simple approximation
                "upper_ci": simple_forecast * 1.1,
                "forecast_dates": None,
            }

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "concentrate_scale": self.concentrate_scale,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def summary(self):
        """Get model summary."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting summary")

        if hasattr(self.fitted_model_, "summary"):
            return self.fitted_model_.summary()
        else:
            return f"ARIMA{self.order} model fitted on {len(self.y_)} observations"


class ProphetWrapper(BaseEstimator, RegressorMixin):
    """
    Facebook Prophet model wrapper compatible with sklearn.
    """

    def __init__(
        self,
        growth: str = "linear",
        changepoints: Optional[list] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Union[bool, str] = "auto",
        weekly_seasonality: Union[bool, str] = "auto",
        daily_seasonality: Union[bool, str] = "auto",
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.80,
        uncertainty_samples: int = 1000,
    ):
        """
        Initialize Prophet wrapper.

        Args:
            growth: Growth type ('linear' or 'logistic')
            changepoints: List of changepoint dates
            n_changepoints: Number of changepoints
            changepoint_range: Proportion of history for changepoints
            yearly_seasonality: Yearly seasonality
            weekly_seasonality: Weekly seasonality
            daily_seasonality: Daily seasonality
            seasonality_mode: Seasonality mode ('additive' or 'multiplicative')
            seasonality_prior_scale: Prior scale for seasonality
            holidays_prior_scale: Prior scale for holidays
            changepoint_prior_scale: Prior scale for changepoints
            mcmc_samples: Number of MCMC samples
            interval_width: Width of uncertainty intervals
            uncertainty_samples: Number of uncertainty samples
        """
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.model_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit Prophet model.

        Args:
            X: Features (should contain date column)
            y: Target values

        Returns:
            self
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required for Prophet. Install with: pip install prophet")

        # Prepare data for Prophet
        if isinstance(X, pd.DataFrame) and "date" in X.columns:
            df = pd.DataFrame({"ds": X["date"], "y": y})
        elif hasattr(y, "index") and isinstance(y.index, pd.DatetimeIndex):
            df = pd.DataFrame({"ds": y.index, "y": y.values})
        else:
            # Create artificial dates
            dates = pd.date_range(start="2020-01-01", periods=len(y), freq="D")
            df = pd.DataFrame({"ds": dates, "y": y})

        # Initialize Prophet model
        self.model_ = Prophet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples,
        )

        # Fit the model
        self.model_.fit(df)
        self.is_fitted_ = True

        print("Prophet model fitted successfully")
        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Number of periods or DataFrame with dates

        Returns:
            Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        if isinstance(X, (int, float)):
            # Create future dataframe
            future = self.model_.make_future_dataframe(periods=int(X))
        elif isinstance(X, pd.DataFrame):
            future = X
        else:
            # Fallback
            future = self.model_.make_future_dataframe(periods=len(X))

        forecast = self.model_.predict(future)
        return forecast["yhat"].tail(len(future) - len(self.model_.history)).values

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "growth": self.growth,
            "changepoints": self.changepoints,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "mcmc_samples": self.mcmc_samples,
            "interval_width": self.interval_width,
            "uncertainty_samples": self.uncertainty_samples,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
