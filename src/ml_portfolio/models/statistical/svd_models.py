"""
SVD-based Cross-Series Forecasting Models.

Implements Singular Value Decomposition across stores to share patterns,
combined with ETS or STLF forecasting. This was cited as the best single
model by the M5 competition winner.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .ets import ETSForecaster


class SVDETSForecaster(BaseEstimator, RegressorMixin):
    """
    SVD + ETS forecasting model.

    Applies SVD to decompose cross-series patterns, then forecasts each
    component with ETS and reconstructs the final forecast.
    """

    def __init__(
        self, n_components: int = 10, seasonal_period: int = 52, standardize: bool = True, ets_auto_select: bool = True
    ):
        """
        Initialize SVD+ETS forecaster.

        Args:
            n_components: Number of SVD components to retain
            seasonal_period: Seasonal period for ETS models
            standardize: Whether to standardize series before SVD
            ets_auto_select: Whether ETS should auto-select best configuration
        """
        self.n_components = n_components
        self.seasonal_period = seasonal_period
        self.standardize = standardize
        self.ets_auto_select = ets_auto_select

        # Fitted components
        self.svd = None
        self.scaler = None
        self.component_forecasters = []
        self.series_means = None
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit SVD+ETS model on multiple time series.

        Args:
            X: Multiple time series data (n_timesteps, n_series) or DataFrame
            y: Not used (for sklearn compatibility)
        """
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = np.array(X)

        # Ensure correct shape (n_timesteps, n_series)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_timesteps, n_series = data.shape

        # Store series means for later reconstruction
        self.series_means = np.mean(data, axis=0)

        # Center the data
        centered_data = data - self.series_means

        # Standardize if requested
        if self.standardize:
            self.scaler = StandardScaler()
            centered_data = self.scaler.fit_transform(centered_data)

        # Apply SVD
        self.n_components = min(self.n_components, min(n_timesteps, n_series))
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)

        # Fit SVD and transform data
        transformed_data = self.svd.fit_transform(centered_data)

        # Fit ETS model to each SVD component
        self.component_forecasters = []

        for i in range(self.n_components):
            component_series = transformed_data[:, i]

            try:
                forecaster = ETSForecaster(seasonal_period=self.seasonal_period, auto_select=self.ets_auto_select)
                forecaster.fit(component_series)
                self.component_forecasters.append(forecaster)

            except Exception as e:
                print(f"Warning: Failed to fit ETS to component {i}: {e}")

                # Create a simple fallback forecaster
                class SimpleForecast:
                    def __init__(self, last_value):
                        self.last_value = last_value

                    def predict(self, steps):
                        return np.full(steps, self.last_value)

                self.component_forecasters.append(SimpleForecast(component_series[-1]))

        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate SVD+ETS forecasts (sklearn-compatible interface).

        Args:
            X: Feature matrix (sklearn interface) - we use X.shape[0] as number of steps

        Returns:
            Forecasts for all series (steps, n_series)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert sklearn interface to time series interface
        if hasattr(X, "shape"):
            steps = X.shape[0]  # Number of samples to predict
        else:
            steps = len(X) if hasattr(X, "__len__") else 1

        # Forecast each SVD component
        component_forecasts = []

        for i, forecaster in enumerate(self.component_forecasters):
            try:
                forecast = forecaster.predict(steps)
                component_forecasts.append(forecast)
            except Exception as e:
                print(f"Warning: Component {i} forecast failed: {e}")
                # Fallback to zero forecast for this component
                component_forecasts.append(np.zeros(steps))

        # Stack component forecasts
        component_forecasts = np.column_stack(component_forecasts)

        # Reconstruct original series forecasts using SVD inverse transform
        reconstructed = self.svd.inverse_transform(component_forecasts)

        # Reverse standardization if applied
        if self.standardize and self.scaler is not None:
            reconstructed = self.scaler.inverse_transform(reconstructed)

        # Add back series means
        reconstructed = reconstructed + self.series_means

        return reconstructed

    def predict_single_series(self, series_idx: int, steps: int = 1) -> np.ndarray:
        """
        Generate forecast for a single series.

        Args:
            series_idx: Index of the series to forecast
            steps: Number of steps to forecast

        Returns:
            Forecast for the specified series
        """
        all_forecasts = self.predict(steps)
        return all_forecasts[:, series_idx]

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "n_components": self.n_components,
            "seasonal_period": self.seasonal_period,
            "standardize": self.standardize,
            "ets_auto_select": self.ets_auto_select,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class SVDSTLFForecaster(BaseEstimator, RegressorMixin):
    """
    SVD + STLF forecasting model.

    Combines SVD cross-series pattern sharing with STL decomposition
    and forecasting, as used by the M5 winner.
    """

    def __init__(
        self,
        n_components: int = 10,
        seasonal_period: int = 52,
        stl_seasonal: Optional[int] = None,
        forecast_method: str = "ets",
    ):
        """
        Initialize SVD+STLF forecaster.

        Args:
            n_components: Number of SVD components
            seasonal_period: Seasonal period
            stl_seasonal: STL seasonal parameter (default: seasonal_period)
            forecast_method: Method for forecasting decomposed components ('ets', 'arima')
        """
        self.n_components = n_components
        self.seasonal_period = seasonal_period
        self.stl_seasonal = stl_seasonal or seasonal_period
        self.forecast_method = forecast_method

        # Fitted components
        self.svd = None
        self.component_forecasters = []
        self.series_means = None
        self.is_fitted = False

    def _stl_decompose(self, series: np.ndarray) -> dict:
        """
        Perform STL decomposition.

        Args:
            series: Time series to decompose

        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        try:
            from statsmodels.tsa.seasonal import STL

            stl = STL(series, seasonal=self.stl_seasonal, period=self.seasonal_period)
            result = stl.fit()

            return {"trend": result.trend.values, "seasonal": result.seasonal.values, "resid": result.resid.values}
        except ImportError:
            # Fallback decomposition without statsmodels
            return self._simple_decompose(series)

    def _simple_decompose(self, series: np.ndarray) -> dict:
        """Simple decomposition fallback."""
        # Simple moving average for trend
        window = min(self.seasonal_period, len(series) // 2)
        if window < 3:
            trend = np.full_like(series, np.mean(series))
        else:
            trend = pd.Series(series).rolling(window=window, center=True).mean()
            trend = trend.fillna(method="bfill").fillna(method="ffill").values

        # Simple seasonal extraction
        detrended = series - trend
        seasonal = np.zeros_like(series)

        if len(series) >= self.seasonal_period:
            for i in range(self.seasonal_period):
                seasonal_values = detrended[i :: self.seasonal_period]
                seasonal[i :: self.seasonal_period] = np.mean(seasonal_values)

        # Residual
        resid = series - trend - seasonal

        return {"trend": trend, "seasonal": seasonal, "resid": resid}

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit SVD+STLF model.

        Args:
            X: Multiple time series data (n_timesteps, n_series)
            y: Not used
        """
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = np.array(X)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_timesteps, n_series = data.shape
        self.series_means = np.mean(data, axis=0)

        # Apply SVD to centered data
        centered_data = data - self.series_means
        self.n_components = min(self.n_components, min(n_timesteps, n_series))
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        transformed_data = self.svd.fit_transform(centered_data)

        # Fit STLF to each component
        self.component_forecasters = []

        for i in range(self.n_components):
            component_series = transformed_data[:, i]

            try:
                # Decompose component
                decomposition = self._stl_decompose(component_series)

                # Forecast each decomposed part
                forecaster_dict = {}

                # Trend + residual (seasonally adjusted)
                seasonally_adjusted = decomposition["trend"] + decomposition["resid"]

                if self.forecast_method == "ets":
                    trend_forecaster = ETSForecaster(seasonal_period=self.seasonal_period, auto_select=True)
                else:  # arima fallback
                    from .statistical import ARIMAWrapper

                    trend_forecaster = ARIMAWrapper(order=(1, 1, 1))

                trend_forecaster.fit(seasonally_adjusted)
                forecaster_dict["trend"] = trend_forecaster

                # Store last seasonal pattern for forecasting
                forecaster_dict["seasonal"] = decomposition["seasonal"]

                self.component_forecasters.append(forecaster_dict)

            except Exception as e:
                print(f"Warning: Failed to fit STLF to component {i}: {e}")

                # Simple fallback
                class SimpleForecast:
                    def predict(self, steps):
                        return np.full(steps, component_series[-1] if len(component_series) > 0 else 0)

                self.component_forecasters.append(
                    {"trend": SimpleForecast(), "seasonal": np.zeros(self.seasonal_period)}
                )

        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Generate SVD+STLF forecasts (sklearn-compatible interface)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert sklearn interface to time series interface
        if hasattr(X, "shape"):
            steps = X.shape[0]  # Number of samples to predict
        else:
            steps = len(X) if hasattr(X, "__len__") else 1

        component_forecasts = []

        for forecaster_dict in self.component_forecasters:
            try:
                # Forecast trend component
                trend_forecast = forecaster_dict["trend"].predict(steps)

                # Add seasonal component
                seasonal_pattern = forecaster_dict["seasonal"]
                seasonal_forecast = []

                for i in range(steps):
                    seasonal_idx = i % len(seasonal_pattern)
                    seasonal_forecast.append(seasonal_pattern[seasonal_idx])

                # Combine trend and seasonal
                total_forecast = trend_forecast + np.array(seasonal_forecast)
                component_forecasts.append(total_forecast)

            except Exception as e:
                print(f"Warning: Component forecast failed: {e}")
                component_forecasts.append(np.zeros(steps))

        # Reconstruct forecasts
        component_forecasts = np.column_stack(component_forecasts)
        reconstructed = self.svd.inverse_transform(component_forecasts)

        # Add back means
        reconstructed = reconstructed + self.series_means

        return reconstructed

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "n_components": self.n_components,
            "seasonal_period": self.seasonal_period,
            "stl_seasonal": self.stl_seasonal,
            "forecast_method": self.forecast_method,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
