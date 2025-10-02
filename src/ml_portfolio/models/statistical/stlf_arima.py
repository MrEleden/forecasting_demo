"""
STL + ARIMA Forecasting Models.

Implements STL decomposition followed by ARIMA on the seasonally adjusted series.
Used as a strong base component in M5 competition approaches.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class STLFARIMAForecaster(BaseEstimator, RegressorMixin):
    """
    STL + ARIMA forecasting model.

    Performs STL decomposition and then applies ARIMA to the
    seasonally adjusted (trend + residual) series.
    """

    def __init__(
        self,
        seasonal_period: int = 52,
        stl_seasonal: Optional[int] = None,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
        auto_arima: bool = True,
        seasonal_forecast_method: str = "naive",
    ):
        """
        Initialize STL+ARIMA forecaster.

        Args:
            seasonal_period: Seasonal period
            stl_seasonal: STL seasonal parameter (default: seasonal_period)
            arima_order: ARIMA order (p, d, q) if not using auto_arima
            auto_arima: Whether to automatically select ARIMA order
            seasonal_forecast_method: How to forecast seasonal component ('naive', 'drift')
        """
        self.seasonal_period = seasonal_period
        self.stl_seasonal = stl_seasonal or seasonal_period
        self.arima_order = arima_order
        self.auto_arima = auto_arima
        self.seasonal_forecast_method = seasonal_forecast_method

        # Fitted components
        self.arima_model = None
        self.seasonal_component = None
        self.trend_component = None
        self.residual_component = None
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
            print("statsmodels not available, using simple decomposition")
            return self._simple_decompose(series)
        except Exception as e:
            print(f"STL decomposition failed: {e}, using simple decomposition")
            return self._simple_decompose(series)

    def _simple_decompose(self, series: np.ndarray) -> dict:
        """
        Simple decomposition fallback without statsmodels.

        Args:
            series: Time series to decompose

        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        # Simple moving average for trend
        window = min(self.seasonal_period, len(series) // 2)
        if window < 3:
            window = 3

        # Centered moving average for trend
        trend = pd.Series(series).rolling(window=window, center=True).mean()
        trend = trend.fillna(method="bfill").fillna(method="ffill").values

        # Extract seasonal component
        detrended = series - trend
        seasonal = np.zeros_like(series)

        if len(series) >= self.seasonal_period:
            # Calculate seasonal indices
            for i in range(self.seasonal_period):
                seasonal_values = detrended[i :: self.seasonal_period]
                if len(seasonal_values) > 0:
                    seasonal[i :: self.seasonal_period] = np.median(seasonal_values)

        # Residual component
        resid = series - trend - seasonal

        return {"trend": trend, "seasonal": seasonal, "resid": resid}

    def _fit_arima(self, series: np.ndarray):
        """
        Fit ARIMA model to seasonally adjusted series.

        Args:
            series: Seasonally adjusted time series

        Returns:
            Fitted ARIMA model
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA

            if self.auto_arima:
                # Try different ARIMA orders
                orders_to_try = [
                    (1, 1, 1),
                    (2, 1, 2),
                    (1, 1, 0),
                    (0, 1, 1),
                    (2, 1, 1),
                    (1, 1, 2),
                    (2, 0, 2),
                    (1, 0, 1),
                    (0, 1, 0),
                    (1, 0, 0),
                    (0, 0, 1),
                ]

                best_aic = float("inf")
                best_model = None

                for order in orders_to_try:
                    try:
                        model = ARIMA(series, order=order)
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted

                    except Exception:
                        continue

                if best_model is None:
                    # Final fallback
                    model = ARIMA(series, order=(1, 1, 1))
                    best_model = model.fit()

                return best_model
            else:
                # Use specified order
                model = ARIMA(series, order=self.arima_order)
                return model.fit()

        except ImportError:
            # Fallback using ARIMAWrapper
            from .statistical import ARIMAWrapper

            arima = ARIMAWrapper(order=self.arima_order)
            arima.fit(None, series)

            # Wrapper to match statsmodels interface
            class ARIMAWrapper2:
                def __init__(self, arima_model):
                    self.arima_model = arima_model

                def forecast(self, steps):
                    return self.arima_model.predict(steps)

            return ARIMAWrapper2(arima)

    def _forecast_seasonal_component(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for seasonal component.

        Args:
            steps: Number of steps to forecast

        Returns:
            Seasonal component forecasts
        """
        if self.seasonal_component is None:
            return np.zeros(steps)

        seasonal_forecasts = []

        if self.seasonal_forecast_method == "naive":
            # Repeat seasonal pattern
            for i in range(steps):
                seasonal_idx = i % len(self.seasonal_component)
                seasonal_forecasts.append(self.seasonal_component[seasonal_idx])

        elif self.seasonal_forecast_method == "drift":
            # Apply slight drift to seasonal pattern
            base_seasonal = [self.seasonal_component[i % len(self.seasonal_component)] for i in range(steps)]

            # Calculate seasonal drift (change in last vs first seasonal cycle)
            if len(self.seasonal_component) >= 2 * self.seasonal_period:
                last_cycle = self.seasonal_component[-self.seasonal_period :]
                first_cycle = self.seasonal_component[: self.seasonal_period]
                drift_per_period = (np.mean(last_cycle) - np.mean(first_cycle)) / len(self.seasonal_component)

                for i in range(steps):
                    drift_adjustment = drift_per_period * (len(self.seasonal_component) + i)
                    seasonal_forecasts.append(base_seasonal[i] + drift_adjustment)
            else:
                seasonal_forecasts = base_seasonal

        return np.array(seasonal_forecasts)

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """
        Fit STL+ARIMA model.

        Args:
            X: Time series data
            y: Not used (for sklearn compatibility)
        """
        if isinstance(X, pd.Series):
            series = X.values.astype(float)
        else:
            series = np.array(X, dtype=float).flatten()

        # Perform STL decomposition
        decomposition = self._stl_decompose(series)

        self.trend_component = decomposition["trend"]
        self.seasonal_component = decomposition["seasonal"]
        self.residual_component = decomposition["resid"]

        # Create seasonally adjusted series (trend + residual)
        seasonally_adjusted = self.trend_component + self.residual_component

        # Fit ARIMA to seasonally adjusted series
        self.arima_model = self._fit_arima(seasonally_adjusted)

        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate STL+ARIMA forecasts (sklearn-compatible interface).

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

        # Forecast seasonally adjusted series with ARIMA
        try:
            if hasattr(self.arima_model, "forecast"):
                trend_forecast = self.arima_model.forecast(steps)
                if hasattr(trend_forecast, "values"):
                    trend_forecast = trend_forecast.values
            else:
                trend_forecast = self.arima_model.forecast(steps)
        except Exception as e:
            print(f"ARIMA forecast failed: {e}")
            # Fallback to last trend value
            last_trend = self.trend_component[-1] + self.residual_component[-1]
            trend_forecast = np.full(steps, last_trend)

        # Forecast seasonal component
        seasonal_forecast = self._forecast_seasonal_component(steps)

        # Combine forecasts
        total_forecast = trend_forecast + seasonal_forecast

        return total_forecast

    def get_decomposition(self) -> dict:
        """
        Get the STL decomposition components.

        Returns:
            Dictionary with decomposition components
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return {"trend": self.trend_component, "seasonal": self.seasonal_component, "residual": self.residual_component}

    def plot_decomposition(self):
        """Plot STL decomposition components."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            # Original series
            original = self.trend_component + self.seasonal_component + self.residual_component
            axes[0].plot(original)
            axes[0].set_title("Original Series")
            axes[0].grid(True, alpha=0.3)

            # Trend
            axes[1].plot(self.trend_component)
            axes[1].set_title("Trend Component")
            axes[1].grid(True, alpha=0.3)

            # Seasonal
            axes[2].plot(self.seasonal_component)
            axes[2].set_title("Seasonal Component")
            axes[2].grid(True, alpha=0.3)

            # Residual
            axes[3].plot(self.residual_component)
            axes[3].set_title("Residual Component")
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not available for plotting")

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "seasonal_period": self.seasonal_period,
            "stl_seasonal": self.stl_seasonal,
            "arima_order": self.arima_order,
            "auto_arima": self.auto_arima,
            "seasonal_forecast_method": self.seasonal_forecast_method,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class RobustSTLFARIMA(STLFARIMAForecaster):
    """
    Robust STL+ARIMA with outlier handling and adaptive parameters.

    Enhanced version that handles outliers and automatically adapts
    STL parameters based on series characteristics.
    """

    def __init__(self, seasonal_period: int = 52, outlier_threshold: float = 3.0, robust_stl: bool = True, **kwargs):
        """
        Initialize robust STL+ARIMA.

        Args:
            seasonal_period: Seasonal period
            outlier_threshold: Threshold for outlier detection (standard deviations)
            robust_stl: Whether to use robust STL parameters
            **kwargs: Additional arguments for base class
        """
        super().__init__(seasonal_period=seasonal_period, **kwargs)
        self.outlier_threshold = outlier_threshold
        self.robust_stl = robust_stl
        self.outliers_detected = None

    def _detect_outliers(self, series: np.ndarray) -> np.ndarray:
        """
        Detect outliers using modified z-score.

        Args:
            series: Time series data

        Returns:
            Boolean array indicating outliers
        """
        median = np.median(series)
        mad = np.median(np.abs(series - median))

        # Modified z-score
        modified_z_scores = 0.6745 * (series - median) / mad if mad > 0 else np.zeros_like(series)

        return np.abs(modified_z_scores) > self.outlier_threshold

    def _adaptive_stl_params(self, series: np.ndarray) -> dict:
        """
        Determine adaptive STL parameters based on series characteristics.

        Args:
            series: Time series data

        Returns:
            Dictionary with STL parameters
        """
        n = len(series)

        if self.robust_stl:
            # Robust parameters for noisy data
            seasonal_param = max(7, self.seasonal_period)

            if n < 2 * self.seasonal_period:
                seasonal_param = n // 2 if n > 6 else 3

            return {"seasonal": seasonal_param}
        else:
            return {"seasonal": self.stl_seasonal}

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """
        Fit robust STL+ARIMA model.

        Args:
            X: Time series data
            y: Not used
        """
        if isinstance(X, pd.Series):
            series = X.values.astype(float)
        else:
            series = np.array(X, dtype=float).flatten()

        # Detect outliers
        self.outliers_detected = self._detect_outliers(series)

        # Get adaptive STL parameters
        stl_params = self._adaptive_stl_params(series)
        original_stl_seasonal = self.stl_seasonal
        self.stl_seasonal = stl_params["seasonal"]

        # Fit the model
        result = super().fit(X, y)

        # Restore original parameter
        self.stl_seasonal = original_stl_seasonal

        return result

    def get_outlier_summary(self) -> dict:
        """
        Get summary of detected outliers.

        Returns:
            Dictionary with outlier information
        """
        if self.outliers_detected is None:
            return {"message": "No outlier detection performed yet"}

        return {
            "total_outliers": np.sum(self.outliers_detected),
            "outlier_percentage": np.mean(self.outliers_detected) * 100,
            "outlier_indices": np.where(self.outliers_detected)[0].tolist(),
        }
