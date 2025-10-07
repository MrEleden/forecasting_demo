"""
ARIMA forecasting model wrapper.

AutoRegressive Integrated Moving Average model for time series forecasting.
Excellent for non-seasonal univariate time series with trends.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster

try:
    from statsmodels.tsa.arima.model import ARIMA

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


class ARIMAForecaster(StatisticalForecaster):
    """
    ARIMA forecasting model with sklearn-like interface.

    ARIMA (AutoRegressive Integrated Moving Average) is designed for:
    - Univariate time series forecasting
    - Capturing trends and patterns
    - Non-seasonal data (or use SARIMAX for seasonal)
    - Short to medium term forecasts

    Great for financial data, demand forecasting without strong seasonality.

    Args:
        order: Tuple of (p, d, q) for ARIMA model
            - p: AR order (autoregressive terms)
            - d: Degree of differencing
            - q: MA order (moving average terms)
        seasonal_order: Tuple of (P, D, Q, s) for seasonal ARIMA
            - Set to (0, 0, 0, 0) for non-seasonal
        trend: Trend component ('n', 'c', 't', 'ct')
            - n: no trend
            - c: constant
            - t: linear trend
            - ct: constant + linear trend
        method: Optimization method ('lbfgs', 'bfgs', etc.)
        maxiter: Maximum iterations for optimization
        suppress_warnings: Suppress convergence warnings

    Example:
        >>> from ml_portfolio.models.statistical.arima import ARIMAForecaster
        >>> model = ARIMAForecaster(order=(1, 1, 1))
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        method: str = "lbfgs",
        maxiter: int = 50,
        suppress_warnings: bool = True,
        **kwargs,
    ):
        super().__init__()

        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels is not installed. Install with: pip install statsmodels")

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.method = method
        self.maxiter = maxiter
        self.suppress_warnings = suppress_warnings
        self.kwargs = kwargs

        # Will be initialized in fit()
        self.model = None
        self.model_fit_ = None
        self.date_column_ = None
        self.feature_names_ = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Internal fit method for ARIMA model.

        Args:
            X: Features dataframe (date column required)
            y: Target values
            **kwargs: Additional fitting parameters
        """
        # Store feature names
        self.feature_names_ = list(X.columns)

        # Find date column
        date_cols = [col for col in X.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            self.date_column_ = date_cols[0]

        # Create ARIMA model
        # ARIMA expects just the target series
        self.model = ARIMA(
            endog=y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            **self.kwargs,
        )

        # Fit the model
        try:
            self.model_fit_ = self.model.fit(method=self.method, maxiter=self.maxiter)
        except Exception as e:
            if not self.suppress_warnings:
                raise
            # Fall back to simpler model
            print(f"ARIMA fitting failed: {e}")
            print("Falling back to (1,1,0) model")
            self.model = ARIMA(endog=y, order=(1, 1, 0), trend=self.trend)
            self.model_fit_ = self.model.fit(method=self.method, maxiter=self.maxiter)

    def predict(
        self, X: pd.DataFrame, return_std: bool = False, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.

        Args:
            X: Features dataframe
            return_std: Return standard deviation if True
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Predicted values
            std: Standard deviations (if return_std=True)
        """
        if self.model_fit_ is None:
            raise ValueError("Model must be fit before making predictions")

        n_periods = len(X)

        # Get forecast
        forecast_result = self.model_fit_.forecast(steps=n_periods)

        predictions = forecast_result.values if hasattr(forecast_result, "values") else np.array(forecast_result)

        if return_std:
            # Get confidence intervals
            forecast_obj = self.model_fit_.get_forecast(steps=n_periods)
            conf_int = forecast_obj.conf_int()
            # Approximate std from confidence interval
            std = (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]) / (2 * 1.96)  # 95% CI
            return predictions, std.values

        return predictions

    def predict_interval(
        self, X: pd.DataFrame, confidence: float = 0.95, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.

        Args:
            X: Features dataframe
            confidence: Confidence level (default 0.95)
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Point forecasts
            lower: Lower confidence bounds
            upper: Upper confidence bounds
        """
        if self.model_fit_ is None:
            raise ValueError("Model must be fit before making predictions")

        n_periods = len(X)

        # Get forecast with confidence intervals
        forecast_obj = self.model_fit_.get_forecast(steps=n_periods)
        predictions = forecast_obj.predicted_mean.values

        # Get confidence intervals
        alpha = 1 - confidence
        conf_int = forecast_obj.conf_int(alpha=alpha)
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values

        return predictions, lower, upper

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "method": self.method,
            "maxiter": self.maxiter,
            "suppress_warnings": self.suppress_warnings,
            **self.kwargs,
        }

    def set_params(self, **params) -> "ARIMAForecaster":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and diagnostics."""
        if self.model_fit_ is None:
            return {}

        info = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": self.model_fit_.aic,
            "bic": self.model_fit_.bic,
            "hqic": self.model_fit_.hqic,
            "llf": self.model_fit_.llf,  # Log-likelihood
        }

        return info
