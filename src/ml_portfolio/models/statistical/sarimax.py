"""
SARIMAX forecasting model wrapper.

Seasonal AutoRegressive Integrated Moving Average with eXogenous variables.
Perfect for seasonal time series with external predictors.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False


class SARIMAXForecaster(StatisticalForecaster):
    """
    SARIMAX forecasting model with sklearn-like interface.

    SARIMAX extends ARIMA with:
    - Seasonal components for periodic patterns
    - Exogenous variables for external predictors
    - More flexible for complex time series

    Great for retail sales, energy demand, transportation with seasonality.

    Args:
        order: Tuple of (p, d, q) for ARIMA component
        seasonal_order: Tuple of (P, D, Q, s) for seasonal component
            - P: Seasonal AR order
            - D: Seasonal differencing
            - Q: Seasonal MA order
            - s: Seasonal period (e.g., 12 for monthly, 52 for weekly)
        trend: Trend component ('n', 'c', 't', 'ct')
        measurement_error: Include measurement error component
        time_varying_regression: Allow time-varying coefficients
        mle_regression: Use MLE for regression coefficients
        method: Optimization method
        maxiter: Maximum iterations
        enforce_stationarity: Enforce stationarity constraints
        enforce_invertibility: Enforce invertibility constraints
        suppress_warnings: Suppress convergence warnings

    Example:
        >>> from ml_portfolio.models.statistical.sarimax import SARIMAXForecaster
        >>> # Monthly data with yearly seasonality
        >>> model = SARIMAXForecaster(
        ...     order=(1, 1, 1),
        ...     seasonal_order=(1, 1, 1, 12)
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: Optional[str] = None,
        measurement_error: bool = False,
        time_varying_regression: bool = False,
        mle_regression: bool = True,
        method: str = "lbfgs",
        maxiter: int = 50,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        suppress_warnings: bool = True,
        **kwargs,
    ):
        super().__init__()

        if not SARIMAX_AVAILABLE:
            raise ImportError("statsmodels is not installed. Install with: pip install statsmodels")

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.method = method
        self.maxiter = maxiter
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.suppress_warnings = suppress_warnings
        self.kwargs = kwargs

        # Will be initialized in fit()
        self.model = None
        self.model_fit_ = None
        self.date_column_ = None
        self.feature_names_ = None
        self.exog_columns_ = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Internal fit method for SARIMAX model.

        Args:
            X: Features dataframe (may include exogenous variables)
            y: Target values
            **kwargs: Additional fitting parameters
        """
        # Store feature names
        self.feature_names_ = list(X.columns)

        # Find date column
        date_cols = [col for col in X.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            self.date_column_ = date_cols[0]

        # Extract exogenous variables (all non-date columns)
        exog = None
        if len(X.columns) > 0:
            exog_cols = [col for col in X.columns if col != self.date_column_]
            if exog_cols:
                self.exog_columns_ = exog_cols
                exog = X[exog_cols]

        # Create SARIMAX model
        self.model = SARIMAX(
            endog=y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            measurement_error=self.measurement_error,
            time_varying_regression=self.time_varying_regression,
            mle_regression=self.mle_regression,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            **self.kwargs,
        )

        # Fit the model
        try:
            self.model_fit_ = self.model.fit(method=self.method, maxiter=self.maxiter, disp=False)
        except Exception as e:
            if not self.suppress_warnings:
                raise
            # Fall back to simpler model
            print(f"SARIMAX fitting failed: {e}")
            print("Falling back to (1,1,0)(0,0,0,0) model")
            self.model = SARIMAX(
                endog=y,
                exog=exog,
                order=(1, 1, 0),
                seasonal_order=(0, 0, 0, 0),
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit_ = self.model.fit(method=self.method, maxiter=self.maxiter, disp=False)

    def predict(
        self, X: pd.DataFrame, return_std: bool = False, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.

        Args:
            X: Features dataframe (must include exog vars if used in fit)
            return_std: Return standard deviation if True
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Predicted values
            std: Standard deviations (if return_std=True)
        """
        if self.model_fit_ is None:
            raise ValueError("Model must be fit before making predictions")

        n_periods = len(X)

        # Extract exogenous variables if they were used
        exog = None
        if self.exog_columns_ is not None:
            exog = X[self.exog_columns_]

        # Get forecast
        forecast_result = self.model_fit_.forecast(steps=n_periods, exog=exog)

        predictions = forecast_result.values if hasattr(forecast_result, "values") else np.array(forecast_result)

        if return_std:
            # Get confidence intervals
            forecast_obj = self.model_fit_.get_forecast(steps=n_periods, exog=exog)
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

        # Extract exogenous variables
        exog = None
        if self.exog_columns_ is not None:
            exog = X[self.exog_columns_]

        # Get forecast with confidence intervals
        forecast_obj = self.model_fit_.get_forecast(steps=n_periods, exog=exog)
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
            "measurement_error": self.measurement_error,
            "time_varying_regression": self.time_varying_regression,
            "mle_regression": self.mle_regression,
            "method": self.method,
            "maxiter": self.maxiter,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "suppress_warnings": self.suppress_warnings,
            **self.kwargs,
        }

    def set_params(self, **params) -> "SARIMAXForecaster":
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
            "llf": self.model_fit_.llf,
            "exog_used": self.exog_columns_ is not None,
            "exog_columns": self.exog_columns_ if self.exog_columns_ else [],
        }

        return info
