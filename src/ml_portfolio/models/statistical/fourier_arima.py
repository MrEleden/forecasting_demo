"""
Fourier-based ARIMA Models (Dynamic Harmonic Regression).

Implements ARIMA models with Fourier terms as exogenous regressors.
This approach was noted as pivotal to the M5 winning blend.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class FourierARIMAForecaster(BaseEstimator, RegressorMixin):
    """
    Fourier + ARIMA forecasting model.

    Implements Dynamic Harmonic Regression where ARIMA is fitted
    on seasonal regressors built from Fourier terms.
    """

    def __init__(
        self,
        seasonal_period: int = 52,
        fourier_terms: int = 6,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
        auto_arima: bool = True,
    ):
        """
        Initialize Fourier+ARIMA forecaster.

        Args:
            seasonal_period: Seasonal period for Fourier terms
            fourier_terms: Number of Fourier term pairs (sin/cos)
            arima_order: ARIMA order (p, d, q) if not using auto_arima
            auto_arima: Whether to automatically select ARIMA order
        """
        self.seasonal_period = seasonal_period
        self.fourier_terms = fourier_terms
        self.arima_order = arima_order
        self.auto_arima = auto_arima

        # Fitted components
        self.arima_model = None
        self.fourier_coefficients = None
        self.training_length = 0
        self.is_fitted = False

    def _create_fourier_terms(self, n_periods: int, start_period: int = 1) -> np.ndarray:
        """
        Create Fourier terms for harmonic regression.

        Args:
            n_periods: Number of periods to generate terms for
            start_period: Starting period number

        Returns:
            Matrix of Fourier terms (n_periods, 2*fourier_terms)
        """
        t = np.arange(start_period, start_period + n_periods)
        fourier_matrix = []

        for k in range(1, self.fourier_terms + 1):
            # Sine and cosine terms for each harmonic
            sin_term = np.sin(2 * np.pi * k * t / self.seasonal_period)
            cos_term = np.cos(2 * np.pi * k * t / self.seasonal_period)
            fourier_matrix.append(sin_term)
            fourier_matrix.append(cos_term)

        return np.column_stack(fourier_matrix)

    def _fit_arima_with_fourier(self, series: np.ndarray, fourier_matrix: np.ndarray):
        """
        Fit ARIMA model with Fourier terms as exogenous variables.

        Args:
            series: Time series data
            fourier_matrix: Fourier terms matrix

        Returns:
            Fitted ARIMA model
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA

            if self.auto_arima:
                # Try different ARIMA orders
                orders_to_try = [(1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1), (2, 0, 2), (1, 0, 0), (0, 0, 1)]

                best_aic = float("inf")
                best_model = None

                for order in orders_to_try:
                    try:
                        model = ARIMA(series, order=order, exog=fourier_matrix)
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted

                    except Exception:
                        continue

                if best_model is None:
                    # Fallback to simple ARIMA without exogenous variables
                    model = ARIMA(series, order=(1, 1, 1))
                    best_model = model.fit()

                return best_model
            else:
                # Use specified order
                model = ARIMA(series, order=self.arima_order, exog=fourier_matrix)
                return model.fit()

        except ImportError:
            # Fallback using ARIMAWrapper from ml_portfolio
            from .statistical import ARIMAWrapper

            # Simple ARIMA without exogenous terms as fallback
            arima = ARIMAWrapper(order=self.arima_order)
            arima.fit(None, series)

            # Create wrapper to match statsmodels interface
            class ARIMAFourierWrapper:
                def __init__(self, arima_model, fourier_matrix, series):
                    self.arima_model = arima_model
                    self.fourier_matrix = fourier_matrix
                    self.series = series

                def forecast(self, steps, exog=None):
                    # Simple forecast using ARIMA only
                    return self.arima_model.predict(steps)

            return ARIMAFourierWrapper(arima, fourier_matrix, series)

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """
        Fit Fourier+ARIMA model.

        Args:
            X: Time series data
            y: Not used (for sklearn compatibility)
        """
        if isinstance(X, pd.Series):
            series = X.values.astype(float)
        else:
            series = np.array(X, dtype=float).flatten()

        self.training_length = len(series)

        # Create Fourier terms for training period
        fourier_matrix = self._create_fourier_terms(len(series))

        # Fit ARIMA with Fourier terms
        self.arima_model = self._fit_arima_with_fourier(series, fourier_matrix)

        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate Fourier+ARIMA forecasts (sklearn-compatible interface).

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
            # Create Fourier terms for forecast period
            forecast_fourier = self._create_fourier_terms(steps, start_period=self.training_length + 1)

            # Generate forecast
            if hasattr(self.arima_model, "forecast"):
                forecast = self.arima_model.forecast(steps=steps, exog=forecast_fourier)
                return forecast.values if hasattr(forecast, "values") else forecast
            else:
                # Fallback for wrapped ARIMA
                return self.arima_model.forecast(steps, exog=forecast_fourier)

        except Exception as e:
            print(f"Warning: Fourier+ARIMA forecast failed: {e}")
            # Fallback to simple ARIMA forecast
            try:
                return self.arima_model.forecast(steps)
            except Exception:
                # Last resort - return zeros
                return np.zeros(steps)

    def get_fourier_importance(self) -> pd.DataFrame:
        """
        Get importance of Fourier terms from fitted model.

        Returns:
            DataFrame with Fourier term coefficients and significance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        try:
            if hasattr(self.arima_model, "params") and hasattr(self.arima_model, "pvalues"):
                # Extract Fourier coefficients from statsmodels ARIMA
                fourier_names = []
                for k in range(1, self.fourier_terms + 1):
                    fourier_names.extend([f"sin_{k}", f"cos_{k}"])

                # Get coefficients for exogenous variables (Fourier terms)
                n_arima_params = len(self.arima_model.params) - len(fourier_names)
                fourier_params = self.arima_model.params[n_arima_params:]
                fourier_pvals = self.arima_model.pvalues[n_arima_params:]

                return pd.DataFrame(
                    {
                        "fourier_term": fourier_names,
                        "coefficient": fourier_params.values,
                        "p_value": fourier_pvals.values,
                        "significant": fourier_pvals.values < 0.05,
                    }
                )
            else:
                return pd.DataFrame({"message": ["Fourier importance not available for this model type"]})

        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "seasonal_period": self.seasonal_period,
            "fourier_terms": self.fourier_terms,
            "arima_order": self.arima_order,
            "auto_arima": self.auto_arima,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class AdaptiveFourierARIMA(FourierARIMAForecaster):
    """
    Adaptive Fourier+ARIMA that automatically selects number of Fourier terms.

    Uses information criteria to select optimal number of Fourier terms.
    """

    def __init__(
        self, seasonal_period: int = 52, max_fourier_terms: int = 10, selection_criterion: str = "aic", **kwargs
    ):
        """
        Initialize adaptive Fourier+ARIMA.

        Args:
            seasonal_period: Seasonal period
            max_fourier_terms: Maximum number of Fourier terms to try
            selection_criterion: Criterion for model selection ('aic', 'bic')
            **kwargs: Additional arguments for base class
        """
        super().__init__(seasonal_period=seasonal_period, fourier_terms=1, **kwargs)
        self.max_fourier_terms = max_fourier_terms
        self.selection_criterion = selection_criterion
        self.optimal_fourier_terms = None

    def fit(self, X: Union[np.ndarray, pd.Series], y: Optional[np.ndarray] = None):
        """
        Fit adaptive Fourier+ARIMA with optimal number of terms.

        Args:
            X: Time series data
            y: Not used
        """
        if isinstance(X, pd.Series):
            series = X.values.astype(float)
        else:
            series = np.array(X, dtype=float).flatten()

        self.training_length = len(series)

        # Try different numbers of Fourier terms
        best_criterion = float("inf")
        best_model = None
        best_fourier_terms = 1

        for n_terms in range(1, min(self.max_fourier_terms + 1, self.seasonal_period // 2)):
            try:
                # Set fourier_terms for this iteration
                self.fourier_terms = n_terms

                # Create Fourier matrix
                fourier_matrix = self._create_fourier_terms(len(series))

                # Fit model
                model = self._fit_arima_with_fourier(series, fourier_matrix)

                # Get criterion value
                if hasattr(model, self.selection_criterion):
                    criterion_value = getattr(model, self.selection_criterion)

                    if criterion_value < best_criterion:
                        best_criterion = criterion_value
                        best_model = model
                        best_fourier_terms = n_terms

            except Exception as e:
                print(f"Failed to fit model with {n_terms} Fourier terms: {e}")
                continue

        # Set optimal configuration
        self.fourier_terms = best_fourier_terms
        self.optimal_fourier_terms = best_fourier_terms
        self.arima_model = best_model

        if best_model is None:
            # Fallback to simple model
            self.fourier_terms = 1
            super().fit(X, y)
        else:
            self.is_fitted = True

        return self

    def get_model_selection_summary(self) -> dict:
        """
        Get summary of model selection process.

        Returns:
            Dictionary with model selection details
        """
        return {
            "optimal_fourier_terms": self.optimal_fourier_terms,
            "max_terms_tried": self.max_fourier_terms,
            "selection_criterion": self.selection_criterion,
            "seasonal_period": self.seasonal_period,
        }
