"""
Prophet forecasting model wrapper.

Facebook's Prophet library for time series forecasting.
Excellent for handling seasonality, holidays, and missing data.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..base import StatisticalForecaster

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetForecaster(StatisticalForecaster):
    """
    Prophet forecasting model with sklearn-like interface.

    Facebook's Prophet is designed for:
    - Time series with strong seasonal patterns
    - Holiday effects
    - Missing data handling
    - Interpretable components (trend, seasonality)
    - Uncertainty intervals

    Great for business forecasting with clear seasonality.

    Args:
        growth: Growth model ('linear' or 'logistic')
        yearly_seasonality: Fit yearly seasonality (True, False, or 'auto')
        weekly_seasonality: Fit weekly seasonality (True, False, or 'auto')
        daily_seasonality: Fit daily seasonality (True, False, or 'auto')
        n_changepoints: Number of potential changepoints
        changepoint_range: Proportion of history for changepoints
        changepoint_prior_scale: Flexibility of changepoint selection
        seasonality_prior_scale: Strength of seasonality model
        seasonality_mode: 'additive' or 'multiplicative'
        holidays_prior_scale: Strength of holiday effect
        mcmc_samples: Number of MCMC samples (0 for MAP estimation)
        uncertainty_samples: Number of samples for uncertainty intervals
        interval_width: Width of uncertainty intervals
        **kwargs: Additional Prophet parameters

    Example:
        >>> from ml_portfolio.models.statistical.prophet import ProphetForecaster
        >>> model = ProphetForecaster(yearly_seasonality=True, weekly_seasonality=True)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        growth: str = "linear",
        yearly_seasonality: Union[bool, str] = "auto",
        weekly_seasonality: Union[bool, str] = "auto",
        daily_seasonality: Union[bool, str] = False,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        holidays_prior_scale: float = 10.0,
        mcmc_samples: int = 0,
        uncertainty_samples: int = 1000,
        interval_width: float = 0.8,
        **kwargs,
    ):
        # Initialize base class
        super().__init__()

        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.holidays_prior_scale = holidays_prior_scale
        self.mcmc_samples = mcmc_samples
        self.uncertainty_samples = uncertainty_samples
        self.interval_width = interval_width
        self.kwargs = kwargs

        # Will be initialized in fit()
        self.model = None
        self.date_column_ = None
        self.feature_names_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        date_column: Optional[str] = None,
        **fit_kwargs,
    ) -> "ProphetForecaster":
        """
        Fit the Prophet model.

        Prophet expects data in a specific format with 'ds' (date) and 'y' (target) columns.
        If X contains a date column, specify it with date_column parameter.

        Args:
            X: Training features (must include date information)
            y: Training target (n_samples,)
            date_column: Name of the date column in X (if DataFrame)
            **fit_kwargs: Additional fit parameters

        Returns:
            self: Fitted model
        """
        # Convert inputs to DataFrame format expected by Prophet
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_ = X.columns.tolist()

            # Find date column
            if date_column is not None:
                self.date_column_ = date_column
            else:
                # Try to find date column automatically
                date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
                if date_cols:
                    self.date_column_ = date_cols[0]
                else:
                    raise ValueError("Could not find date column. Please specify with date_column parameter.")
        else:
            # If numpy array, assume first column is date
            if X.shape[1] < 1:
                raise ValueError("X must have at least one column (date)")

            # Create DataFrame with string column names
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            self.date_column_ = feature_names[0]
            self.feature_names_ = feature_names

        # Convert y to series
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()

        # Handle NaN values (Prophet doesn't accept NaN)
        # Fill NaN with forward fill, then backward fill, then 0
        df_filled = df.ffill().bfill().fillna(0)

        # Create Prophet DataFrame format
        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df_filled[self.date_column_]), "y": y})

        # Add additional regressors if present
        regressor_cols = [col for col in df_filled.columns if col != self.date_column_]
        for col in regressor_cols:
            prophet_df[str(col)] = df_filled[col].values  # Ensure column name is string

        # Initialize Prophet model
        self.model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            holidays_prior_scale=self.holidays_prior_scale,
            mcmc_samples=self.mcmc_samples,
            uncertainty_samples=self.uncertainty_samples,
            interval_width=self.interval_width,
            **self.kwargs,
        )

        # Add regressors to model
        for col in regressor_cols:
            self.model.add_regressor(str(col))  # Ensure regressor name is string

        # Fit model
        self.model.fit(prophet_df, **fit_kwargs)

        # Mark as fitted
        self.is_fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the fitted model.

        Args:
            X: Features to predict on (must include date information)

        Returns:
            predictions: Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to DataFrame
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_)

        # Handle NaN values (Prophet doesn't accept NaN)
        df_filled = df.ffill().bfill().fillna(0)

        # Create Prophet DataFrame format
        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df_filled[self.date_column_])})

        # Add regressors
        regressor_cols = [col for col in df_filled.columns if col != self.date_column_]
        for col in regressor_cols:
            prophet_df[str(col)] = df_filled[col].values  # Ensure column name is string

        # Make predictions
        forecast = self.model.predict(prophet_df)

        return forecast["yhat"].values

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate R^2 score on test data.

        Args:
            X: Test features
            y: True target values

        Returns:
            r2_score: R^2 coefficient of determination
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")

        from sklearn.metrics import r2_score

        y_pred = self.predict(X)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if y.ndim > 1:
            y = y.ravel()

        return r2_score(y, y_pred)

    def predict_with_uncertainty(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Make predictions with uncertainty intervals.

        Args:
            X: Features to predict on

        Returns:
            DataFrame with columns: yhat (prediction), yhat_lower, yhat_upper
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to DataFrame
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_)

        # Create Prophet DataFrame format
        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df[self.date_column_])})

        # Add regressors
        regressor_cols = [col for col in df.columns if col != self.date_column_]
        for col in regressor_cols:
            prophet_df[col] = df[col].values

        # Make predictions
        forecast = self.model.predict(prophet_df)

        return forecast[["yhat", "yhat_lower", "yhat_upper"]]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get model parameters (sklearn compatibility).

        Args:
            deep: If True, return parameters for nested objects

        Returns:
            params: Model parameters
        """
        params = {
            "growth": self.growth,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "seasonality_mode": self.seasonality_mode,
            "holidays_prior_scale": self.holidays_prior_scale,
            "mcmc_samples": self.mcmc_samples,
            "uncertainty_samples": self.uncertainty_samples,
            "interval_width": self.interval_width,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params) -> "ProphetForecaster":
        """
        Set model parameters (sklearn compatibility).

        Args:
            **params: Parameters to set

        Returns:
            self: Updated model
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"ProphetForecaster(growth={self.growth}, "
            f"yearly_seasonality={self.yearly_seasonality}, "
            f"weekly_seasonality={self.weekly_seasonality})"
        )
