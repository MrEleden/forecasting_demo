"""
Classical machine learning pipelines for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, TransformerMixin


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create lag features for time series data.
    """

    def __init__(self, lags: List[int], columns: Optional[List[str]] = None):
        """
        Initialize LagFeatureTransformer.

        Args:
            lags: List of lag periods
            columns: Columns to create lags for (None for all)
        """
        self.lags = lags
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            if self.columns is None:
                self.columns = self.feature_names_
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            if self.columns is None:
                self.columns = self.feature_names_

        return self

    def transform(self, X):
        """Transform the data by adding lag features."""
        if isinstance(X, pd.DataFrame):
            X_lagged = X.copy()
        else:
            X_lagged = pd.DataFrame(X, columns=self.feature_names_)

        for col in self.columns:
            for lag in self.lags:
                lag_col_name = f"{col}_lag_{lag}"
                X_lagged[lag_col_name] = X_lagged[col].shift(lag)

        return X_lagged

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = self.feature_names_

        output_features = list(input_features)
        for col in self.columns:
            for lag in self.lags:
                output_features.append(f"{col}_lag_{lag}")

        return np.array(output_features)


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Create rolling statistics features.
    """

    def __init__(self, windows: List[int], stats: List[str] = ["mean", "std"], columns: Optional[List[str]] = None):
        """
        Initialize RollingStatsTransformer.

        Args:
            windows: List of rolling window sizes
            stats: List of statistics to compute
            columns: Columns to compute stats for (None for all)
        """
        self.windows = windows
        self.stats = stats
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            if self.columns is None:
                self.columns = self.feature_names_
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            if self.columns is None:
                self.columns = self.feature_names_

        return self

    def transform(self, X):
        """Transform the data by adding rolling statistics."""
        if isinstance(X, pd.DataFrame):
            X_rolled = X.copy()
        else:
            X_rolled = pd.DataFrame(X, columns=self.feature_names_)

        for col in self.columns:
            for window in self.windows:
                for stat in self.stats:
                    stat_col_name = f"{col}_rolling_{window}_{stat}"
                    if stat == "mean":
                        X_rolled[stat_col_name] = X_rolled[col].rolling(window).mean()
                    elif stat == "std":
                        X_rolled[stat_col_name] = X_rolled[col].rolling(window).std()
                    elif stat == "min":
                        X_rolled[stat_col_name] = X_rolled[col].rolling(window).min()
                    elif stat == "max":
                        X_rolled[stat_col_name] = X_rolled[col].rolling(window).max()
                    elif stat == "median":
                        X_rolled[stat_col_name] = X_rolled[col].rolling(window).median()

        return X_rolled

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = self.feature_names_

        output_features = list(input_features)
        for col in self.columns:
            for window in self.windows:
                for stat in self.stats:
                    output_features.append(f"{col}_rolling_{window}_{stat}")

        return np.array(output_features)


class DifferenceTransformer(BaseEstimator, TransformerMixin):
    """
    Apply differencing to make time series stationary.
    """

    def __init__(self, periods: int = 1, columns: Optional[List[str]] = None):
        """
        Initialize DifferenceTransformer.

        Args:
            periods: Number of periods to difference
            columns: Columns to difference (None for all)
        """
        self.periods = periods
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            if self.columns is None:
                self.columns = self.feature_names_
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            if self.columns is None:
                self.columns = self.feature_names_

        return self

    def transform(self, X):
        """Transform the data by differencing."""
        if isinstance(X, pd.DataFrame):
            X_diff = X.copy()
        else:
            X_diff = pd.DataFrame(X, columns=self.feature_names_)

        for col in self.columns:
            diff_col_name = f"{col}_diff_{self.periods}"
            X_diff[diff_col_name] = X_diff[col].diff(self.periods)

        return X_diff

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = self.feature_names_

        output_features = list(input_features)
        for col in self.columns:
            output_features.append(f"{col}_diff_{self.periods}")

        return np.array(output_features)


def create_time_series_pipeline(
    target_column: str,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    lag_features: Optional[Dict[str, List[int]]] = None,
    rolling_features: Optional[Dict[str, Dict]] = None,
    scaler: str = "standard",
    feature_selection: Optional[int] = None,
    handle_missing: str = "drop",
) -> Pipeline:
    """
    Create a comprehensive time series preprocessing pipeline.

    Args:
        target_column: Name of target column
        numeric_features: List of numeric feature columns
        categorical_features: List of categorical feature columns
        lag_features: Dictionary mapping columns to lag periods
        rolling_features: Dictionary with rolling window configurations
        scaler: Type of scaler ('standard', 'minmax', 'robust')
        feature_selection: Number of features to select (None for no selection)
        handle_missing: How to handle missing values ('drop', 'impute')

    Returns:
        Sklearn Pipeline
    """
    steps = []

    # Add lag features
    if lag_features:
        for col, lags in lag_features.items():
            lag_transformer = LagFeatureTransformer(lags=lags, columns=[col])
            steps.append((f"lag_{col}", lag_transformer))

    # Add rolling statistics
    if rolling_features:
        for col, config in rolling_features.items():
            windows = config.get("windows", [7, 30])
            stats = config.get("stats", ["mean", "std"])
            rolling_transformer = RollingStatsTransformer(windows=windows, stats=stats, columns=[col])
            steps.append((f"rolling_{col}", rolling_transformer))

    # Handle missing values
    if handle_missing == "impute":
        # Create column transformer for different feature types
        transformers = []

        if numeric_features:
            numeric_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )
            transformers.append(("numeric", numeric_transformer, numeric_features))

        if categorical_features:
            categorical_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            )
            transformers.append(("categorical", categorical_transformer, categorical_features))

        if transformers:
            preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
            steps.append(("preprocessing", preprocessor))

    # Scaling
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    else:
        scaler_obj = StandardScaler()

    steps.append(("scaler", scaler_obj))

    # Feature selection
    if feature_selection:
        selector = SelectKBest(score_func=f_regression, k=feature_selection)
        steps.append(("feature_selection", selector))

    return Pipeline(steps)


def create_baseline_pipeline(model_type: str = "linear", **pipeline_kwargs) -> Pipeline:
    """
    Create a baseline forecasting pipeline.

    Args:
        model_type: Type of baseline model ('linear', 'ridge', 'lasso', 'random_forest')
        **pipeline_kwargs: Additional arguments for pipeline creation

    Returns:
        Complete baseline pipeline
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor

    # Create preprocessing pipeline
    preprocessing_pipeline = create_time_series_pipeline(**pipeline_kwargs)

    # Add model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        model = Lasso(alpha=1.0)
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    # Combine preprocessing and model
    full_pipeline = Pipeline([("preprocessing", preprocessing_pipeline), ("model", model)])

    return full_pipeline
