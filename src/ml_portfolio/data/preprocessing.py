"""
Preprocessing pipelines for time series forecasting.

Handles all statistical preprocessing following Burkov's principle:
Fit on train, transform all splits.
"""

import numpy as np
import pandas as pd

from ml_portfolio.data.datasets import TimeSeriesDataset


class StatisticalPreprocessingPipeline:
    """
    Handles all statistical preprocessing following Burkov's principle:
    Fit on train, transform all splits.
    """

    def __init__(self, steps=None):
        """
        Args:
            steps: List of (name, transformer) tuples
        """
        self.steps = steps or []
        self.is_fitted = False

    def fit(self, dataset: TimeSeriesDataset):
        """Fit all transformers on training data."""
        X, y = dataset.get_data()

        for name, transformer in self.steps:
            if name.startswith("target_"):
                # Target transformers
                if hasattr(transformer, "fit"):
                    transformer.fit(y.reshape(-1, 1))
            else:
                # Feature transformers
                if hasattr(transformer, "fit"):
                    transformer.fit(X)

        self.is_fitted = True
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform dataset using fitted preprocessors."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        X, y = dataset.get_data()

        for name, transformer in self.steps:
            if name.startswith("target_"):
                y = transformer.transform(y.reshape(-1, 1)).ravel()
            else:
                X = transformer.transform(X)

        # Return new dataset with transformed data
        return TimeSeriesDataset(
            X=X,
            y=y,
            timestamps=dataset.timestamps,
            feature_names=dataset.feature_names,
            metadata={**dataset.metadata, "preprocessed": True},
        )

    def fit_transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """Fit and transform in one step."""
        return self.fit(dataset).transform(dataset)


class StaticTimeSeriesPreprocessingPipeline:
    """
    Reusable time series feature engineering.
    These are static transformations safe to apply before splitting.
    """

    def __init__(
        self,
        date_column: str = None,
        group_columns: list = None,
        target_column: str = None,
        lag_features: list = None,
        rolling_windows: list = None,
        date_features: bool = True,
        cyclical_features: list = None,
    ):
        self.date_column = date_column
        self.group_columns = group_columns or []
        self.target_column = target_column
        self.lag_features = lag_features or [1, 7, 14]
        self.rolling_windows = rolling_windows or [7, 30]
        self.date_features = date_features
        self.cyclical_features = cyclical_features or []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time series features to dataframe.
        All transformations are backward-looking or deterministic (safe before split).
        """
        df = df.copy()

        # Date features (deterministic - safe)
        if self.date_column and self.date_column in df.columns:
            df = self._add_date_features(df)

        # Lag features (backward-looking - safe)
        if self.target_column and self.lag_features:
            df = self._add_lag_features(df)

        # Rolling features (backward-looking - safe)
        if self.target_column and self.rolling_windows:
            df = self._add_rolling_features(df)

        # Cyclical encoding (deterministic - safe)
        if self.cyclical_features:
            df = self._add_cyclical_features(df)

        return df

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add deterministic date features."""
        date_col = pd.to_datetime(df[self.date_column])

        if self.date_features:
            df["year"] = date_col.dt.year
            df["month"] = date_col.dt.month
            df["day"] = date_col.dt.day
            df["dayofweek"] = date_col.dt.dayofweek
            df["quarter"] = date_col.dt.quarter
            df["week"] = date_col.dt.isocalendar().week
            df["is_weekend"] = (date_col.dt.dayofweek >= 5).astype(int)
            df["is_month_start"] = date_col.dt.is_month_start.astype(int)
            df["is_month_end"] = date_col.dt.is_month_end.astype(int)
            df["is_quarter_start"] = date_col.dt.is_quarter_start.astype(int)
            df["is_quarter_end"] = date_col.dt.is_quarter_end.astype(int)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features (backward-looking only)."""
        for lag in self.lag_features:
            if self.group_columns:
                # Group-wise lags (e.g., per store)
                for group_col in self.group_columns:
                    df[f"{self.target_column}_lag_{lag}_{group_col}"] = df.groupby(group_col)[self.target_column].shift(
                        lag
                    )
            else:
                # Global lags
                df[f"{self.target_column}_lag_{lag}"] = df[self.target_column].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features (backward-looking only)."""
        for window in self.rolling_windows:
            if self.group_columns:
                # Group-wise rolling features
                for group_col in self.group_columns:
                    # Shift by 1 to avoid leakage
                    df[f"{self.target_column}_rolling_mean_{window}_{group_col}"] = df.groupby(group_col)[
                        self.target_column
                    ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    df[f"{self.target_column}_rolling_std_{window}_{group_col}"] = df.groupby(group_col)[
                        self.target_column
                    ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            else:
                # Global rolling features
                shifted = df[self.target_column].shift(1)
                df[f"{self.target_column}_rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
                df[f"{self.target_column}_rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std()

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for periodic features."""
        for col in self.cyclical_features:
            if col in df.columns:
                max_val = df[col].max()
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)

        return df
