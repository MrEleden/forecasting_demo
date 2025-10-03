"""
Dataset Factory for time series data.

Handles:
1. Loading raw data
2. Temporal splitting (no data leakage)
3. Creating dataset instances

NO preprocessing happens here - following Burkov's principle.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_portfolio.data.datasets import TimeSeriesDataset


class DatasetFactory:
    """
    Factory that handles:
    1. Loading raw data
    2. Static feature engineering (BEFORE split - safe, deterministic)
    3. Temporal splitting (no data leakage)
    4. Creating dataset instances

    Statistical preprocessing happens AFTER in train.py - following Burkov's principle.
    """

    def __init__(
        self,
        data_path: str,
        target_column: str,
        feature_columns: list = None,
        timestamp_column: str = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        static_feature_engineer=None,
        **kwargs,
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.timestamp_column = timestamp_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.static_feature_engineer = static_feature_engineer
        self.kwargs = kwargs

    def create_datasets(self) -> tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """
        Create train/val/test splits with optional static feature engineering.

        Static feature engineering (deterministic, backward-looking) is applied
        BEFORE splitting - this is safe and prevents data leakage.

        Statistical preprocessing happens AFTER splitting in train.py.
        """
        # Load raw data
        df = self._load_raw_data()

        # Apply static feature engineering BEFORE splitting (safe, no leakage)
        if self.static_feature_engineer is not None:
            df = self.static_feature_engineer.engineer_features(df)

        # Extract features and targets (no transformation)
        X, y, timestamps, feature_names = self._extract_arrays(df)

        # Temporal split (respecting time order - no leakage)
        train_idx, val_idx, test_idx = self._get_temporal_indices(len(X))

        # Create dataset instances
        train_dataset = TimeSeriesDataset(
            X=X[train_idx],
            y=y[train_idx],
            timestamps=timestamps[train_idx] if timestamps is not None else None,
            feature_names=feature_names,
            metadata={"split": "train", "source": self.data_path},
        )

        val_dataset = TimeSeriesDataset(
            X=X[val_idx],
            y=y[val_idx],
            timestamps=timestamps[val_idx] if timestamps is not None else None,
            feature_names=feature_names,
            metadata={"split": "validation", "source": self.data_path},
        )

        test_dataset = TimeSeriesDataset(
            X=X[test_idx],
            y=y[test_idx],
            timestamps=timestamps[test_idx] if timestamps is not None else None,
            feature_names=feature_names,
            metadata={"split": "test", "source": self.data_path},
        )

        return train_dataset, val_dataset, test_dataset

    def _load_raw_data(self) -> pd.DataFrame:
        """Load data without any processing."""
        path = Path(self.data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return df

    def _extract_arrays(self, df: pd.DataFrame) -> tuple:
        """Extract numpy arrays from dataframe - no transformations."""
        # Get target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        y = df[self.target_column].values

        # Get timestamps if specified
        timestamps = None
        if self.timestamp_column and self.timestamp_column in df.columns:
            timestamps = df[self.timestamp_column].values

        # Get features
        if self.feature_columns:
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
            X = df[self.feature_columns].values
            feature_names = self.feature_columns
        else:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != self.target_column]
            X = df[feature_cols].values
            feature_names = feature_cols

        return X.astype(np.float32), y.astype(np.float32), timestamps, feature_names

    def _get_temporal_indices(self, n_samples: int) -> tuple:
        """Get indices for temporal split - no shuffling."""
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_samples)

        return train_idx, val_idx, test_idx
