"""
PyTorch-friendly base Dataset class for time series forecasting.

This module provides a flexible base class that child classes can inherit from
to implement project-specific data loading, preprocessing, and feature engineering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
import logging

# Optional PyTorch imports
try:
    import torch
    from torch.utils.data import TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TensorDataset = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """
    Base PyTorch-compatible Dataset for time series forecasting.

    This is the MAIN PARENT CLASS that all project-specific datasets inherit from.
    Child classes override preprocess_data() for domain-specific logic.
    Data loading is handled by the DatasetFactory.

    PyTorch Compatibility:
    - Implements __len__() and __getitem__() for use with torch.utils.data.DataLoader
    - Returns numpy arrays compatible with torch.from_numpy()
    - Works seamlessly with both PyTorch and sklearn models

    Key Methods Child Classes Override:
    1. preprocess_data(df) -> pd.DataFrame
       - Domain-specific preprocessing
       - Feature engineering
       - Data cleaning

    Example Child Class:
        class WalmartDataset(TimeSeriesDataset):
            def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
                # Add holiday features
                df['is_holiday'] = self._add_holidays(df)
                # Add store embeddings
                df = self._add_store_features(df)
                return df
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize TimeSeriesDataset.

        Args:
            data_path: Path to data file (used by factory for loading)
            target_column: Name of column to predict
            feature_columns: List of feature column names (None = all except target)
            **kwargs: Additional parameters for child classes
        """
        # Store configuration
        self.data_path = data_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.kwargs = kwargs

        # Data containers (populated by factory)
        self.processed_data = None
        self.X = None  # Feature matrix
        self.y = None  # Target vector

        # Metadata
        self.feature_names = []

        logger.info(f"Initialized {self.__class__.__name__}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and engineer features.

        **OVERRIDE THIS in child classes for domain-specific preprocessing.**

        Args:
            df: Raw data DataFrame

        Returns:
            Preprocessed DataFrame with engineered features

        Example:
            class WalmartDataset(TimeSeriesDataset):
                def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
                    # Handle missing values
                    df = df.fillna(method='ffill')

                    # Add temporal features
                    df['week_of_year'] = df['Date'].dt.isocalendar().week
                    df['month'] = df['Date'].dt.month

                    # Add domain features
                    df['is_holiday'] = self._identify_holidays(df)
                    df['days_to_holiday'] = self._days_to_next_holiday(df)

                    return df
        """
        # Default: basic cleaning
        df = df.copy()

        # Handle missing values
        if df.isnull().any().any():
            logger.warning("Missing values detected. Filling with forward fill.")
            df = df.fillna(method="ffill").fillna(method="bfill")

        # Sort by time if date column exists
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            df = df.sort_values(date_cols[0]).reset_index(drop=True)

        return df

    def _extract_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix X and target vector y from preprocessed data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Identify target column
        if self.target_column:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
            y = df[self.target_column].values
        else:
            # Default: last column is target
            y = df.iloc[:, -1].values
            self.target_column = df.columns[-1]

        # Identify feature columns
        if self.feature_columns:
            # Use specified features
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
            X = df[self.feature_columns].values
            feature_names = self.feature_columns
        else:
            # Use all columns except target and non-numeric columns
            exclude_cols = [self.target_column]
            # Exclude date/time columns
            exclude_cols.extend([col for col in df.columns if "date" in col.lower() or "time" in col.lower()])

            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Only keep numeric columns
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                # If no features, create time index
                X = np.arange(len(df)).reshape(-1, 1)
                feature_names = ["time_index"]
            else:
                X = df[numeric_cols].values
                feature_names = numeric_cols

        return X, y, feature_names

    def __len__(self) -> int:
        """Return number of samples (PyTorch DataLoader compatible)."""
        if self.X is not None:
            return len(self.X)
        return 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample (PyTorch DataLoader compatible).

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not loaded. No data available.")

        return self.X[idx], self.y[idx]

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get dataset information.

        Returns:
            Dictionary with dataset metadata
        """
        return {
            "class": self.__class__.__name__,
            "n_samples": len(self),
            "n_features": self.X.shape[1] if self.X is not None and self.X.ndim > 1 else 0,
            "feature_names": self.feature_names,
            "target_column": self.target_column,
        }

    def to_pytorch(self):
        """
        Convert to PyTorch tensors (optional utility).

        Returns:
            Dataset compatible with torch.utils.data.DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not loaded. No data available.")

        X_tensor = torch.from_numpy(self.X).float()
        y_tensor = torch.from_numpy(self.y).float()

        return TensorDataset(X_tensor, y_tensor)


class MultiSeriesDataset(TimeSeriesDataset):
    """
    Base class for datasets with multiple time series (e.g., multiple stores, products, zones).

    Inherits from TimeSeriesDataset and adds multi-entity handling.

    Example:
        class WalmartMultiStoreDataset(MultiSeriesDataset):
            def __init__(self, data_path: str, **kwargs):
                super().__init__(
                    data_path=data_path,
                    series_column='Store',
                    target_column='Weekly_Sales',
                    **kwargs
                )
    """

    def __init__(self, series_column: str, **kwargs):
        """
        Initialize MultiSeriesDataset.

        Args:
            series_column: Column identifying different series (e.g., 'Store', 'Product', 'Zone')
            **kwargs: Arguments passed to TimeSeriesDataset
        """
        self.series_column = series_column
        self.series_ids = None
        self.n_series = 0

        super().__init__(**kwargs)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess multi-series data.

        Args:
            df: Raw data with multiple series

        Returns:
            Preprocessed DataFrame
        """
        # Identify unique series
        if self.series_column not in df.columns:
            raise ValueError(f"Series column '{self.series_column}' not found in data")

        self.series_ids = df[self.series_column].unique()
        self.n_series = len(self.series_ids)

        logger.info(f"Found {self.n_series} series in column '{self.series_column}'")

        # Sort by series and time
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            df = df.sort_values([self.series_column, date_cols[0]]).reset_index(drop=True)

        # Call parent preprocessing
        return super().preprocess_data(df)

    def get_series_data(self, series_id: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific series.

        Args:
            series_id: ID of the series to extract

        Returns:
            Tuple of (X, y) for that series
        """
        if self.processed_data is None:
            raise RuntimeError("Dataset not loaded. No data available.")

        mask = self.processed_data[self.series_column] == series_id
        series_data = self.processed_data[mask]

        X, y, _ = self._extract_features_targets(series_data)
        return X, y


class DatasetFactory:
    """
    Factory class for creating train/val/test dataset splits.

    This factory pattern allows Hydra to instantiate a single factory object
    that can then create the three dataset splits as needed.
    """

    def __init__(
        self,
        dataset_class: str,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        **dataset_kwargs,
    ):
        """
        Initialize the dataset factory.

        Args:
            dataset_class: The dataset class to instantiate (e.g., 'ml_portfolio.data.datasets.TimeSeriesDataset')
            train_ratio: Training split ratio
            validation_ratio: Validation split ratio
            test_ratio: Test split ratio
            **dataset_kwargs: Additional arguments to pass to the dataset constructor
        """
        self.dataset_class = dataset_class
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.dataset_kwargs = dataset_kwargs

        # Validate ratios
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + validation_ratio + test_ratio}")

    def create_datasets(self) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """
        Create train/validation/test dataset splits.

        **OVERRIDE THIS in child classes for domain-specific data loading and preprocessing.**

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)

        Example:
            class WalmartFactory(DatasetFactory):
                def create_datasets(self):
                    # Load raw data
                    df = self._load_raw_data()

                    # Apply domain-specific preprocessing
                    df = self._preprocess_walmart_data(df)

                    # Create time-based splits
                    train_df, val_df, test_df = self._split_data(df)

                    # Instantiate dataset objects
                    train_dataset = self._create_dataset_instance(train_df)
                    val_dataset = self._create_dataset_instance(val_df)
                    test_dataset = self._create_dataset_instance(test_df)

                    return train_dataset, val_dataset, test_dataset
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_datasets() method. "
            "This method should load data, apply preprocessing, create splits, "
            "and return (train_dataset, val_dataset, test_dataset)."
        )
