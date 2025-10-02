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
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize TimeSeriesDataset.

        Args:
            target_column: Name of column to predict
            feature_columns: List of feature column names (None = all except target)
            **kwargs: Additional parameters for child classes
        """
        # Store configuration

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
            def __init__(self, **kwargs):
                super().__init__(
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
        data_path: str,
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
        # Dynamically import and resolve the dataset class
        if isinstance(dataset_class, str):
            # Import the class from string
            module_path, class_name = dataset_class.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            self.dataset_class = getattr(module, class_name)
        else:
            self.dataset_class = dataset_class

        self.data_path = data_path
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.dataset_kwargs = dataset_kwargs

        # Validate ratios
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + validation_ratio + test_ratio}")

    def create_datasets(self) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """
        Create train/validation/test dataset splits using the 5-step factory pattern.

        This method orchestrates the standard 5-step process:
        1. Load raw data (_step1_load_data)
        2. Preprocess data (_step2_preprocess_data)
        3. Extract features and targets (_step3_extract_features_targets)
        4. Split data temporally (_step4_split_data)
        5. Create dataset instances (_step5_create_dataset_instances)

        Child classes should override individual step methods as needed for domain-specific logic.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)

        Example:
            class WalmartFactory(DatasetFactory):
                def _step2_preprocess_data(self, raw_data):
                    # Custom Walmart preprocessing
                    return self.preprocess_walmart_data(raw_data)

                def _step4_split_data(self, X, y):
                    # Custom time-based splitting for Walmart
                    return self.walmart_temporal_split(X, y)
        """
        # Step 1: Load raw data
        raw_data = self._load_data()

        # Step 2: Preprocess with domain-specific logic
        processed_data = self._preprocess_data(raw_data)

        # Step 3: Extract features and targets
        X, y, feature_names = self._extract_features_targets(processed_data)

        # Step 4: Split data temporally
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(X, y)

        # Step 5: Create TimeSeriesDataset instances
        train_dataset, val_dataset, test_dataset = self._create_dataset_instances(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names, processed_data
        )

        return train_dataset, val_dataset, test_dataset

    def _load_data(self):
        """
        Step 1: Load raw data.

        **OVERRIDE THIS in child classes for domain-specific data loading.**

        Returns:
            Raw data (typically pandas DataFrame)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _load_data() method.")

    def _preprocess_data(self, raw_data):
        """
        Step 2: Preprocess and engineer features.

        **OVERRIDE THIS in child classes for domain-specific preprocessing.**

        Args:
            raw_data: Raw data from step 1

        Returns:
            Preprocessed data with engineered features
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _preprocess_data() method.")

    def _extract_features_targets(self, processed_data):
        """
        Step 3: Extract feature matrix X and target vector y.

        **OVERRIDE THIS in child classes for domain-specific feature extraction.**

        Args:
            processed_data: Preprocessed data from step 2

        Returns:
            Tuple of (X, y, feature_names)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _extract_features_targets() method.")

    def _split_data(self, X, y):
        """
        Step 4: Split data into train/validation/test sets.

        Default implementation: Temporal split preserving chronological order.
        Child classes can override this for domain-specific splitting logic.

        Args:
            X: Feature matrix from step 3
            y: Target vector from step 3

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n_samples = len(X)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _create_dataset_instances(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, processed_data):
        """
        Step 5: Create TimeSeriesDataset instances for each split.

        Default implementation: Creates standard TimeSeriesDataset instances.
        Child classes can override this for custom dataset instance creation.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            feature_names: List of feature names
            processed_data: Original processed data

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        target_column = self.dataset_kwargs.get("target_column")

        # Create train dataset
        train_dataset = self.dataset_class(target_column=target_column)
        train_dataset.X = X_train
        train_dataset.y = y_train
        train_dataset.feature_names = feature_names
        train_dataset.processed_data = processed_data

        # Create validation dataset
        val_dataset = self.dataset_class(target_column=target_column)
        val_dataset.X = X_val
        val_dataset.y = y_val
        val_dataset.feature_names = feature_names
        val_dataset.processed_data = processed_data

        # Create test dataset
        test_dataset = self.dataset_class(target_column=target_column)
        test_dataset.X = X_test
        test_dataset.y = y_test
        test_dataset.feature_names = feature_names
        test_dataset.processed_data = processed_data

        return train_dataset, val_dataset, test_dataset
