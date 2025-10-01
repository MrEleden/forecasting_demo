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

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """
    Base PyTorch-compatible Dataset for time series forecasting.

    This is the MAIN PARENT CLASS that all project-specific datasets inherit from.
    Child classes override load_data() and preprocess_data() for domain-specific logic.

    PyTorch Compatibility:
    - Implements __len__() and __getitem__() for use with torch.utils.data.DataLoader
    - Returns numpy arrays compatible with torch.from_numpy()
    - Works seamlessly with both PyTorch and sklearn models

    Key Methods Child Classes Override:
    1. load_data(data_path) -> pd.DataFrame
       - Load raw data from file
       - Handle domain-specific file formats

    2. preprocess_data(df) -> pd.DataFrame
       - Domain-specific preprocessing
       - Feature engineering
       - Data cleaning

    Example Child Class:
        class WalmartDataset(TimeSeriesDataset):
            def load_data(self, data_path: str) -> pd.DataFrame:
                df = pd.read_csv(data_path)
                # Walmart-specific loading logic
                return df

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
        lookback_window: int = 24,
        forecast_horizon: int = 1,
        stride: int = 1,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        mode: str = "train",  # NEW: train, val, or test mode
        **kwargs,
    ):
        """
        Initialize TimeSeriesDataset.

        Args:
            data_path: Path to data file (if None, child class must handle)
            target_column: Name of column to predict
            feature_columns: List of feature column names (None = all except target)
            lookback_window: Number of past timesteps for input
            forecast_horizon: Number of future timesteps to predict
            stride: Step size between sequences
            train_ratio: Proportion of data for training
            validation_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            mode: Dataset mode - 'train', 'val', or 'test'
            **kwargs: Additional parameters for child classes
        """
        # Store configuration
        self.data_path = data_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.mode = mode  # NEW: train/val/test mode
        self.kwargs = kwargs

        # Data containers (populated by load())
        self.raw_data = None
        self.processed_data = None
        self.sequences = None
        self.X = None  # Feature matrix
        self.y = None  # Target vector

        # Split indices (populated after load())
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # Metadata
        self.feature_names = []
        self.data_info = {}

        logger.info(f"Initialized {self.__class__.__name__}")

    def load(self) -> "TimeSeriesDataset":
        """
        Main entry point: Load and prepare dataset.

        This method orchestrates the full data pipeline:
        1. load_data() -> Load raw data
        2. preprocess_data() -> Clean and engineer features
        3. _extract_features_targets() -> Separate X and y
        4. _create_sequences() -> Create windowed sequences (if needed)

        Returns:
            self (for method chaining)
        """
        logger.info(f"Loading data for {self.__class__.__name__}...")

        # Step 1: Load raw data
        if self.data_path:
            self.raw_data = self.load_data(self.data_path)
            logger.info(
                f"Loaded raw data: {self.raw_data.shape if hasattr(self.raw_data, 'shape') else len(self.raw_data)} samples"
            )
        else:
            logger.warning("No data_path provided. Child class must handle data loading.")

        # Step 2: Preprocess
        if self.raw_data is not None:
            self.processed_data = self.preprocess_data(self.raw_data)
            logger.info(f"Preprocessed data: {self.processed_data.shape}")

        # Step 3: Extract features and targets
        if self.processed_data is not None:
            self.X, self.y, self.feature_names = self._extract_features_targets(self.processed_data)
            logger.info(f"Features: {self.X.shape}, Targets: {self.y.shape}")
            logger.info(f"Feature names: {self.feature_names}")

        # Step 4: Create sequences (optional, for sequence-to-sequence models)
        # For now, keep it simple: each row is a sample
        # Child classes can override to create sliding windows

        # Step 5: Compute split indices for train/val/test modes
        self._compute_split_indices()

        logger.info(f"Dataset ready: {len(self)} samples in '{self.mode}' mode")
        return self

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load raw data from file.

        **OVERRIDE THIS in child classes for domain-specific loading.**

        Args:
            data_path: Path to data file

        Returns:
            Raw data as DataFrame

        Example:
            class WalmartDataset(TimeSeriesDataset):
                def load_data(self, data_path: str) -> pd.DataFrame:
                    df = pd.read_csv(data_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values(['Store', 'Dept', 'Date'])
                    return df
        """
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Default loading logic
        if path.suffix == ".csv":
            return pd.read_csv(data_path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        elif path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

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

    def _compute_split_indices(self):
        """Compute train/val/test split indices based on ratios."""
        if self.X is None:
            return

        n_samples = len(self.X)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))

        self.train_indices = np.arange(0, train_end)
        self.val_indices = np.arange(train_end, val_end)
        self.test_indices = np.arange(val_end, n_samples)

        logger.info(
            f"Split indices computed: train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}"
        )

    def set_mode(self, mode: str):
        """
        Set dataset mode (train/val/test).

        Args:
            mode: One of 'train', 'val', or 'test'
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{mode}'")
        self.mode = mode
        logger.info(f"Dataset mode set to: {mode}")

    def _get_active_indices(self) -> np.ndarray:
        """Get indices for current mode (train/val/test)."""
        if self.train_indices is None:
            self._compute_split_indices()

        if self.mode == "train":
            return self.train_indices
        elif self.mode == "val":
            return self.val_indices
        else:  # test
            return self.test_indices

    def __len__(self) -> int:
        """Return number of samples in current mode (PyTorch DataLoader compatible)."""
        if self.X is not None:
            indices = self._get_active_indices()
            return len(indices)
        return 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample in current mode (PyTorch DataLoader compatible).

        Args:
            idx: Sample index (relative to current mode)

        Returns:
            Tuple of (features, target)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not loaded. Call dataset.load() first.")

        # Map idx to actual data index based on current mode
        indices = self._get_active_indices()
        actual_idx = indices[idx]

        return self.X[actual_idx], self.y[actual_idx]

    def get_splits(
        self, train_ratio: Optional[float] = None, val_ratio: Optional[float] = None, test_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get temporal train/val/test splits.

        Args:
            train_ratio: Training proportion (uses self.train_ratio if None)
            val_ratio: Validation proportion (uses self.validation_ratio if None)
            test_ratio: Test proportion (uses self.test_ratio if None)

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not loaded. Call dataset.load() first.")

        # Use provided ratios or defaults
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        val_ratio = val_ratio if val_ratio is not None else self.validation_ratio
        test_ratio = test_ratio if test_ratio is not None else self.test_ratio

        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            logger.warning(f"Split ratios don't sum to 1.0: {train_ratio + val_ratio + test_ratio}. Normalizing...")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        # Calculate split indices (temporal order preserved)
        n_samples = len(self.X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Split data
        X_train, y_train = self.X[:train_end], self.y[:train_end]
        X_val, y_val = self.X[train_end:val_end], self.y[train_end:val_end]
        X_test, y_test = self.X[val_end:], self.y[val_end:]

        logger.info(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

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
            "lookback_window": self.lookback_window,
            "forecast_horizon": self.forecast_horizon,
            "split_ratios": {"train": self.train_ratio, "validation": self.validation_ratio, "test": self.test_ratio},
        }

    def to_pytorch(self):
        """
        Convert to PyTorch tensors (optional utility).

        Returns:
            Dataset compatible with torch.utils.data.DataLoader
        """
        try:
            import torch
            from torch.utils.data import TensorDataset

            if self.X is None or self.y is None:
                raise RuntimeError("Dataset not loaded. Call dataset.load() first.")

            X_tensor = torch.from_numpy(self.X).float()
            y_tensor = torch.from_numpy(self.y).float()

            return TensorDataset(X_tensor, y_tensor)
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")


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
            raise RuntimeError("Dataset not loaded. Call dataset.load() first.")

        mask = self.processed_data[self.series_column] == series_id
        series_data = self.processed_data[mask]

        X, y, _ = self._extract_features_targets(series_data)
        return X, y
