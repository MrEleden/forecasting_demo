"""
PyTorch Dataset wrappers for time series forecasting.

This module provides base Dataset classes that projects should inherit from for domain-specific customization.

Project-specific inheritance pattern:
    - WalmartTimeSeriesDataset(TimeSeriesDataset)
    - OlaRideShareDataset(TimeSeriesDataset)
    - InventoryDataset(TimeSeriesDataset)
    - TSIDataset(TimeSeriesDataset)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List, Dict, Any


class TimeSeriesDataset:
    """
    Base PyTorch-style Dataset for time series forecasting.

    This is a BASE CLASS that projects should inherit from for domain-specific customization.

    Project-specific inheritance pattern:
        class WalmartTimeSeriesDataset(TimeSeriesDataset):
            def __init__(self, data_path: str, **kwargs):
                # Load Walmart-specific data
                data = pd.read_csv(data_path)
                super().__init__(data, **kwargs)

            def _preprocess_walmart_data(self, data):
                # Walmart-specific preprocessing (holidays, promotions)
                return data

            def __getitem__(self, idx):
                # Add Walmart-specific features to samples
                X, y = super().__getitem__(idx)
                X = self._add_holiday_features(X, idx)
                return X, y

    Methods that projects commonly override:
        - __init__(): Load domain-specific data
        - _preprocess_data(): Add domain-specific preprocessing
        - __getitem__(): Add domain-specific features to samples
        - _create_sequences(): Customize windowing for domain
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_column: str = None,
        lookback_window: int = 24,
        forecast_horizon: int = 1,
        stride: int = 1,
        include_features: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize TimeSeriesDataset.

        Args:
            data: Time series data (DataFrame or array)
            target_column: Name of target column (for DataFrames)
            lookback_window: Number of past observations to use
            forecast_horizon: Number of future observations to predict
            stride: Step size between sequences
            include_features: List of feature columns to include
            **kwargs: Additional parameters for project-specific datasets
        """
        self.data = data
        self.target_column = target_column
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.include_features = include_features

        # Preprocess data (can be overridden by projects)
        self.processed_data = self._preprocess_data(data)

        # Create sequences
        self.sequences = self._create_sequences()

    def _preprocess_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess data before creating sequences.
        Override in project-specific classes for domain-specific preprocessing.

        Args:
            data: Raw data

        Returns:
            Preprocessed data as numpy array
        """
        if isinstance(data, pd.DataFrame):
            if self.target_column:
                target = data[self.target_column].values
                if self.include_features:
                    features = data[self.include_features].values
                    return np.column_stack([target.reshape(-1, 1), features])
                else:
                    return target.reshape(-1, 1)
            else:
                return data.values
        else:
            return np.array(data)

    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create sequences from processed data.
        Override in project-specific classes for custom windowing strategies.

        Returns:
            List of (input_sequence, target_sequence) tuples
        """
        sequences = []
        data = self.processed_data

        for i in range(0, len(data) - self.lookback_window - self.forecast_horizon + 1, self.stride):
            # Input sequence (past observations)
            X = data[i : i + self.lookback_window]

            # Target sequence (future observations)
            y = data[i + self.lookback_window : i + self.lookback_window + self.forecast_horizon]

            # For single-step prediction, flatten target
            if self.forecast_horizon == 1:
                y = y.ravel()

            sequences.append((X, y))

        return sequences

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sequence by index.
        Override in project-specific classes to add domain-specific features.

        Args:
            idx: Sequence index

        Returns:
            (input_sequence, target_sequence) tuple
        """
        return self.sequences[idx]

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset information
        """
        return {
            "num_sequences": len(self),
            "lookback_window": self.lookback_window,
            "forecast_horizon": self.forecast_horizon,
            "stride": self.stride,
            "input_shape": self.sequences[0][0].shape if self.sequences else None,
            "target_shape": self.sequences[0][1].shape if self.sequences else None,
            "data_shape": self.processed_data.shape,
        }


class MultiSeriesDataset(TimeSeriesDataset):
    """
    Base Dataset for multiple time series (e.g., multiple stores, products).

    This is a BASE CLASS for project-specific multi-series implementations.

    Project-specific inheritance examples:
        class WalmartMultiStoreDataset(MultiSeriesDataset):
            def __init__(self, data_path: str, **kwargs):
                data = pd.read_csv(data_path)
                super().__init__(data, series_column='Store', **kwargs)

            def _add_store_features(self, X, series_id, sequence_idx):
                # Add store-specific features (size, location, demographics)
                return X

        class OlaMultiZoneDataset(MultiSeriesDataset):
            def __init__(self, data_path: str, **kwargs):
                data = pd.read_csv(data_path)
                super().__init__(data, series_column='pickup_zone', **kwargs)
    """

    def __init__(self, data: pd.DataFrame, series_column: str, target_column: str = None, **kwargs):
        """
        Initialize MultiSeriesDataset.

        Args:
            data: Time series data with multiple series
            series_column: Column identifying different series (e.g., 'Store', 'Product')
            target_column: Name of target column
            **kwargs: Additional parameters passed to parent class
        """
        self.series_column = series_column
        self.series_ids = data[series_column].unique()

        # Store original data
        self.original_data = data

        # Initialize with first series (will be updated in _create_sequences)
        first_series_data = data[data[series_column] == self.series_ids[0]]
        super().__init__(first_series_data, target_column=target_column, **kwargs)

    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Create sequences from all series.

        Returns:
            List of (input_sequence, target_sequence, series_id) tuples
        """
        all_sequences = []

        for series_id in self.series_ids:
            # Get data for this series
            series_data = self.original_data[self.original_data[self.series_column] == series_id]

            # Preprocess series data
            processed_series = self._preprocess_data(series_data)

            # Create sequences for this series
            for i in range(0, len(processed_series) - self.lookback_window - self.forecast_horizon + 1, self.stride):
                X = processed_series[i : i + self.lookback_window]
                y = processed_series[i + self.lookback_window : i + self.lookback_window + self.forecast_horizon]

                if self.forecast_horizon == 1:
                    y = y.ravel()

                all_sequences.append((X, y, series_id))

        return all_sequences

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get a single sequence by index.

        Args:
            idx: Sequence index

        Returns:
            (input_sequence, target_sequence, series_id) tuple
        """
        return self.sequences[idx]


class SlidingWindowDataset(TimeSeriesDataset):
    """
    Base Dataset with sliding window approach for online learning.

    This is a BASE CLASS for project-specific sliding window implementations.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray], window_size: int = 100, **kwargs):
        """
        Initialize SlidingWindowDataset.

        Args:
            window_size: Size of sliding window for training
            **kwargs: Additional parameters passed to parent class
        """
        self.window_size = window_size
        super().__init__(data, **kwargs)

    def get_window(self, start_idx: int) -> "TimeSeriesDataset":
        """
        Get a sliding window as a new dataset.

        Args:
            start_idx: Starting index for the window

        Returns:
            New TimeSeriesDataset with windowed data
        """
        end_idx = min(start_idx + self.window_size, len(self.processed_data))
        windowed_data = self.processed_data[start_idx:end_idx]

        return TimeSeriesDataset(
            windowed_data,
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon,
            stride=self.stride,
        )


# Factory Functions for Project-Specific Dataset Creation


def create_project_dataset_class(project_name: str, dataset_type: str = "single", base_class=None):
    """
    Factory function to create project-specific dataset classes following naming conventions.

    This function creates classes that follow the pattern: {ProjectName}{DatasetType}Dataset

    Args:
        project_name: Name of the project (e.g., "Walmart", "Ola", "Inventory", "TSI")
        dataset_type: Type of dataset ("TimeSeries", "MultiSeries", "SlidingWindow")
        base_class: Base class to inherit from (auto-detected if None)

    Returns:
        Project-specific dataset class

    Example:
        WalmartTimeSeriesDataset = create_project_dataset_class("Walmart", "TimeSeries")
        OlaMultiSeriesDataset = create_project_dataset_class("Ola", "MultiSeries")
    """
    if base_class is None:
        if dataset_type.lower() == "timeseries" or dataset_type.lower() == "single":
            base_class = TimeSeriesDataset
        elif dataset_type.lower() == "multiseries" or dataset_type.lower() == "multi":
            base_class = MultiSeriesDataset
        elif dataset_type.lower() == "slidingwindow" or dataset_type.lower() == "sliding":
            base_class = SlidingWindowDataset
        else:
            base_class = TimeSeriesDataset

    class_name = f"{project_name}{dataset_type}Dataset"

    class ProjectDatasetClass(base_class):
        """Dynamically created project-specific dataset class."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.project_name = project_name
            self.dataset_type = dataset_type

        def __repr__(self):
            return f"{class_name}(project={self.project_name}, type={self.dataset_type}, num_sequences={len(self)})"

    ProjectDatasetClass.__name__ = class_name
    ProjectDatasetClass.__qualname__ = class_name

    return ProjectDatasetClass


def get_project_dataset_params(project_name: str) -> Dict[str, Any]:
    """
    Get recommended dataset parameters for specific projects.

    Args:
        project_name: Project name

    Returns:
        Dictionary of recommended parameters
    """
    params = {}

    if project_name.lower() == "walmart":
        params = {
            "lookback_window": 52,  # 52 weeks of history
            "forecast_horizon": 1,  # 1 week ahead
            "stride": 1,
            "target_column": "Weekly_Sales",
        }

    elif project_name.lower() == "ola" or project_name.lower() == "rideshare":
        params = {
            "lookback_window": 24,  # 24 hours of history
            "forecast_horizon": 1,  # 1 hour ahead
            "stride": 1,
            "target_column": "demand",
        }

    elif project_name.lower() == "inventory":
        params = {
            "lookback_window": 30,  # 30 days of history
            "forecast_horizon": 7,  # 7 days ahead
            "stride": 1,
            "target_column": "demand",
        }

    elif project_name.lower() == "tsi" or project_name.lower() == "transportation":
        params = {
            "lookback_window": 12,  # 12 months of history
            "forecast_horizon": 1,  # 1 month ahead
            "stride": 1,
            "target_column": "value",
        }

    return params
