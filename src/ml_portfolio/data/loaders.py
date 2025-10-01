"""
DataLoader factories with time series sampling strategies.

This module provides utilities for creating DataLoaders optimized for
time series forecasting tasks.
"""

from typing import Optional, Union

try:
    import torch
    from torch.utils.data import DataLoader, Sampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TimeSeriesSampler:
    """
    Custom sampler for time series data that respects temporal order.
    """

    def __init__(self, dataset_size: int, shuffle: bool = False):
        """
        Initialize TimeSeriesSampler.

        Args:
            dataset_size: Size of the dataset
            shuffle: Whether to shuffle the data (usually False for time series)
        """
        self.dataset_size = dataset_size
        self.shuffle = shuffle

    def __iter__(self):
        """Iterate over indices."""
        if self.shuffle:
            # For time series, we might want controlled shuffling
            # that preserves temporal relationships
            indices = list(range(self.dataset_size))
        else:
            indices = list(range(self.dataset_size))

        return iter(indices)

    def __len__(self):
        """Return dataset size."""
        return self.dataset_size


def create_time_series_dataloader(dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0, **kwargs):
    """
    Create a DataLoader optimized for time series forecasting.

    Args:
        dataset: Time series dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle data (typically False for time series)
        num_workers: Number of worker processes for data loading
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader functionality")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)


def create_train_val_dataloaders(train_dataset, val_dataset, batch_size: int = 32, **kwargs):
    """
    Create train and validation DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        **kwargs: Additional DataLoader arguments

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = create_time_series_dataloader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = create_time_series_dataloader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader
