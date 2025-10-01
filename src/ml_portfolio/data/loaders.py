"""
DataLoader factories with time series sampling strategies.

This module provides utilities for creating DataLoaders optimized for
time series forecasting tasks with proper PyTorch integration.
"""

from typing import Optional, Union, Iterator, List
import random


class DataLoader:
    """Simple DataLoader implementation without PyTorch dependency."""
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, 
                 sampler=None, drop_last: bool = False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.drop_last = drop_last
    
    def __iter__(self):
        if self.sampler:
            indices = list(self.sampler)
        elif self.shuffle:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))
        
        # Create batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last=True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
                
            batch_data = []
            batch_targets = []
            
            for idx in batch_indices:
                data, target = self.dataset[idx]
                batch_data.append(data)
                batch_targets.append(target)
            
            yield batch_data, batch_targets
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class Sampler:
    """Base sampler class without PyTorch dependency."""
    
    def __init__(self, data_source=None):
        self.data_source = data_source
    
    def __iter__(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError


class TimeSeriesSampler(Sampler):
    """
    Custom sampler for time series data that respects temporal order.
    Inherits from torch.utils.data.Sampler for full PyTorch compatibility.
    """

    def __init__(self, dataset_size: int, shuffle: bool = False, block_size: Optional[int] = None):
        """
        Initialize TimeSeriesSampler.

        Args:
            dataset_size: Size of the dataset
            shuffle: Whether to shuffle the data (usually False for time series)
            block_size: Size of blocks for block-wise shuffling (preserves local order)
        """
        super().__init__(None)  # Pass None as data_source since we handle indices manually
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.block_size = block_size

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices maintaining temporal relationships."""
        if self.shuffle:
            if self.block_size:
                # Block-wise shuffling: shuffle blocks but maintain order within blocks
                indices = list(range(self.dataset_size))
                blocks = [indices[i : i + self.block_size] for i in range(0, len(indices), self.block_size)]
                random.shuffle(blocks)
                shuffled_indices = [idx for block in blocks for idx in block]
                return iter(shuffled_indices)
            else:
                # Regular shuffling (not recommended for time series)
                indices = list(range(self.dataset_size))
                random.shuffle(indices)
                return iter(indices)
        else:
            # Sequential order (recommended for time series)
            return iter(range(self.dataset_size))

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size


def create_time_series_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    block_shuffle_size: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader optimized for time series forecasting.

    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data (False recommended for time series)
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        block_shuffle_size: Size for block-wise shuffling (preserves local temporal order)
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured PyTorch DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader functionality")

    # Use custom sampler if shuffle is requested with block size
    if shuffle and block_shuffle_size:
        sampler = TimeSeriesSampler(dataset_size=len(dataset), shuffle=True, block_size=block_shuffle_size)
        # Don't pass shuffle=True when using custom sampler
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )
    else:
        # Standard DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    return dataloader


class TimeSeriesDataLoaderFactory:
    """
    Factory class for creating different types of time series DataLoaders.
    """

    @staticmethod
    def create_train_loader(dataset, batch_size: int = 32, **kwargs) -> DataLoader:
        """Create DataLoader for training with time series considerations."""
        return create_time_series_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep temporal order for training
            drop_last=True,  # Drop last incomplete batch
            **kwargs,
        )

    @staticmethod
    def create_val_loader(dataset, batch_size: int = 32, **kwargs) -> DataLoader:
        """Create DataLoader for validation."""
        return create_time_series_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle validation data
            drop_last=False,  # Keep all validation data
            **kwargs,
        )

    @staticmethod
    def create_test_loader(dataset, batch_size: int = 1, **kwargs) -> DataLoader:
        """Create DataLoader for testing/inference."""
        return create_time_series_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle test data
            drop_last=False,  # Keep all test data
            **kwargs,
        )


def create_train_val_test_loaders(train_dataset, val_dataset, test_dataset, batch_size: int = 32, **kwargs):
    """
    Convenience function to create train, validation, and test DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        **kwargs: Additional DataLoader arguments

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    factory = TimeSeriesDataLoaderFactory()

    train_loader = factory.create_train_loader(train_dataset, batch_size, **kwargs)
    val_loader = factory.create_val_loader(val_dataset, batch_size, **kwargs)
    test_loader = factory.create_test_loader(test_dataset, batch_size=1, **kwargs)

    return train_loader, val_loader, test_loader
