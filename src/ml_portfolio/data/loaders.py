"""
DataLoader implementations for time series forecasting.

This module provides a unified interface for batch data loading with both
PyTorch and non-PyTorch implementations.

Architecture:
- BaseDataLoader: Master abstract class defining the interface
- PyTorchDataLoader: PyTorch-based implementation (when torch available)
- SimpleDataLoader: NumPy-based implementation (no dependencies)
"""

from typing import Optional, Iterator, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseDataLoader(ABC):
    """
    Master abstract class for all DataLoader implementations.

    Defines the standard interface that all loaders must implement.
    Can be instantiated via Hydra with _target_ pointing to specific implementations.

    Usage with Hydra:
        # In config.yaml
        dataloader:
            _target_: ml_portfolio.data.loaders.SimpleDataLoader
            batch_size: 32
            shuffle: true
    """

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = False, **kwargs):
        """
        Initialize base dataloader.

        Args:
            dataset: Dataset with __len__ and __getitem__ methods
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            **kwargs: Additional parameters for specific implementations
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over batches.

        Yields:
            Tuple of (X_batch, y_batch) as numpy arrays
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return number of batches.

        Returns:
            Number of batches in the dataset
        """
        pass


class SimpleDataLoader(BaseDataLoader):
    """
    Simple NumPy-based DataLoader for sklearn-style models.

    This loader returns ALL data in a single batch (infinite batch size).
    Perfect for sklearn models that expect fit(X, y) with complete dataset.

    Features:
    - No dependencies (pure NumPy)
    - Single batch = entire dataset
    - Optional shuffling of indices
    - Always works, no special requirements

    Usage:
        loader = SimpleDataLoader(dataset, shuffle=True)
        for X_batch, y_batch in loader:
            # X_batch and y_batch contain ALL data as numpy arrays
            model.fit(X_batch, y_batch)

    Hydra Config:
        dataloader:
            _target_: ml_portfolio.data.loaders.SimpleDataLoader
            shuffle: true
    """

    def __init__(
        self,
        dataset,
        batch_size: Optional[int] = None,  # Ignored, always uses full dataset
        shuffle: bool = False,
        **kwargs,
    ):
        """
        Initialize SimpleDataLoader.

        Args:
            dataset: Dataset with __len__ and __getitem__
            batch_size: Ignored (always returns full dataset)
            shuffle: Whether to shuffle indices before creating batch
            **kwargs: Ignored (for compatibility)
        """
        super().__init__(dataset, batch_size=len(dataset), shuffle=shuffle, **kwargs)
        self.n_samples = len(dataset)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over single batch (all data).

        Yields:
            Tuple of (X_batch, y_batch) containing entire dataset as numpy arrays
        """
        indices = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        # Collect all samples
        X_list = []
        y_list = []
        for idx in indices:
            X, y = self.dataset[int(idx)]
            X_list.append(X)
            y_list.append(y)

        # Stack into numpy arrays
        X_batch = np.stack(X_list, axis=0) if len(X_list) > 0 else np.array([])
        y_batch = np.stack(y_list, axis=0) if len(y_list) > 0 else np.array([])

        yield X_batch, y_batch

    def __len__(self) -> int:
        """
        Return number of batches (always 1).

        Returns:
            1 (single batch containing all data)
        """
        return 1


class PyTorchDataLoader(BaseDataLoader):
    """
    PyTorch DataLoader wrapper that inherits from torch.utils.data.DataLoader.

    This provides native PyTorch batching capabilities:
    - Multi-process data loading (num_workers)
    - GPU memory pinning (pin_memory)
    - Advanced sampling strategies
    - Efficient mini-batch iteration

    Usage:
        loader = PyTorchDataLoader(dataset, batch_size=64, num_workers=4)
        for X_batch, y_batch in loader:
            # X_batch and y_batch are numpy arrays (converted from tensors)
            model.fit(X_batch, y_batch)

    Hydra Config:
        dataloader:
            _target_: ml_portfolio.data.loaders.PyTorchDataLoader
            batch_size: 64
            shuffle: true
            num_workers: 4
            pin_memory: true
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Initialize PyTorchDataLoader.

        Args:
            dataset: Dataset with __len__ and __getitem__
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data before each epoch
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            **kwargs: Additional PyTorch DataLoader parameters
        """
        # Import PyTorch DataLoader
        try:
            from torch.utils.data import DataLoader as TorchDataLoader

            # Initialize the PyTorch DataLoader as the base class
            self.torch_loader = TorchDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                **kwargs,
            )

            # Store parameters for BaseDataLoader interface
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            self.kwargs = kwargs

        except ImportError:
            raise ImportError(
                "PyTorch is required for PyTorchDataLoader. "
                "Install with: pip install torch\n"
                "Or use SimpleDataLoader instead."
            )

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over batches from PyTorch DataLoader.

        Yields:
            Tuple of (X_batch, y_batch) as numpy arrays
        """
        for X_batch, y_batch in self.torch_loader:
            # Convert tensors to numpy
            if hasattr(X_batch, "numpy"):
                X_batch = X_batch.numpy()
            if hasattr(y_batch, "numpy"):
                y_batch = y_batch.numpy()

            yield X_batch, y_batch

    def __len__(self) -> int:
        """
        Return number of batches.

        Returns:
            Number of batches in the dataset
        """
        return len(self.torch_loader)
