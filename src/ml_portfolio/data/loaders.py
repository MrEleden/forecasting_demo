"""
DataLoader implementations for time series forecasting.

This module provides a unified interface for batch data loading with both
PyTorch and non-PyTorch implementations.

Architecture:
- BaseDataLoader: Master abstract class defining the interface
- PyTorchDataLoader: PyTorch-based implementation (when torch available)
- SimpleDataLoader: NumPy-based implementation (no dependencies)
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from ml_portfolio.data.datasets import TimeSeriesDataset

# Optional PyTorch support
try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    TorchDataLoader = object  # Fallback for class inheritance


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

    def __init__(self, dataset: TimeSeriesDataset, batch_size: int = 32, shuffle: bool = False, **kwargs):
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
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over batches.

        Yields:
            Tuple of (X_batch, y_batch) as numpy arrays or torch tensors
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
        dataset: TimeSeriesDataset,
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


class PyTorchDataLoader(TorchDataLoader, BaseDataLoader):
    """
    PyTorch DataLoader that inherits from torch.utils.data.DataLoader.

    This provides native PyTorch batching capabilities:
    - Multi-process data loading (num_workers)
    - GPU memory pinning (pin_memory)
    - Advanced sampling strategies
    - Efficient mini-batch iteration

    By default this loader yields native torch tensors so deep learning
    models can train efficiently. Set ``return_numpy=True`` when
    instantiating to force conversion back to NumPy for hybrid workflows.

    Usage:
        loader = PyTorchDataLoader(dataset, batch_size=64, num_workers=4)
        for X_batch, y_batch in loader:
            # X_batch and y_batch are torch tensors by default
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
        dataset: TimeSeriesDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        return_numpy: bool = False,
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
            return_numpy: If True, convert batches to numpy arrays when iterating
            **kwargs: Additional PyTorch DataLoader parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PyTorchDataLoader. "
                "Install with: pip install torch\n"
                "Or use SimpleDataLoader instead."
            )

        BaseDataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        # Initialize PyTorch DataLoader parent class
        TorchDataLoader.__init__(
            self,
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

        self.return_numpy = return_numpy

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over batches from PyTorch DataLoader.

        By default this yields native torch tensors. Set ``return_numpy=True``
        to obtain NumPy batches for compatibility with sklearn-style models.

        Yields:
            Tuple of (X_batch, y_batch) as torch tensors or numpy arrays
        """
        for X_batch, y_batch in TorchDataLoader.__iter__(self):
            if self.return_numpy:
                if TORCH_AVAILABLE and isinstance(X_batch, torch.Tensor):
                    X_batch = X_batch.detach().cpu().numpy()
                if TORCH_AVAILABLE and isinstance(y_batch, torch.Tensor):
                    y_batch = y_batch.detach().cpu().numpy()

            yield X_batch, y_batch
