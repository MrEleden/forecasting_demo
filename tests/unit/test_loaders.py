"""
Unit tests for DataLoader implementations.
"""

import numpy as np
import pytest
from ml_portfolio.data.datasets import TimeSeriesDataset
from ml_portfolio.data.loaders import SimpleDataLoader

try:
    from ml_portfolio.data.loaders import TORCH_AVAILABLE, PyTorchDataLoader
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    return TimeSeriesDataset(X=X, y=y)


class TestSimpleDataLoader:
    """Test SimpleDataLoader functionality."""

    def test_initialization(self, sample_dataset):
        """Test loader initialization."""
        loader = SimpleDataLoader(sample_dataset, batch_size=10, shuffle=False)

        assert loader.dataset is sample_dataset
        assert loader.batch_size == len(sample_dataset)  # SimpleDataLoader uses full dataset
        assert loader.shuffle is False

    def test_len_method(self, sample_dataset):
        """Test __len__ returns 1 (single batch for all data)."""
        loader = SimpleDataLoader(sample_dataset, batch_size=32)
        assert len(loader) == 1  # SimpleDataLoader always returns 1 batch

    def test_iteration_without_shuffle(self, sample_dataset):
        """Test iteration through single batch without shuffling."""
        loader = SimpleDataLoader(sample_dataset, batch_size=25, shuffle=False)

        batches = list(loader)
        assert len(batches) == 1  # Single batch

        # Check batch contains all data
        X_batch, y_batch = batches[0]
        assert X_batch.shape[0] == 100  # All samples
        assert y_batch.shape[0] == 100
        assert X_batch.shape[1] == 5

    def test_iteration_with_shuffle(self, sample_dataset):
        """Test iteration with shuffling."""
        np.random.seed(42)
        loader1 = SimpleDataLoader(sample_dataset, batch_size=20, shuffle=True)
        batches1 = list(loader1)
        X1, _ = batches1[0]

        np.random.seed(123)
        loader2 = SimpleDataLoader(sample_dataset, batch_size=20, shuffle=True)
        batches2 = list(loader2)
        X2, _ = batches2[0]

        # Different seeds should produce different ordering
        assert not np.array_equal(X1, X2), "Shuffled batches should differ with different seeds"

    def test_batch_size_parameter_ignored(self, sample_dataset):
        """Test that batch_size parameter is ignored (always full dataset)."""
        loader = SimpleDataLoader(sample_dataset, batch_size=10)

        assert len(loader) == 1
        batches = list(loader)
        X_batch, y_batch = batches[0]

        # Should still return all 100 samples despite batch_size=10
        assert X_batch.shape[0] == 100
        assert y_batch.shape[0] == 100

    def test_returns_numpy_arrays(self, sample_dataset):
        """Test that SimpleDataLoader returns numpy arrays."""
        loader = SimpleDataLoader(sample_dataset, shuffle=False)

        for X_batch, y_batch in loader:
            assert isinstance(X_batch, np.ndarray)
            assert isinstance(y_batch, np.ndarray)

    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        dataset = TimeSeriesDataset(X=X, y=y)

        loader = SimpleDataLoader(dataset, batch_size=10)
        batches = list(loader)

        # Empty dataset still returns 1 batch (with empty arrays)
        assert len(batches) == 1
        X_batch, y_batch = batches[0]
        assert X_batch.shape[0] == 0
        assert y_batch.shape[0] == 0

    def test_drop_last_true(self):
        """Test that drop_last parameter is ignored by SimpleDataLoader."""
        # SimpleDataLoader always returns full dataset, ignoring drop_last
        X = np.random.rand(95, 3)
        y = np.random.rand(95)
        dataset = TimeSeriesDataset(X=X, y=y)

        loader = SimpleDataLoader(dataset, batch_size=10, drop_last=True)
        batches = list(loader)

        # SimpleDataLoader returns all data in 1 batch regardless of drop_last
        assert len(batches) == 1
        X_batch, y_batch = batches[0]
        assert X_batch.shape[0] == 95  # All samples
        assert y_batch.shape[0] == 95

    def test_single_sample_dataset(self):
        """Test with dataset containing single sample."""
        X = np.random.rand(1, 5)
        y = np.random.rand(1)
        dataset = TimeSeriesDataset(X=X, y=y)

        loader = SimpleDataLoader(dataset, shuffle=False)
        batches = list(loader)

        assert len(batches) == 1
        X_batch, y_batch = batches[0]
        assert X_batch.shape == (1, 5)
        assert y_batch.shape == (1,)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchDataLoader:
    """Test PyTorchDataLoader functionality."""

    def test_initialization(self, sample_dataset):
        """Test PyTorch loader initialization."""
        loader = PyTorchDataLoader(sample_dataset, batch_size=16, shuffle=True)

        assert loader.dataset is sample_dataset
        assert loader.batch_size == 16
        assert loader.shuffle is True

    def test_iteration(self, sample_dataset):
        """Test iteration through PyTorch batches."""
        loader = PyTorchDataLoader(sample_dataset, batch_size=20, shuffle=False, return_numpy=False)

        batch_count = 0
        for X_batch, y_batch in loader:
            batch_count += 1
            assert X_batch.shape[1] == 5
            # PyTorch tensors
            assert hasattr(X_batch, "shape")
            assert hasattr(y_batch, "shape")

        assert batch_count == 5  # 100 / 20 = 5 batches

    def test_return_numpy(self, sample_dataset):
        """Test return_numpy option converts tensors to numpy."""
        loader = PyTorchDataLoader(sample_dataset, batch_size=20, return_numpy=True)

        for X_batch, y_batch in loader:
            assert isinstance(X_batch, np.ndarray)
            assert isinstance(y_batch, np.ndarray)
            break

    def test_drop_last_true(self):
        """Test drop_last=True drops incomplete batch."""
        X = np.random.rand(95, 3)
        y = np.random.rand(95)
        dataset = TimeSeriesDataset(X=X, y=y)

        loader = PyTorchDataLoader(dataset, batch_size=10, drop_last=True, return_numpy=True)
        batches = list(loader)

        # Should have 9 batches (drop the incomplete last batch of 5)
        assert len(batches) == 9
        for X_batch, y_batch in batches:
            assert X_batch.shape[0] == 10

    def test_drop_last_false(self):
        """Test drop_last=False keeps incomplete batch."""
        X = np.random.rand(95, 3)
        y = np.random.rand(95)
        dataset = TimeSeriesDataset(X=X, y=y)

        loader = PyTorchDataLoader(dataset, batch_size=10, drop_last=False, return_numpy=True)
        batches = list(loader)

        # Should have 10 batches (last batch has 5 samples)
        assert len(batches) == 10
        # Check last batch is incomplete
        X_last, y_last = batches[-1]
        assert X_last.shape[0] == 5

    def test_pin_memory(self, sample_dataset):
        """Test pin_memory option."""
        loader = PyTorchDataLoader(sample_dataset, batch_size=10, pin_memory=True)
        batches = list(loader)

        assert len(batches) == 10


def test_pytorch_import_error_without_torch():
    """Test that PyTorchDataLoader raises ImportError when PyTorch is not available."""
    if not TORCH_AVAILABLE:
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        dataset = TimeSeriesDataset(X=X, y=y)

        with pytest.raises(ImportError, match="PyTorch is required"):
            PyTorchDataLoader(dataset, batch_size=5)
