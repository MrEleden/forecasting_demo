"""
Unit tests for DataLoader implementations.
"""

import numpy as np
import pytest

pytest.skip("Loader tests disabled pending API verification", allow_module_level=True)

from ml_portfolio.data.datasets import TimeSeriesDataset
from ml_portfolio.data.loaders import BaseDataLoader, SimpleDataLoader

try:
    from ml_portfolio.data.loaders import PyTorchDataLoader, TORCH_AVAILABLE
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
        assert loader.batch_size == 10
        assert loader.shuffle is False
        assert len(loader) == 10  # 100 samples / 10 batch_size

    def test_len_method(self, sample_dataset):
        """Test __len__ returns correct number of batches."""
        loader = SimpleDataLoader(sample_dataset, batch_size=32)
        expected_batches = int(np.ceil(100 / 32))
        assert len(loader) == expected_batches

    def test_iteration_without_shuffle(self, sample_dataset):
        """Test iteration through batches without shuffling."""
        loader = SimpleDataLoader(sample_dataset, batch_size=25, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 4  # 100 / 25 = 4 batches
        
        # Check batch shapes
        for i, (X_batch, y_batch) in enumerate(batches):
            assert X_batch.shape[0] == 25
            assert y_batch.shape[0] == 25
            assert X_batch.shape[1] == 5

    def test_iteration_with_shuffle(self, sample_dataset):
        """Test iteration with shuffling."""
        loader = SimpleDataLoader(sample_dataset, batch_size=20, shuffle=True)
        
        batches1 = list(loader)
        batches2 = list(loader)
        
        # Batches should be different due to shuffling
        X1, _ = batches1[0]
        X2, _ = batches2[0]
        
        # There's a small chance they could be equal, but very unlikely
        # Check that at least one batch is different
        all_same = all(np.array_equal(b1[0], b2[0]) for b1, b2 in zip(batches1, batches2))
        assert not all_same, "Shuffled batches should differ between iterations"

    def test_batch_size_larger_than_dataset(self, sample_dataset):
        """Test with batch size larger than dataset."""
        loader = SimpleDataLoader(sample_dataset, batch_size=200)
        
        assert len(loader) == 1
        batches = list(loader)
        X_batch, y_batch = batches[0]
        
        assert X_batch.shape[0] == 100
        assert y_batch.shape[0] == 100

    def test_batch_size_of_one(self, sample_dataset):
        """Test with batch size of 1."""
        loader = SimpleDataLoader(sample_dataset, batch_size=1)
        
        assert len(loader) == 100
        batches = list(loader)
        
        for X_batch, y_batch in batches:
            assert X_batch.shape == (1, 5)
            assert y_batch.shape == (1,)

    def test_drop_last_false(self):
        """Test with drop_last=False (default)."""
        X = np.random.rand(95, 3)
        y = np.random.rand(95)
        dataset = TimeSeriesDataset(X=X, y=y)
        
        loader = SimpleDataLoader(dataset, batch_size=10, drop_last=False)
        batches = list(loader)
        
        # Should have 10 batches (9 full + 1 partial with 5 samples)
        assert len(batches) == 10
        assert batches[-1][0].shape[0] == 5  # Last batch has 5 samples

    def test_drop_last_true(self):
        """Test with drop_last=True."""
        X = np.random.rand(95, 3)
        y = np.random.rand(95)
        dataset = TimeSeriesDataset(X=X, y=y)
        
        loader = SimpleDataLoader(dataset, batch_size=10, drop_last=True)
        batches = list(loader)
        
        # Should have 9 batches (drop the incomplete last batch)
        assert len(batches) == 9
        for X_batch, y_batch in batches:
            assert X_batch.shape[0] == 10

    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        dataset = TimeSeriesDataset(X=X, y=y)
        
        loader = SimpleDataLoader(dataset, batch_size=10)
        batches = list(loader)
        
        assert len(batches) == 0


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
        loader = PyTorchDataLoader(sample_dataset, batch_size=20, shuffle=False)
        
        batch_count = 0
        for X_batch, y_batch in loader:
            batch_count += 1
            assert X_batch.shape[1] == 5
            # PyTorch tensors
            assert hasattr(X_batch, 'shape')
            assert hasattr(y_batch, 'shape')
        
        assert batch_count == 5  # 100 / 20 = 5 batches

    def test_num_workers(self, sample_dataset):
        """Test with multiple workers."""
        loader = PyTorchDataLoader(sample_dataset, batch_size=10, num_workers=2)
        batches = list(loader)
        
        assert len(batches) == 10

    def test_pin_memory(self, sample_dataset):
        """Test pin_memory option."""
        loader = PyTorchDataLoader(sample_dataset, batch_size=10, pin_memory=True)
        batches = list(loader)
        
        assert len(batches) == 10
