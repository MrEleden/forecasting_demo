"""
Unit tests for TimeSeriesDataset class.
"""

import numpy as np

from ml_portfolio.data.datasets import TimeSeriesDataset


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset functionality."""

    def test_initialization_with_minimal_args(self):
        """Test dataset creation with only X and y."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])

        dataset = TimeSeriesDataset(X=X, y=y)

        assert len(dataset) == 3
        assert dataset.X.shape == (3, 2)
        assert dataset.y.shape == (3,)
        assert dataset.timestamps is None
        assert dataset.feature_names == []
        assert dataset.metadata == {}

    def test_initialization_with_all_args(self):
        """Test dataset creation with all optional parameters."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([10, 20])
        timestamps = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        feature_names = ["feature_1", "feature_2"]
        metadata = {"source": "test", "version": 1}

        dataset = TimeSeriesDataset(X=X, y=y, timestamps=timestamps, feature_names=feature_names, metadata=metadata)

        assert len(dataset) == 2
        assert dataset.timestamps is not None
        assert len(dataset.timestamps) == 2
        assert dataset.feature_names == ["feature_1", "feature_2"]
        assert dataset.metadata["source"] == "test"
        assert dataset.metadata["version"] == 1

    def test_len_method(self):
        """Test __len__ returns correct number of samples."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        dataset = TimeSeriesDataset(X=X, y=y)

        assert len(dataset) == 100
        assert len(dataset) == X.shape[0]

    def test_getitem_method(self):
        """Test __getitem__ returns correct sample."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])

        dataset = TimeSeriesDataset(X=X, y=y)

        # Get first sample
        X_sample, y_sample = dataset[0]
        np.testing.assert_array_equal(X_sample, np.array([1, 2]))
        assert y_sample == 10

        # Get last sample
        X_sample, y_sample = dataset[2]
        np.testing.assert_array_equal(X_sample, np.array([5, 6]))
        assert y_sample == 30

    def test_getitem_with_slice(self):
        """Test __getitem__ with slice indexing."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([10, 20, 30, 40])

        dataset = TimeSeriesDataset(X=X, y=y)

        # Slice indexing
        X_slice, y_slice = dataset[1:3]
        assert X_slice.shape == (2, 2)
        assert y_slice.shape == (2,)
        np.testing.assert_array_equal(X_slice, np.array([[3, 4], [5, 6]]))
        np.testing.assert_array_equal(y_slice, np.array([20, 30]))

    def test_get_feature_dim_2d(self):
        """Test get_feature_dim with 2D feature array."""
        X = np.random.rand(50, 10)
        y = np.random.rand(50)

        dataset = TimeSeriesDataset(X=X, y=y)

        assert dataset.get_feature_dim() == 10

    def test_get_feature_dim_1d(self):
        """Test get_feature_dim with 1D feature array."""
        X = np.random.rand(50)
        y = np.random.rand(50)

        dataset = TimeSeriesDataset(X=X, y=y)

        assert dataset.get_feature_dim() == 1

    def test_get_data_method(self):
        """Test get_data returns X and y arrays."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([10, 20])

        dataset = TimeSeriesDataset(X=X, y=y)

        X_returned, y_returned = dataset.get_data()

        np.testing.assert_array_equal(X_returned, X)
        np.testing.assert_array_equal(y_returned, y)
        # Verify they're the same objects (not copies)
        assert X_returned is X
        assert y_returned is y

    def test_empty_metadata_dict(self):
        """Test that empty metadata dict is created if None provided."""
        X = np.array([[1, 2]])
        y = np.array([10])

        dataset = TimeSeriesDataset(X=X, y=y, metadata=None)

        assert isinstance(dataset.metadata, dict)
        assert len(dataset.metadata) == 0

    def test_empty_feature_names_list(self):
        """Test that empty feature names list is created if None provided."""
        X = np.array([[1, 2]])
        y = np.array([10])

        dataset = TimeSeriesDataset(X=X, y=y, feature_names=None)

        assert isinstance(dataset.feature_names, list)
        assert len(dataset.feature_names) == 0

    def test_metadata_is_mutable(self):
        """Test that metadata can be modified after creation."""
        X = np.array([[1, 2]])
        y = np.array([10])
        metadata = {"original": "value"}

        dataset = TimeSeriesDataset(X=X, y=y, metadata=metadata)

        # Add new metadata
        dataset.metadata["new_key"] = "new_value"

        assert "new_key" in dataset.metadata
        assert dataset.metadata["new_key"] == "new_value"
        assert dataset.metadata["original"] == "value"

    def test_with_3d_features(self):
        """Test dataset with 3D feature array (e.g., for RNNs)."""
        # Shape: (samples, timesteps, features)
        X = np.random.rand(20, 10, 5)
        y = np.random.rand(20)

        dataset = TimeSeriesDataset(X=X, y=y)

        assert len(dataset) == 20
        assert dataset.X.shape == (20, 10, 5)
        # get_feature_dim returns second dimension (shape[1]) for any ndim > 1
        assert dataset.get_feature_dim() == 10

    def test_single_sample_dataset(self):
        """Test dataset with only one sample."""
        X = np.array([[1, 2, 3]])
        y = np.array([10])

        dataset = TimeSeriesDataset(X=X, y=y)

        assert len(dataset) == 1
        X_sample, y_sample = dataset[0]
        np.testing.assert_array_equal(X_sample, np.array([1, 2, 3]))
        assert y_sample == 10


class TestTimeSeriesDatasetEdgeCases:
    """Test edge cases and error conditions."""

    def test_mismatched_X_y_lengths(self):
        """Test that mismatched X and y lengths work (no validation in constructor)."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20])  # Intentionally shorter

        # Constructor doesn't validate - this is by design (pure data container)
        dataset = TimeSeriesDataset(X=X, y=y)

        # Length is based on X
        assert len(dataset) == 3

    def test_with_different_dtypes(self):
        """Test dataset with different data types."""
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([10, 20], dtype=np.int64)

        dataset = TimeSeriesDataset(X=X, y=y)

        assert dataset.X.dtype == np.float32
        assert dataset.y.dtype == np.int64

    def test_negative_indexing(self):
        """Test negative indexing works correctly."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])

        dataset = TimeSeriesDataset(X=X, y=y)

        # Get last sample using negative index
        X_sample, y_sample = dataset[-1]
        np.testing.assert_array_equal(X_sample, np.array([5, 6]))
        assert y_sample == 30

        # Get second-to-last
        X_sample, y_sample = dataset[-2]
        np.testing.assert_array_equal(X_sample, np.array([3, 4]))
        assert y_sample == 20

    def test_large_dataset(self):
        """Test with large dataset to verify performance."""
        X = np.random.rand(10000, 50)
        y = np.random.rand(10000)

        dataset = TimeSeriesDataset(X=X, y=y)

        assert len(dataset) == 10000
        assert dataset.get_feature_dim() == 50

        # Test random access
        X_sample, y_sample = dataset[5000]
        assert X_sample.shape == (50,)
        assert isinstance(y_sample, (int, float, np.number))
