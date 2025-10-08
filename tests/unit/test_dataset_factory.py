"""
Unit tests for DatasetFactory.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_portfolio.data.dataset_factory import DatasetFactory
from ml_portfolio.data.datasets import TimeSeriesDataset


@pytest.fixture
def sample_csv_data():
    """Create a temporary CSV file with sample data."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "feature_1": np.random.rand(100),
            "feature_2": np.random.rand(100),
            "target": np.random.rand(100),
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    Path(f.name).unlink()


@pytest.fixture
def sample_parquet_data():
    """Create a temporary Parquet file with sample data."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "feature_1": np.random.rand(100),
            "feature_2": np.random.rand(100),
            "target": np.random.rand(100),
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name, index=False)
        yield f.name

    # Cleanup
    Path(f.name).unlink()


class TestDatasetFactory:
    """Test DatasetFactory functionality."""

    def test_initialization(self, sample_csv_data):
        """Test factory initialization."""
        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2"],
            timestamp_column="timestamp",
        )

        assert factory.data_path == sample_csv_data
        assert factory.target_column == "target"
        assert factory.feature_columns == ["feature_1", "feature_2"]
        assert factory.timestamp_column == "timestamp"

    def test_create_datasets_csv(self, sample_csv_data):
        """Test creating datasets from CSV."""
        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2"],
            timestamp_column="timestamp",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        train_ds, val_ds, test_ds = factory.create_datasets()

        # Check dataset types
        assert isinstance(train_ds, TimeSeriesDataset)
        assert isinstance(val_ds, TimeSeriesDataset)
        assert isinstance(test_ds, TimeSeriesDataset)

        # Check split sizes (approximately)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100
        assert len(train_ds) == 70
        assert len(val_ds) == 15
        assert len(test_ds) == 15

    def test_create_datasets_parquet(self, sample_parquet_data):
        """Test creating datasets from Parquet."""
        factory = DatasetFactory(
            data_path=sample_parquet_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2"],
            timestamp_column="timestamp",
        )

        train_ds, val_ds, test_ds = factory.create_datasets()

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_temporal_split_order(self, sample_csv_data):
        """Test that temporal split respects time order."""
        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2"],
            timestamp_column="timestamp",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        train_ds, val_ds, test_ds = factory.create_datasets()

        # Train should have earliest timestamps
        # Val should have middle timestamps
        # Test should have latest timestamps
        if train_ds.timestamps is not None:
            assert train_ds.timestamps[-1] <= val_ds.timestamps[0]
            assert val_ds.timestamps[-1] <= test_ds.timestamps[0]

    def test_feature_extraction(self, sample_csv_data):
        """Test feature and target extraction."""
        factory = DatasetFactory(
            data_path=sample_csv_data, target_column="target", feature_columns=["feature_1", "feature_2"]
        )

        train_ds, _, _ = factory.create_datasets()

        assert train_ds.X.shape[1] == 2  # 2 features
        assert train_ds.feature_names == ["feature_1", "feature_2"]

    def test_auto_feature_columns(self, sample_csv_data):
        """Test automatic feature column detection."""
        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=None,  # Auto-detect
            timestamp_column="timestamp",
        )

        train_ds, _, _ = factory.create_datasets()

        # Should use all columns except target and timestamp
        assert "target" not in train_ds.feature_names
        assert "timestamp" not in train_ds.feature_names

    def test_custom_split_ratios(self, sample_csv_data):
        """Test custom split ratios."""
        factory = DatasetFactory(
            data_path=sample_csv_data, target_column="target", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        train_ds, val_ds, test_ds = factory.create_datasets()

        assert len(train_ds) == 80
        assert len(val_ds) == 10
        assert len(test_ds) == 10

    def test_no_timestamp_column(self, sample_csv_data):
        """Test without timestamp column."""
        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2"],
            timestamp_column=None,
        )

        train_ds, val_ds, test_ds = factory.create_datasets()

        assert train_ds.timestamps is None
        assert val_ds.timestamps is None
        assert test_ds.timestamps is None

    def test_missing_file(self):
        """Test with non-existent file."""
        factory = DatasetFactory(data_path="non_existent_file.csv", target_column="target")

        with pytest.raises(FileNotFoundError):
            factory.create_datasets()

    def test_invalid_target_column(self, sample_csv_data):
        """Test with invalid target column."""
        factory = DatasetFactory(data_path=sample_csv_data, target_column="invalid_column")

        with pytest.raises((KeyError, ValueError)):
            factory.create_datasets()

    def test_metadata_preservation(self, sample_csv_data):
        """Test that metadata is preserved in datasets."""
        factory = DatasetFactory(
            data_path=sample_csv_data, target_column="target", feature_columns=["feature_1", "feature_2"]
        )

        train_ds, val_ds, test_ds = factory.create_datasets()

        # Datasets should have metadata
        assert isinstance(train_ds.metadata, dict)
        assert isinstance(val_ds.metadata, dict)
        assert isinstance(test_ds.metadata, dict)


class TestStaticFeatureEngineering:
    """Test static feature engineering in DatasetFactory."""

    def test_with_static_feature_engineer(self, sample_csv_data):
        """Test with custom static feature engineer."""

        class SimpleFeatureEngineer:
            def engineer_features(self, df):
                # Add a simple derived feature
                df["feature_sum"] = df["feature_1"] + df["feature_2"]
                return df

        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2", "feature_sum"],
            static_feature_engineer=SimpleFeatureEngineer(),
        )

        train_ds, _, _ = factory.create_datasets()

        assert train_ds.X.shape[1] == 3  # 3 features including engineered one
        assert "feature_sum" in train_ds.feature_names

    def test_without_static_feature_engineer(self, sample_csv_data):
        """Test without feature engineering."""
        factory = DatasetFactory(
            data_path=sample_csv_data,
            target_column="target",
            feature_columns=["feature_1", "feature_2"],
            static_feature_engineer=None,
        )

        train_ds, _, _ = factory.create_datasets()

        assert train_ds.X.shape[1] == 2
