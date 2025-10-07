"""
Unit tests for data preprocessing pipelines.
"""

import numpy as np
import pandas as pd
import pytest
from ml_portfolio.data.preprocessing import StaticTimeSeriesPreprocessingPipeline, StatisticalPreprocessingPipeline


@pytest.mark.skip(reason="API mismatch - needs refactoring to use TimeSeriesDataset")
class TestStaticTimeSeriesPreprocessing:
    """Test static feature engineering pipeline."""

    def test_lag_features_creation(self, sample_timeseries_data):
        """Test lag feature creation."""
        pipeline = StaticTimeSeriesPreprocessingPipeline(
            date_column="Date",
            target_column="Weekly_Sales",
            group_columns=["Store"],
            lag_features=[1, 2, 4],
            rolling_windows=[],
            date_features=False,
            cyclical_features=[],
        )

        result = pipeline.fit_transform(sample_timeseries_data)

        assert "lag_1" in result.columns
        assert "lag_2" in result.columns
        assert "lag_4" in result.columns

    def test_rolling_window_features(self, sample_timeseries_data):
        """Test rolling window statistics."""
        pipeline = StaticTimeSeriesPreprocessingPipeline(
            date_column="Date",
            target_column="Weekly_Sales",
            group_columns=["Store"],
            lag_features=[],
            rolling_windows=[4, 8],
            date_features=False,
            cyclical_features=[],
        )

        result = pipeline.fit_transform(sample_timeseries_data)

        assert "rolling_mean_4" in result.columns
        assert "rolling_mean_8" in result.columns

    def test_date_features_extraction(self, sample_timeseries_data):
        """Test date feature extraction."""
        pipeline = StaticTimeSeriesPreprocessingPipeline(
            date_column="Date",
            target_column="Weekly_Sales",
            group_columns=["Store"],
            lag_features=[],
            rolling_windows=[],
            date_features=True,
            cyclical_features=[],
        )

        result = pipeline.fit_transform(sample_timeseries_data)

        assert "year" in result.columns
        assert "month" in result.columns
        assert "day" in result.columns
        assert "dayofweek" in result.columns

    def test_cyclical_encoding(self, sample_timeseries_data):
        """Test cyclical feature encoding."""
        pipeline = StaticTimeSeriesPreprocessingPipeline(
            date_column="Date",
            target_column="Weekly_Sales",
            group_columns=["Store"],
            lag_features=[],
            rolling_windows=[],
            date_features=False,
            cyclical_features=["month", "dayofweek"],
        )

        result = pipeline.fit_transform(sample_timeseries_data)

        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
        assert "dayofweek_sin" in result.columns
        assert "dayofweek_cos" in result.columns

    def test_no_data_leakage(self, sample_timeseries_data):
        """Test that lag features don't leak future data."""
        pipeline = StaticTimeSeriesPreprocessingPipeline(
            date_column="Date",
            target_column="Weekly_Sales",
            group_columns=["Store"],
            lag_features=[1],
            rolling_windows=[],
            date_features=False,
            cyclical_features=[],
        )

        result = pipeline.fit_transform(sample_timeseries_data)

        # First row should have NaN lag (no previous data)
        first_row = result.iloc[0]
        assert pd.isna(first_row["lag_1"]) or first_row["lag_1"] != first_row["Weekly_Sales"]


@pytest.mark.skip(reason="API mismatch - needs refactoring to use TimeSeriesDataset")
class TestStatisticalPreprocessing:
    """Test statistical preprocessing pipeline."""

    def test_standard_scaling(self, sample_arrays):
        """Test StandardScaler integration."""
        from sklearn.preprocessing import StandardScaler

        pipeline = StatisticalPreprocessingPipeline(
            steps=[["feature_scaler", StandardScaler()], ["target_scaler", StandardScaler()]]
        )

        y_true, y_pred = sample_arrays
        X = np.column_stack([y_true, y_pred])

        X_scaled, y_scaled = pipeline.fit_transform(X, y_true)

        # Check that scaling was applied
        assert np.abs(X_scaled.mean()) < 1e-10  # Mean should be ~0
        assert np.abs(X_scaled.std() - 1.0) < 1e-1  # Std should be ~1

    def test_inverse_transform(self, sample_arrays):
        """Test inverse transform restores original values."""
        from sklearn.preprocessing import StandardScaler

        pipeline = StatisticalPreprocessingPipeline(
            steps=[["feature_scaler", StandardScaler()], ["target_scaler", StandardScaler()]]
        )

        y_true, y_pred = sample_arrays
        X = np.column_stack([y_true, y_pred])

        X_scaled, y_scaled = pipeline.fit_transform(X, y_true)
        y_restored = pipeline.inverse_transform_target(y_scaled)

        np.testing.assert_array_almost_equal(y_true, y_restored, decimal=5)

    def test_pipeline_persistence(self, sample_arrays, tmp_path):
        """Test pipeline can be saved and loaded."""
        import pickle

        from sklearn.preprocessing import StandardScaler

        pipeline = StatisticalPreprocessingPipeline(
            steps=[["feature_scaler", StandardScaler()], ["target_scaler", StandardScaler()]]
        )

        y_true, y_pred = sample_arrays
        X = np.column_stack([y_true, y_pred])

        pipeline.fit_transform(X, y_true)

        # Save pipeline
        pipeline_path = tmp_path / "pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Load pipeline
        with open(pipeline_path, "rb") as f:
            loaded_pipeline = pickle.load(f)

        # Test loaded pipeline works
        X_scaled, y_scaled = loaded_pipeline.transform(X, y_true)
        assert X_scaled is not None
        assert y_scaled is not None
