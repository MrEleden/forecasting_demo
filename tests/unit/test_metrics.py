"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest
from ml_portfolio.evaluation.metrics import MAEMetric, MAPEMetric, RMSEMetric


class TestMAPEMetric:
    """Test Mean Absolute Percentage Error metric."""

    def test_perfect_prediction(self):
        """Test MAPE with perfect predictions."""
        metric = MAPEMetric()
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])
        result = metric(y_true, y_pred)
        assert result == 0.0, "MAPE should be 0 for perfect predictions"

    def test_known_values(self):
        """Test MAPE with known expected values."""
        metric = MAPEMetric()
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = metric(y_true, y_pred)
        # Expected: mean([10/100, 10/200, 10/300]) * 100 = mean([10, 5, 3.33]) = 6.11
        expected = (10 + 5 + 10 / 3) / 3
        np.testing.assert_almost_equal(result, expected, decimal=2)

    def test_zero_division_protection(self):
        """Test MAPE handles zero values with epsilon."""
        metric = MAPEMetric(epsilon=1e-8)
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        result = metric(y_true, y_pred)
        assert not np.isnan(result), "MAPE should not return NaN"
        assert not np.isinf(result), "MAPE should not return Inf"

    def test_negative_values(self):
        """Test MAPE with negative values."""
        metric = MAPEMetric()
        y_true = np.array([-100, -200, -300])
        y_pred = np.array([-110, -190, -310])
        result = metric(y_true, y_pred)
        assert result >= 0, "MAPE should always be non-negative"

    def test_single_value(self):
        """Test MAPE with single value."""
        metric = MAPEMetric()
        y_true = np.array([100])
        y_pred = np.array([110])
        result = metric(y_true, y_pred)
        expected = 10.0
        np.testing.assert_almost_equal(result, expected, decimal=2)


class TestRMSEMetric:
    """Test Root Mean Squared Error metric."""

    def test_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        metric = RMSEMetric()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        result = metric(y_true, y_pred)
        assert result == 0.0, "RMSE should be 0 for perfect predictions"

    def test_known_values(self):
        """Test RMSE with known expected values."""
        metric = RMSEMetric()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        result = metric(y_true, y_pred)
        # Expected: sqrt(mean([1, 1, 1])) = 1.0
        expected = 1.0
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_larger_errors_penalized(self):
        """Test RMSE penalizes larger errors more."""
        metric = RMSEMetric()
        y_true = np.array([1, 2, 3, 4])
        y_pred1 = np.array([2, 3, 4, 5])  # All errors = 1
        y_pred2 = np.array([1, 2, 3, 8])  # One large error = 4

        rmse1 = metric(y_true, y_pred1)
        rmse2 = metric(y_true, y_pred2)

        assert rmse2 > rmse1, "Larger errors should result in higher RMSE"


class TestMAEMetric:
    """Test Mean Absolute Error metric."""

    def test_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        metric = MAEMetric()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        result = metric(y_true, y_pred)
        assert result == 0.0, "MAE should be 0 for perfect predictions"

    def test_known_values(self):
        """Test MAE with known expected values."""
        metric = MAEMetric()
        y_true = np.array([10, 20, 30])
        y_pred = np.array([15, 25, 35])
        result = metric(y_true, y_pred)
        expected = 5.0
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_negative_errors_cancel(self):
        """Test MAE treats positive and negative errors equally."""
        metric = MAEMetric()
        y_true = np.array([10, 20])
        y_pred1 = np.array([15, 25])  # Both overestimate
        y_pred2 = np.array([5, 15])  # Both underestimate

        mae1 = metric(y_true, y_pred1)
        mae2 = metric(y_true, y_pred2)

        np.testing.assert_almost_equal(mae1, mae2, decimal=6)


class TestMetricEdgeCases:
    """Test edge cases across all metrics."""

    def test_empty_arrays(self):
        """Test metrics with empty arrays."""
        metrics = [MAPEMetric(), RMSEMetric(), MAEMetric()]
        y_true = np.array([])
        y_pred = np.array([])

        for metric in metrics:
            with pytest.raises(ValueError):
                metric(y_true, y_pred)

    def test_mismatched_shapes(self):
        """Test metrics with mismatched array shapes."""
        metrics = [MAPEMetric(), RMSEMetric(), MAEMetric()]
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        for metric in metrics:
            with pytest.raises(ValueError):
                metric(y_true, y_pred)

    def test_very_large_values(self):
        """Test metrics with very large values."""
        metrics = [RMSEMetric(), MAEMetric()]
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1.1e10, 1.9e10, 3.1e10])

        for metric in metrics:
            result = metric(y_true, y_pred)
            assert not np.isnan(result)
            assert not np.isinf(result)
