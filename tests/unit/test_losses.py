"""
Unit tests for loss functions.
"""

import numpy as np

from ml_portfolio.evaluation.losses import (
    huber_loss,
    mae_loss,
    mse_loss,
    pinball_loss,
    quantile_loss,
    smape_loss,
)


class TestBasicLosses:
    """Test basic loss functions."""

    def test_mse_loss(self):
        """Test MSE loss calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        loss = mse_loss(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)

        assert np.isclose(loss, expected)
        assert loss >= 0

    def test_mae_loss(self):
        """Test MAE loss calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])

        loss = mae_loss(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))

        assert np.isclose(loss, expected)
        assert loss == 0.5

    def test_mse_various_cases(self):
        """Test MSE with various cases."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        loss = mse_loss(y_true, y_pred)

        assert np.isclose(loss, 0.0)
        assert loss >= 0

    def test_smape_loss(self):
        """Test SMAPE loss calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 2.1, 2.9, 4.2])

        loss = smape_loss(y_true, y_pred)

        assert loss >= 0
        assert loss <= 200  # SMAPE range is [0, 200]


class TestQuantileLosses:
    """Test quantile and pinball loss functions."""

    def test_quantile_loss_median(self):
        """Test quantile loss at median (q=0.5)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        loss = quantile_loss(y_true, y_pred, quantile=0.5)

        assert np.isclose(loss, 0.0)

    def test_quantile_loss_upper(self):
        """Test quantile loss at upper quantile."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.5, 1.5, 2.5, 3.5])

        loss = quantile_loss(y_true, y_pred, quantile=0.9)

        assert loss >= 0

    def test_quantile_loss_lower(self):
        """Test quantile loss at lower quantile."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])

        loss = quantile_loss(y_true, y_pred, quantile=0.1)

        assert loss >= 0

    def test_pinball_loss(self):
        """Test pinball loss."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        loss = pinball_loss(y_true, y_pred, quantile=0.5)

        assert loss >= 0


class TestRobustLosses:
    """Test robust loss functions."""

    def test_huber_loss(self):
        """Test Huber loss."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        loss = huber_loss(y_true, y_pred, delta=1.0)

        assert loss >= 0

    def test_huber_loss_with_outliers(self):
        """Test Huber loss handles outliers."""
        y_true = np.array([1.0, 2.0, 3.0, 100.0])  # Outlier
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        huber = huber_loss(y_true, y_pred, delta=1.0)
        mse = mse_loss(y_true, y_pred)

        # Huber should be more robust to outliers than MSE
        assert huber < mse


class TestEdgeCases:
    """Test edge cases for loss functions."""

    def test_perfect_predictions(self):
        """Test losses with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()

        assert np.isclose(mse_loss(y_true, y_pred), 0.0)
        assert np.isclose(mae_loss(y_true, y_pred), 0.0)

    def test_single_value(self):
        """Test losses with single value."""
        y_true = np.array([1.0])
        y_pred = np.array([1.5])

        loss = mae_loss(y_true, y_pred)
        assert np.isclose(loss, 0.5)

    def test_large_arrays(self):
        """Test losses with large arrays."""
        np.random.seed(42)
        y_true = np.random.rand(1000)
        y_pred = y_true + np.random.rand(1000) * 0.1

        loss = mse_loss(y_true, y_pred)
        assert loss >= 0
        assert np.isfinite(loss)
