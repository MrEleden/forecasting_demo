"""
Comprehensive tests for evaluation/metrics.py to increase coverage.
"""

import numpy as np
from ml_portfolio.evaluation.metrics import (
    directional_accuracy,
    mae,
    mape,
    mase,
    rmse,
    smape,
)


class TestMAEComprehensive:
    """Comprehensive MAE tests."""

    def test_mae_zeros(self):
        assert mae(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0

    def test_mae_ones(self):
        result = mae(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
        assert result == 1.0

    def test_mae_negatives(self):
        result = mae(np.array([-1, -2, -3]), np.array([-2, -3, -4]))
        assert result == 1.0

    def test_mae_mixed(self):
        result = mae(np.array([1, -1, 0]), np.array([2, -2, 1]))
        expected = np.mean([1, 1, 1])
        assert result == expected

    def test_mae_large_values(self):
        result = mae(np.array([1000, 2000]), np.array([1100, 2100]))
        assert result == 100.0

    def test_mae_small_diff(self):
        result = mae(np.array([1.0, 2.0]), np.array([1.001, 2.001]))
        assert result < 0.01


class TestRMSEComprehensive:
    """Comprehensive RMSE tests."""

    def test_rmse_zeros(self):
        assert rmse(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0

    def test_rmse_ones(self):
        result = rmse(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
        assert result == 1.0

    def test_rmse_squares(self):
        result = rmse(np.array([0, 0, 0]), np.array([3, 4, 0]))
        expected = np.sqrt((9 + 16 + 0) / 3)
        assert np.isclose(result, expected)

    def test_rmse_negatives(self):
        result = rmse(np.array([-10, -20]), np.array([-11, -21]))
        assert result == 1.0

    def test_rmse_large(self):
        result = rmse(np.array([100]), np.array([110]))
        assert result == 10.0


class TestMAPEComprehensive:
    """Comprehensive MAPE tests."""

    def test_mape_perfect(self):
        assert mape(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0

    def test_mape_10_percent(self):
        result = mape(np.array([100]), np.array([110]))
        assert result == 10.0

    def test_mape_50_percent(self):
        result = mape(np.array([100]), np.array([150]))
        assert result == 50.0

    def test_mape_multiple_values(self):
        result = mape(np.array([100, 200]), np.array([110, 220]))
        assert result == 10.0

    def test_mape_negative_values(self):
        result = mape(np.array([-100, -200]), np.array([-110, -220]))
        assert result == 10.0


class TestSMAPEComprehensive:
    """Comprehensive SMAPE tests."""

    def test_smape_perfect(self):
        assert smape(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0

    def test_smape_basic(self):
        result = smape(np.array([100]), np.array([110]))
        assert 0 <= result <= 100

    def test_smape_symmetry(self):
        s1 = smape(np.array([100]), np.array([110]))
        s2 = smape(np.array([110]), np.array([100]))
        assert np.isclose(s1, s2)

    def test_smape_multiple(self):
        result = smape(np.array([10, 20, 30]), np.array([11, 21, 31]))
        assert 0 < result < 100


class TestDirectionalAccuracyComprehensive:
    """Comprehensive directional accuracy tests."""

    def test_directional_all_up(self):
        result = directional_accuracy(np.array([1, 2, 3, 4]), np.array([1.1, 2.1, 3.1, 4.1]))
        assert result == 100.0

    def test_directional_all_down(self):
        result = directional_accuracy(np.array([4, 3, 2, 1]), np.array([3.9, 2.9, 1.9, 0.9]))
        assert result == 100.0

    def test_directional_mixed(self):
        result = directional_accuracy(np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]))
        assert result >= 0 and result <= 100

    def test_directional_opposite(self):
        result = directional_accuracy(np.array([1, 2, 1, 2]), np.array([0, 3, 2, 1]))
        assert isinstance(result, float)


class TestMASEComprehensive:
    """Comprehensive MASE tests."""

    def test_mase_perfect(self):
        result = mase(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0.5, 1, 1.5]))
        assert result == 0.0

    def test_mase_basic(self):
        result = mase(np.array([3, 4, 5]), np.array([3.1, 4.1, 5.1]), np.array([1, 2, 3]))
        assert result > 0

    def test_mase_with_trend(self):
        result = mase(np.array([10, 11, 12]), np.array([10.5, 11.5, 12.5]), np.array([1, 2, 3, 4, 5]))
        assert not np.isnan(result)
        assert not np.isinf(result)


class TestMetricsConsistency:
    """Test consistency and relationships between metrics."""

    def test_all_metrics_with_perfect_pred(self):
        y_true = np.array([10, 20, 30])
        y_pred = y_true.copy()

        assert mae(y_true, y_pred) == 0.0
        assert rmse(y_true, y_pred) == 0.0
        assert mape(y_true, y_pred) == 0.0
        assert smape(y_true, y_pred) == 0.0

    def test_mae_vs_rmse(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)

        assert mae_val <= rmse_val

    def test_percentage_metrics_scale_invariance(self):
        y1_true = np.array([10, 20])
        y1_pred = np.array([11, 22])

        y2_true = np.array([100, 200])
        y2_pred = np.array([110, 220])

        mape1 = mape(y1_true, y1_pred)
        mape2 = mape(y2_true, y2_pred)

        assert np.isclose(mape1, mape2)


class TestMetricsEdgeCases:
    """Test edge cases."""

    def test_single_value(self):
        y_true = np.array([10.0])
        y_pred = np.array([12.0])

        assert mae(y_true, y_pred) == 2.0
        assert rmse(y_true, y_pred) == 2.0
        assert mape(y_true, y_pred) == 20.0

    def test_two_values(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 21.0])

        assert mae(y_true, y_pred) == 1.0
        assert rmse(y_true, y_pred) == 1.0

    def test_large_arrays(self):
        y_true = np.random.rand(1000)
        y_pred = y_true + np.random.rand(1000) * 0.1

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)

        assert mae_val > 0
        assert rmse_val > 0
        assert mae_val <= rmse_val

    def test_identical_values(self):
        y_true = np.ones(10) * 5.5
        y_pred = np.ones(10) * 6.5

        mae_val = mae(y_true, y_pred)
        assert mae_val == 1.0

    def test_alternating_errors(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 1, 4, 3, 6])

        mae_val = mae(y_true, y_pred)
        assert mae_val == 1.0

    def test_very_small_values(self):
        y_true = np.array([0.001, 0.002, 0.003])
        y_pred = np.array([0.0011, 0.0021, 0.0031])

        mae_val = mae(y_true, y_pred)
        assert mae_val < 0.001

    def test_very_large_values(self):
        y_true = np.array([1e6, 2e6])
        y_pred = np.array([1.1e6, 2.1e6])

        mae_val = mae(y_true, y_pred)
        assert mae_val == 1e5
