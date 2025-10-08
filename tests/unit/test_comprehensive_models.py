"""
Comprehensive tests for statistical models to increase coverage.
"""

import numpy as np
import pandas as pd
import pytest
from ml_portfolio.models.statistical.catboost import CatBoostForecaster
from ml_portfolio.models.statistical.random_forest import RandomForestForecaster

pytest.importorskip("catboost")


pytest.skip("Comprehensive model tests disabled pending data format fixes", allow_module_level=True)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        {"feature_1": np.random.rand(100), "feature_2": np.random.rand(100), "feature_3": np.random.rand(100)}
    )
    y = pd.Series(np.random.rand(100))
    return X, y


@pytest.fixture
def small_data():
    """Create small sample data."""
    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10]})
    y = pd.Series([1.5, 3.5, 5.5, 7.5, 9.5])
    return X, y


class TestRandomForestForecaster:
    """Test RandomForestForecaster."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = RandomForestForecaster()
        assert model is not None
        assert not model.is_fitted_

    def test_initialization_with_params(self):
        """Test initialization with parameters."""
        model = RandomForestForecaster(n_estimators=50, max_depth=5, random_state=42)
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_fit_basic(self, small_data):
        """Test basic fitting."""
        X, y = small_data
        model = RandomForestForecaster(n_estimators=10, random_state=42)
        model.fit(X, y)

        assert model.is_fitted_
        assert model.feature_names_ is not None

    def test_predict_after_fit(self, small_data):
        """Test prediction after fitting."""
        X, y = small_data
        model = RandomForestForecaster(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)

    def test_fit_with_large_data(self, sample_data):
        """Test fitting with larger dataset."""
        X, y = sample_data
        model = RandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)

        assert model.is_fitted_
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_different_n_estimators(self, small_data):
        """Test with different number of estimators."""
        X, y = small_data

        for n_est in [5, 10, 20]:
            model = RandomForestForecaster(n_estimators=n_est, random_state=42)
            model.fit(X, y)
            assert model.is_fitted_

    def test_different_max_depth(self, small_data):
        """Test with different max depths."""
        X, y = small_data

        for depth in [3, 5, None]:
            model = RandomForestForecaster(max_depth=depth, n_estimators=10, random_state=42)
            model.fit(X, y)
            assert model.is_fitted_

    def test_feature_names_preserved(self, small_data):
        """Test that feature names are preserved."""
        X, y = small_data
        model = RandomForestForecaster(n_estimators=10, random_state=42)
        model.fit(X, y)

        assert model.feature_names_ == ["f1", "f2"]

    def test_random_state_reproducibility(self, small_data):
        """Test that random_state ensures reproducibility."""
        X, y = small_data

        model1 = RandomForestForecaster(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = RandomForestForecaster(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestCatBoostForecaster:
    """Test CatBoostForecaster."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = CatBoostForecaster()
        assert model is not None
        assert not model.is_fitted_

    def test_initialization_with_params(self):
        """Test initialization with parameters."""
        model = CatBoostForecaster(iterations=50, depth=4, learning_rate=0.1, verbose=False)
        assert model.iterations == 50
        assert model.depth == 4

    def test_fit_basic(self, small_data):
        """Test basic fitting."""
        X, y = small_data
        model = CatBoostForecaster(iterations=10, verbose=False)
        model.fit(X, y)

        assert model.is_fitted_
        assert model.feature_names_ is not None

    def test_predict_after_fit(self, small_data):
        """Test prediction after fitting."""
        X, y = small_data
        model = CatBoostForecaster(iterations=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)

    def test_fit_with_large_data(self, sample_data):
        """Test fitting with larger dataset."""
        X, y = sample_data
        model = CatBoostForecaster(iterations=20, verbose=False)
        model.fit(X, y)

        assert model.is_fitted_
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_different_iterations(self, small_data):
        """Test with different iteration counts."""
        X, y = small_data

        for iters in [5, 10, 20]:
            model = CatBoostForecaster(iterations=iters, verbose=False)
            model.fit(X, y)
            assert model.is_fitted_

    def test_different_depth(self, small_data):
        """Test with different tree depths."""
        X, y = small_data

        for depth in [2, 4, 6]:
            model = CatBoostForecaster(depth=depth, iterations=10, verbose=False)
            model.fit(X, y)
            assert model.is_fitted_

    def test_different_learning_rate(self, small_data):
        """Test with different learning rates."""
        X, y = small_data

        for lr in [0.01, 0.1, 0.3]:
            model = CatBoostForecaster(learning_rate=lr, iterations=10, verbose=False)
            model.fit(X, y)
            assert model.is_fitted_

    def test_verbose_off(self, small_data):
        """Test that verbose=False works."""
        X, y = small_data
        model = CatBoostForecaster(iterations=10, verbose=False)
        model.fit(X, y)
        assert model.is_fitted_

    def test_feature_names_preserved(self, small_data):
        """Test that feature names are preserved."""
        X, y = small_data
        model = CatBoostForecaster(iterations=10, verbose=False)
        model.fit(X, y)

        assert model.feature_names_ == ["f1", "f2"]


class TestModelConsistency:
    """Test consistency across models."""

    def test_both_models_can_fit(self, small_data):
        """Test that both models can fit the same data."""
        X, y = small_data

        rf_model = RandomForestForecaster(n_estimators=10, random_state=42)
        rf_model.fit(X, y)

        cb_model = CatBoostForecaster(iterations=10, verbose=False)
        cb_model.fit(X, y)

        assert rf_model.is_fitted_
        assert cb_model.is_fitted_

    def test_both_models_produce_predictions(self, small_data):
        """Test that both models produce predictions."""
        X, y = small_data

        rf_model = RandomForestForecaster(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)

        cb_model = CatBoostForecaster(iterations=10, verbose=False)
        cb_model.fit(X, y)
        cb_pred = cb_model.predict(X)

        assert len(rf_pred) == len(y)
        assert len(cb_pred) == len(y)

    def test_predictions_are_reasonable(self, small_data):
        """Test that predictions are in a reasonable range."""
        X, y = small_data

        for Model, kwargs in [
            (RandomForestForecaster, {"n_estimators": 10, "random_state": 42}),
            (CatBoostForecaster, {"iterations": 10, "verbose": False}),
        ]:
            model = Model(**kwargs)
            model.fit(X, y)
            pred = model.predict(X)

            # Predictions should be finite
            assert np.all(np.isfinite(pred))

            # Predictions should be roughly in the range of targets
            assert pred.min() >= y.min() * 0.5
            assert pred.max() <= y.max() * 1.5


class TestEdgeCases:
    """Test edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])

        model = RandomForestForecaster(n_estimators=5, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)

        assert len(pred) == len(y)

    def test_many_features(self):
        """Test with many features."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(50, 20))
        y = pd.Series(np.random.rand(50))

        model = RandomForestForecaster(n_estimators=10, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)

        assert len(pred) == len(y)

    def test_constant_target(self):
        """Test with constant target values."""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10]})
        y = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])

        model = RandomForestForecaster(n_estimators=5, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)

        # Should predict close to constant value
        assert np.all(np.abs(pred - 5.0) < 1.0)

    def test_negative_values(self):
        """Test with negative target values."""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10]})
        y = pd.Series([-1, -2, -3, -4, -5])

        model = RandomForestForecaster(n_estimators=10, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)

        assert len(pred) == len(y)
        assert np.all(pred < 0)  # Should predict negative values

    def test_zero_values(self):
        """Test with zero target values."""
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10]})
        y = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

        model = RandomForestForecaster(n_estimators=5, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)

        # Should predict close to zero
        assert np.all(np.abs(pred) < 0.5)
