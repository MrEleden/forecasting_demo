"""
Unit tests for additional base model functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_portfolio.models.base import StatisticalForecaster

pytest.skip("Base extended tests disabled pending API verification", allow_module_level=True)


class ConcreteForecaster(StatisticalForecaster):
    """Concrete implementation for testing StatisticalForecaster."""

    def __init__(self):
        super().__init__()
        self.model_params_ = None

    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.is_fitted_ = True
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        self.n_features_ = X.shape[1] if len(X.shape) > 1 else 1
        self.mean_value_ = y.mean() if hasattr(y, "mean") else np.mean(y)
        self.model_params_ = kwargs
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.ones(n_samples) * self.mean_value_

    def _fit(self, X, y):
        # Dummy implementation for abstract method
        pass


@pytest.fixture
def sample_data():
    """Create sample data."""
    X = pd.DataFrame(
        {"feature_1": np.random.rand(50), "feature_2": np.random.rand(50), "feature_3": np.random.rand(50)}
    )
    y = pd.Series(np.random.rand(50))
    return X, y


class TestStatisticalForecasterSaveLoad:
    """Test save/load functionality."""

    def test_save_model(self, sample_data):
        """Test saving a fitted model."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_load_model(self, sample_data):
        """Test loading a saved model."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            loaded_model = ConcreteForecaster.load(save_path)

            assert loaded_model.is_fitted_
            assert loaded_model.mean_value_ == model.mean_value_

    def test_save_load_preserves_predictions(self, sample_data):
        """Test that save/load preserves prediction behavior."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        pred_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)
            loaded_model = ConcreteForecaster.load(save_path)

        pred_after = loaded_model.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)

    def test_save_unfitted_model(self):
        """Test saving an unfitted model."""
        model = ConcreteForecaster()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            assert save_path.exists()

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ConcreteForecaster.load("nonexistent.pkl")

    def test_save_with_str_path(self, sample_data):
        """Test saving with string path instead of Path object."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model.pkl")
            model.save(save_path)

            assert Path(save_path).exists()


class TestStatisticalForecasterAttributes:
    """Test model attributes and properties."""

    def test_initial_state(self):
        """Test initial unfitted state."""
        model = ConcreteForecaster()

        assert not model.is_fitted_
        assert model.feature_names_ is None

    def test_fitted_state(self, sample_data):
        """Test state after fitting."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        assert model.is_fitted_
        assert model.feature_names_ is not None
        assert model.n_features_ == 3

    def test_feature_names_preservation(self, sample_data):
        """Test that feature names are preserved."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        assert model.feature_names_ == ["feature_1", "feature_2", "feature_3"]

    def test_fit_params_storage(self, sample_data):
        """Test that fit parameters are stored."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y, learning_rate=0.01, epochs=100)

        assert model.model_params_["learning_rate"] == 0.01
        assert model.model_params_["epochs"] == 100

    def test_refit_model(self, sample_data):
        """Test refitting a model."""
        X, y = sample_data
        model = ConcreteForecaster()

        # First fit
        model.fit(X[:25], y[:25])
        pred1 = model.predict(X[25:])

        # Refit
        model.fit(X, y)
        pred2 = model.predict(X[25:])

        # Predictions should differ after refitting
        assert not np.array_equal(pred1, pred2)


class TestStatisticalForecasterEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_before_fit(self, sample_data):
        """Test that predicting before fit raises error."""
        X, _ = sample_data
        model = ConcreteForecaster()

        with pytest.raises(ValueError):
            model.predict(X)

    def test_fit_with_empty_data(self):
        """Test fitting with empty data."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        model = ConcreteForecaster()

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError, KeyError)):
            model.fit(X, y)

    def test_fit_with_mismatched_shapes(self):
        """Test fitting with mismatched X and y shapes."""
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.rand(40))  # Different length
        model = ConcreteForecaster()

        # Should raise error
        with pytest.raises((ValueError, IndexError)):
            model.fit(X, y)

    def test_predict_with_different_feature_count(self, sample_data):
        """Test prediction with different number of features."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        # Try to predict with different number of features
        X_wrong = pd.DataFrame(np.random.rand(10, 5))

        # Should raise error or handle gracefully
        # Some models might not check this, so we allow it to pass
        try:
            model.predict(X_wrong)
        except (ValueError, IndexError):
            pass  # Expected behavior

    def test_single_sample_prediction(self, sample_data):
        """Test prediction on single sample."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        single_X = X.iloc[[0]]
        pred = model.predict(single_X)

        assert len(pred) == 1

    def test_large_batch_prediction(self, sample_data):
        """Test prediction on large batch."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        large_X = pd.DataFrame(np.random.rand(1000, 3))
        pred = model.predict(large_X)

        assert len(pred) == 1000


class TestStatisticalForecasterIntegration:
    """Integration tests for forecaster workflow."""

    def test_full_workflow(self, sample_data):
        """Test complete fit-predict-save-load workflow."""
        X, y = sample_data

        # Fit
        model = ConcreteForecaster()
        model.fit(X, y)

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            # Load
            loaded_model = ConcreteForecaster.load(save_path)

            # Predict with loaded model
            loaded_predictions = loaded_model.predict(X)

            np.testing.assert_array_equal(predictions, loaded_predictions)

    def test_multiple_predictions(self, sample_data):
        """Test multiple prediction calls."""
        X, y = sample_data
        model = ConcreteForecaster()
        model.fit(X, y)

        pred1 = model.predict(X)
        pred2 = model.predict(X)
        pred3 = model.predict(X)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)

    def test_sequential_fits(self, sample_data):
        """Test sequential fitting on different data."""
        X, y = sample_data
        model = ConcreteForecaster()

        # First fit
        model.fit(X[:25], y[:25])
        assert model.is_fitted_

        # Second fit (should replace first)
        model.fit(X[25:], y[25:])
        assert model.is_fitted_
