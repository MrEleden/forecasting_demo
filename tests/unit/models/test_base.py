"""
Unit tests for base forecasting model classes.
"""

import numpy as np
import pytest
from ml_portfolio.models.base import StatisticalForecaster


class SimpleStatisticalModel(StatisticalForecaster):
    """Simple model for testing base class functionality."""

    def __init__(self, param1=10):
        super().__init__()
        self.param1 = param1
        self.coef_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Simple linear regression fit."""
        # Simple coefficient calculation
        self.coef_ = np.ones(X.shape[1])
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Simple prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return X @ self.coef_


class TestBaseForecaster:
    """Test base forecaster functionality."""

    def test_initialization(self):
        """Test base model initialization."""
        model = SimpleStatisticalModel(param1=5)

        assert model.param1 == 5
        assert not model.is_fitted
        assert model.coef_ is None

    def test_save_and_load(self, tmp_path):
        """Test model can be saved and loaded."""
        X = np.random.rand(30, 3)
        y = np.random.rand(30)

        # Create and fit model
        model = SimpleStatisticalModel(param1=42)
        model._fit(X, y)

        # Save model
        model_path = tmp_path / "model.pkl"
        model.save(model_path)

        # Load into new model
        loaded_model = SimpleStatisticalModel()
        loaded_model.load(model_path)

        # Verify parameters and state persisted
        assert loaded_model.param1 == 42
        assert loaded_model.is_fitted
        assert loaded_model.coef_ is not None

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        model = SimpleStatisticalModel()

        # Path with nested directories that don't exist
        model_path = tmp_path / "nested" / "dirs" / "model.pkl"

        model.save(model_path)

        assert model_path.exists()

    def test_str_representation(self):
        """Test str representation."""
        model = SimpleStatisticalModel()

        str_repr = str(model)

        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


class TestStatisticalForecaster:
    """Test StatisticalForecaster functionality."""

    def test_fit_marks_as_fitted(self):
        """Test that _fit properly marks model as fitted."""
        X = np.random.rand(50, 3)
        y = np.random.rand(50)

        model = SimpleStatisticalModel()
        model._fit(X, y)

        assert model.is_fitted

    def test_predict_before_fit_raises_error(self):
        """Test prediction before fitting raises error."""
        X = np.random.rand(10, 3)

        model = SimpleStatisticalModel()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_after_fit(self):
        """Test predictions work after fitting."""
        X_train = np.random.rand(50, 3)
        y_train = np.random.rand(50)
        X_test = np.random.rand(10, 3)

        model = SimpleStatisticalModel()
        model._fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))

    def test_model_parameters_persist(self):
        """Test that model parameters persist after fitting."""
        X = np.random.rand(30, 5)
        y = np.random.rand(30)

        model = SimpleStatisticalModel(param1=25)
        model._fit(X, y)

        assert model.param1 == 25
        assert model.coef_ is not None

    def test_predictions_are_deterministic(self):
        """Test that predictions are consistent."""
        X_train = np.random.rand(50, 3)
        y_train = np.random.rand(50)
        X_test = np.random.rand(10, 3)

        model = SimpleStatisticalModel()
        model._fit(X_train, y_train)

        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)
