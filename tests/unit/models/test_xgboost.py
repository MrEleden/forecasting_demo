"""
Unit tests for XGBoost forecasting model.
"""

import numpy as np
import pandas as pd
import pytest

from ml_portfolio.models.statistical.xgboost import XGBOOST_AVAILABLE, XGBoostForecaster


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostForecaster:
    """Test XGBoost forecasting model."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1
        return X, y

    def test_initialization_with_defaults(self):
        """Test model initialization with defaults."""
        model = XGBoostForecaster()

        assert model.n_estimators == 500
        assert model.learning_rate == 0.05
        assert model.max_depth == 8
        assert model.random_state == 42
        assert model.model is None
        assert not model.is_fitted

    def test_initialization_with_custom_params(self):
        """Test model with custom parameters."""
        model = XGBoostForecaster(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=123,
        )

        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 5
        assert model.random_state == 123

    def test_fit_with_numpy_arrays(self, sample_data):
        """Test fitting with numpy arrays."""
        X, y = sample_data
        model = XGBoostForecaster(n_estimators=10, verbosity=0)

        model._fit(X, y)

        assert model.is_fitted
        assert model.model is not None
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == X.shape[1]

    def test_predict_after_fit(self, sample_data):
        """Test predictions after fitting."""
        X, y = sample_data
        model = XGBoostForecaster(n_estimators=10, verbosity=0)

        model._fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (len(X),)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict before fit raises error."""
        X, y = sample_data
        model = XGBoostForecaster()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_with_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature_1": np.random.rand(50),
                "feature_2": np.random.rand(50),
                "feature_3": np.random.rand(50),
            }
        )
        y = df["feature_1"] * 2 + np.random.randn(50) * 0.1

        model = XGBoostForecaster(n_estimators=10, verbosity=0)
        model._fit(df, y)

        assert model.is_fitted
        assert model.feature_names_ == list(df.columns)

    def test_feature_importances(self, sample_data):
        """Test feature importances are available."""
        X, y = sample_data
        model = XGBoostForecaster(n_estimators=10, verbosity=0)

        model._fit(X, y)

        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == X.shape[1]
        assert np.all(model.feature_importances_ >= 0)

    def test_score_method(self, sample_data):
        """Test R^2 score calculation."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]

        model = XGBoostForecaster(n_estimators=50, verbosity=0)
        model._fit(X_train, y_train)

        score = model.score(X_test, y_test)

        assert isinstance(score, float)
        assert -1 <= score <= 1

    def test_get_params(self):
        """Test get_params returns parameters."""
        model = XGBoostForecaster(n_estimators=100, learning_rate=0.1)

        params = model.get_params()

        assert isinstance(params, dict)
        assert params["n_estimators"] == 100
        assert params["learning_rate"] == 0.1

    def test_set_params(self):
        """Test set_params updates parameters."""
        model = XGBoostForecaster(n_estimators=100)

        model.set_params(n_estimators=200, learning_rate=0.2)

        assert model.n_estimators == 200
        assert model.learning_rate == 0.2

    def test_with_eval_set(self, sample_data):
        """Test with validation set."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]

        model = XGBoostForecaster(n_estimators=50, early_stopping_rounds=10, verbosity=0)

        model._fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert model.is_fitted
        assert model.model.best_iteration is not None

    def test_reproducibility(self, sample_data):
        """Test reproducibility with same random_state."""
        X, y = sample_data

        model1 = XGBoostForecaster(n_estimators=10, random_state=42, verbosity=0)
        model2 = XGBoostForecaster(n_estimators=10, random_state=42, verbosity=0)

        model1._fit(X, y)
        model2._fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)
