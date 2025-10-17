"""
Unit tests for LightGBM forecasting model.
"""

import numpy as np
import pandas as pd
import pytest

from ml_portfolio.models.statistical.lightgbm import LIGHTGBM_AVAILABLE, LightGBMForecaster


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMForecaster:
    """Test LightGBM forecasting model."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1
        return X, y

    @pytest.fixture
    def sample_dataframe(self):
        """Generate sample DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature_1": np.random.rand(100),
                "feature_2": np.random.rand(100),
                "feature_3": np.random.rand(100),
            }
        )
        y = df["feature_1"] * 2 + df["feature_2"] * 3 + np.random.randn(100) * 0.1
        return df, y

    def test_initialization_with_defaults(self):
        """Test model can be initialized with default parameters."""
        model = LightGBMForecaster()

        assert model.n_estimators == 500
        assert model.learning_rate == 0.05
        assert model.max_depth == 8
        assert model.num_leaves == 31
        assert model.random_state == 42
        assert model.model is None
        assert not model.is_fitted

    def test_initialization_with_custom_params(self):
        """Test model with custom parameters."""
        model = LightGBMForecaster(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=20,
            random_state=123,
        )

        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 5
        assert model.num_leaves == 20
        assert model.random_state == 123

    def test_fit_with_numpy_arrays(self, sample_data):
        """Test fitting model with numpy arrays."""
        X, y = sample_data
        model = LightGBMForecaster(n_estimators=10, verbose=-1)

        model._fit(X, y)  # Call internal fit method directly for unit testing

        assert model.is_fitted
        assert model.model is not None
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == X.shape[1]
        assert model.feature_importances_ is not None

    def test_fit_with_dataframe(self, sample_dataframe):
        """Test fitting model with pandas DataFrame."""
        X_df, y = sample_dataframe
        model = LightGBMForecaster(n_estimators=10, verbose=-1)

        model._fit(X_df, y)

        assert model.is_fitted
        assert model.feature_names_ == list(X_df.columns)
        assert len(model.feature_importances_) == X_df.shape[1]

    def test_predict_after_fit(self, sample_data):
        """Test predictions after fitting."""
        X, y = sample_data
        model = LightGBMForecaster(n_estimators=10, verbose=-1)

        model._fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (len(X),)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict before fit raises error."""
        X, y = sample_data
        model = LightGBMForecaster()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_dataframe):
        """Test predictions with DataFrame input."""
        X_df, y = sample_dataframe
        model = LightGBMForecaster(n_estimators=10, verbose=-1)

        model._fit(X_df, y)
        predictions = model.predict(X_df)

        assert predictions.shape == (len(X_df),)
        assert isinstance(predictions, np.ndarray)

    def test_fit_with_eval_set(self, sample_data):
        """Test fitting with validation set."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]

        model = LightGBMForecaster(n_estimators=50, early_stopping_rounds=10, verbose=-1)

        model._fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert model.is_fitted
        # Model should have stopped early due to overfitting on small dataset
        assert model.model.best_iteration_ is not None

    def test_score_method(self, sample_data):
        """Test R^2 score calculation."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]

        model = LightGBMForecaster(n_estimators=50, verbose=-1)
        model._fit(X_train, y_train)

        score = model.score(X_test, y_test)

        assert isinstance(score, float)
        assert -1 <= score <= 1  # R^2 can be negative for bad fits

    def test_get_params(self):
        """Test get_params returns all parameters."""
        model = LightGBMForecaster(n_estimators=100, learning_rate=0.1, max_depth=5)

        params = model.get_params()

        assert isinstance(params, dict)
        assert params["n_estimators"] == 100
        assert params["learning_rate"] == 0.1
        assert params["max_depth"] == 5
        assert "random_state" in params

    def test_set_params(self):
        """Test set_params updates model parameters."""
        model = LightGBMForecaster(n_estimators=100)

        model.set_params(n_estimators=200, learning_rate=0.2)

        assert model.n_estimators == 200
        assert model.learning_rate == 0.2

    def test_feature_importances_after_fit(self, sample_data):
        """Test feature importances are available after fitting."""
        X, y = sample_data
        model = LightGBMForecaster(n_estimators=10, verbose=-1)

        model._fit(X, y)

        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == X.shape[1]
        assert np.all(model.feature_importances_ >= 0)
        # Feature importances should sum to a positive value
        assert model.feature_importances_.sum() > 0

    def test_with_2d_y_array(self, sample_data):
        """Test model handles 2D y array correctly."""
        X, y = sample_data
        y_2d = y.reshape(-1, 1)  # Make y 2D

        model = LightGBMForecaster(n_estimators=10, verbose=-1)
        model._fit(X, y_2d)

        predictions = model.predict(X)
        assert predictions.ndim == 1  # Output should be 1D

    def test_with_pandas_series_target(self, sample_dataframe):
        """Test model with pandas Series as target."""
        X_df, y_series = sample_dataframe

        model = LightGBMForecaster(n_estimators=10, verbose=-1)
        model._fit(X_df, pd.Series(y_series))

        predictions = model.predict(X_df)
        assert predictions.shape == (len(X_df),)

    def test_multiple_fits(self, sample_data):
        """Test model can be refitted multiple times."""
        X, y = sample_data
        model = LightGBMForecaster(n_estimators=10, verbose=-1)

        # First fit
        model._fit(X, y)
        predictions_1 = model.predict(X)

        # Second fit (should reinitialize)
        model._fit(X, y)
        predictions_2 = model.predict(X)

        # Both should work and be similar (same data/random_state)
        assert predictions_1.shape == predictions_2.shape
        np.testing.assert_allclose(predictions_1, predictions_2, rtol=0.1)

    def test_regularization_parameters(self):
        """Test that regularization parameters are set correctly."""
        model = LightGBMForecaster(reg_alpha=0.5, reg_lambda=1.0, n_estimators=10)

        params = model.get_params()
        assert params["reg_alpha"] == 0.5
        assert params["reg_lambda"] == 1.0

    def test_custom_kwargs(self, sample_data):
        """Test that custom kwargs are passed to LightGBM."""
        X, y = sample_data
        model = LightGBMForecaster(n_estimators=10, verbose=-1, extra_trees=True)  # custom param

        model._fit(X, y)
        assert model.is_fitted

    def test_zero_estimators_raises_error(self, sample_data):
        """Test that zero estimators causes an error."""
        X, y = sample_data
        model = LightGBMForecaster(n_estimators=0, verbose=-1)

        with pytest.raises(Exception):  # LightGBM will raise
            model._fit(X, y)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that same random_state gives reproducible results."""
        X, y = sample_data

        model1 = LightGBMForecaster(n_estimators=10, random_state=42, verbose=-1)
        model2 = LightGBMForecaster(n_estimators=10, random_state=42, verbose=-1)

        model1._fit(X, y)
        model2._fit(X, y)

        predictions1 = model1.predict(X)
        predictions2 = model2.predict(X)

        np.testing.assert_array_almost_equal(predictions1, predictions2)


class TestLightGBMImportError:
    """Test behavior when LightGBM is not available."""

    def test_import_error_message(self):
        """Test that helpful error is raised when LightGBM not available."""
        if LIGHTGBM_AVAILABLE:
            pytest.skip("LightGBM is installed")

        with pytest.raises(ImportError, match="LightGBM is not installed"):
            LightGBMForecaster()
