"""
Unit tests for Hydra configuration system.
"""

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


class TestHydraConfigs:
    """Test Hydra configuration loading and validation."""

    def test_walmart_config_loads(self, project_root):
        """Test walmart.yaml config loads correctly."""
        config_dir = project_root / "src" / "ml_portfolio" / "conf"

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(config_name="walmart")

            assert isinstance(cfg, DictConfig)
            assert "dataset_factory" in cfg
            assert "model" in cfg
            assert "metrics" in cfg

    def test_model_configs_load(self, project_root):
        """Test all model configs can be loaded."""
        config_dir = project_root / "src" / "ml_portfolio" / "conf"
        models = ["lightgbm", "xgboost", "catboost", "random_forest"]

        for model_name in models:
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
                cfg = compose(config_name="walmart", overrides=[f"model={model_name}"])

                assert cfg.model._target_ is not None
                assert "n_estimators" in cfg.model or "iterations" in cfg.model

    def test_optuna_configs_load(self, project_root):
        """Test Optuna search space configs load correctly."""
        config_dir = project_root / "src" / "ml_portfolio" / "conf"
        models = ["lightgbm", "xgboost", "catboost", "random_forest"]

        for model_name in models:
            optuna_config_path = config_dir / "optuna" / f"{model_name}.yaml"
            assert optuna_config_path.exists(), f"Optuna config missing for {model_name}"

    def test_config_overrides(self, project_root):
        """Test config overrides work correctly."""
        config_dir = project_root / "src" / "ml_portfolio" / "conf"

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(config_name="walmart", overrides=["model.n_estimators=999", "seed=123"])

            assert cfg.model.n_estimators == 999
            assert cfg.seed == 123

    def test_metrics_config(self, project_root):
        """Test metrics configuration."""
        config_dir = project_root / "src" / "ml_portfolio" / "conf"

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(config_name="walmart")

            assert "metrics" in cfg
            assert cfg.metrics.primary_metric == "MAPE"
            assert cfg.metrics.minimize is True
            assert len(cfg.metrics.metrics) >= 3  # MAPE, RMSE, MAE
