"""
Integration tests for full training pipeline.
"""

import pytest

pytest.skip("Integration tests disabled - requires full environment setup", allow_module_level=True)


class TestTrainingPipeline:
    """Test end-to-end training pipeline."""

    @pytest.mark.slow
    def test_lightgbm_training(self, project_root, tmp_path):
        """Test LightGBM model training pipeline."""
        import subprocess

        # Run training with minimal trials
        result = subprocess.run(
            [
                "python",
                str(project_root / "src" / "ml_portfolio" / "training" / "train.py"),
                "--config-name",
                "walmart",
                "model=lightgbm",
                "model.n_estimators=10",  # Fast training
                f"checkpoint_dir={tmp_path}",
                "use_mlflow=false",  # Skip MLflow for test
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Training failed: {result.stderr}"

        # Check checkpoint was created
        checkpoint_files = list(tmp_path.glob("*.pkl"))
        assert len(checkpoint_files) > 0, "No checkpoint file created"

    @pytest.mark.slow
    def test_optuna_optimization(self, project_root, tmp_path):
        """Test Optuna optimization runs."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                str(project_root / "src" / "ml_portfolio" / "training" / "train.py"),
                "--multirun",
                "--config-name",
                "walmart",
                "model=lightgbm",
                "use_optuna=true",
                "hydra/sweeper=optuna",
                "+optuna=lightgbm",
                "hydra.sweeper.n_trials=2",  # Minimal trials
                f"checkpoint_dir={tmp_path}",
                "use_mlflow=false",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        assert result.returncode == 0, f"Optimization failed: {result.stderr}"
        assert "Best parameters" in result.stdout


class TestMLflowIntegration:
    """Test MLflow tracking integration."""

    def test_mlflow_logging(self, project_root, tmp_path):
        """Test MLflow experiment logging."""
        import subprocess

        import mlflow

        mlflow_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")

        result = subprocess.run(
            [
                "python",
                str(project_root / "src" / "ml_portfolio" / "training" / "train.py"),
                "--config-name",
                "walmart",
                "model=lightgbm",
                "model.n_estimators=10",
                f"checkpoint_dir={tmp_path}",
                "use_mlflow=true",
                "experiment_name=test_experiment",
            ],
            capture_output=True,
            text=True,
            env={"MLFLOW_TRACKING_URI": f"file://{mlflow_dir}"},
        )

        assert result.returncode == 0

        # Check MLflow recorded the run
        client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{mlflow_dir}")
        experiments = client.search_experiments()
        assert len(experiments) > 0


class TestDataPipeline:
    """Test data loading and processing pipeline."""

    def test_walmart_data_loading(self, project_root):
        """Test Walmart dataset can be loaded."""
        from ml_portfolio.data.dataset_factory import DatasetFactory

        data_path = project_root / "projects" / "retail_sales_walmart" / "data" / "raw" / "Walmart.csv"

        if not data_path.exists():
            pytest.skip("Walmart data not available")

        factory = DatasetFactory(
            data_path=str(data_path),
            target_column="Weekly_Sales",
            timestamp_column="Date",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        train, val, test = factory.get_datasets()

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert "Weekly_Sales" in train.columns
