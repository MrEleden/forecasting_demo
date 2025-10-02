"""
MLflow utilities for experiment tracking and model logging.

This module provides utilities to integrate MLflow with the forecasting pipeline.
"""

import logging
import os
import pickle
import tempfile
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking utilities for forecasting models.

    Handles:
    - Experiment setup and run management
    - Parameter and metric logging
    - Model artifact logging
    - Prediction and plot artifacts
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize MLflow tracker with configuration.

        Args:
            cfg: MLflow configuration from Hydra
        """
        self.cfg = cfg
        self.run_id = None
        self.experiment_id = None

        # Setup MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking URI and experiment."""
        # Set tracking URI
        if self.cfg.tracking_uri:
            mlflow.set_tracking_uri(self.cfg.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.cfg.tracking_uri}")
        else:
            logger.info("Using default MLflow tracking URI (local file://mlruns)")

        # Set experiment
        mlflow.set_experiment(self.cfg.experiment_name)
        logger.info(f"MLflow experiment: {self.cfg.experiment_name}")

        # Enable auto-logging if configured
        if self.cfg.enable_auto_logging:
            self._enable_auto_logging()

    def _enable_auto_logging(self):
        """Enable MLflow auto-logging for supported frameworks."""
        if self.cfg.auto_log_frameworks.sklearn:
            mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=self.cfg.log_model)
            logger.info("Enabled MLflow auto-logging for sklearn")

        if self.cfg.auto_log_frameworks.pytorch:
            try:
                mlflow.pytorch.autolog(log_models=self.cfg.log_model, log_every_n_epoch=1, log_every_n_step=None)
                logger.info("Enabled MLflow auto-logging for PyTorch")
            except ImportError:
                logger.warning("PyTorch not available, skipping auto-logging")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run (auto-generated if None)
            tags: Additional tags for the run (these take priority)
        """
        # Start with default tags from config
        all_tags = dict(self.cfg.tags) if self.cfg.tags else {}

        # Custom tags take priority (these come from train.py with cfg-derived info)
        if tags:
            all_tags.update(tags)

        # Use configured run name or custom name
        final_run_name = run_name or self.cfg.run_name

        # Start run
        mlflow.start_run(run_name=final_run_name, tags=all_tags)
        self.run_id = mlflow.active_run().info.run_id
        self.experiment_id = mlflow.active_run().info.experiment_id

        logger.info(f"Started MLflow run: {self.run_id} with {len(all_tags)} tags")

    def log_config(self, config: DictConfig):
        """
        Log Hydra configuration parameters.

        Args:
            config: Hydra configuration object
        """
        if not self.cfg.log_params:
            return

        # Convert config to flat dictionary for MLflow
        params = self._flatten_config(config)

        # Add configuration metadata derived from the config itself
        try:
            config_metadata = {
                "config.model.target": config.model._target_,
                "config.dataset.target": config.dataset_factory._target_,
                "config.total_sections": len(config.keys()),
                "config.has_optimizer": str(config.optimizer is not None),
                "config.has_scheduler": str(getattr(config, "scheduler", None) is not None),
                "config.training.max_epochs": str(config.training.max_epochs),
                "config.mlflow.enabled": "true",
            }

            # Merge metadata with config parameters
            params.update(config_metadata)

        except Exception as e:
            logger.warning(f"Failed to extract config metadata: {e}")

        # Log parameters in batches (MLflow has limits)
        batch_size = 100
        for i in range(0, len(params), batch_size):
            batch = dict(list(params.items())[i : i + batch_size])
            mlflow.log_params(batch)

        logger.info(f"Logged {len(params)} configuration parameters (including metadata)")

    def _flatten_config(self, config: DictConfig, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten nested configuration into dot-separated keys.

        Args:
            config: Configuration object
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []

        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                # Convert to string for MLflow compatibility
                items.append((new_key, str(v)))

        return dict(items)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number for time series metrics
        """
        if not self.cfg.log_metrics:
            return

        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_model(
        self,
        model: Any,
        model_name: str = "model",
        input_example: Optional[np.ndarray] = None,
        signature: Optional[Any] = None,
    ):
        """
        Log model artifact to MLflow.

        Args:
            model: Trained model object
            model_name: Name for the model artifact
            input_example: Example input for model signature
            signature: MLflow model signature
        """
        if not self.cfg.log_model:
            return

        try:
            # Try sklearn auto-logging first
            if hasattr(model, "fit") and hasattr(model, "predict"):
                mlflow.sklearn.log_model(
                    sk_model=model, artifact_path=model_name, input_example=input_example, signature=signature
                )
                logger.info(f"Logged sklearn model: {model_name}")
            else:
                # Fallback to pickle
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                    pickle.dump(model, f)
                    mlflow.log_artifact(f.name, f"{model_name}.pkl")
                    os.unlink(f.name)
                logger.info(f"Logged model as pickle: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to log model {model_name}: {e}")

    def log_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "test", plot: bool = True):
        """
        Log prediction results and plots.

        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of the dataset (train/val/test)
            plot: Whether to create and log plots
        """
        if not self.cfg.log_predictions:
            return

        # Log predictions as CSV
        predictions_df = pd.DataFrame(
            {
                "y_true": y_true.flatten() if hasattr(y_true, "flatten") else y_true,
                "y_pred": y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred,
                "residual": (y_true - y_pred).flatten() if hasattr(y_true, "flatten") else (y_true - y_pred),
            }
        )

        # Use a more robust temporary file approach
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, f"{dataset_name}_predictions.csv")
        try:
            predictions_df.to_csv(temp_file, index=False)
            mlflow.log_artifact(temp_file, f"predictions/{dataset_name}_predictions.csv")
        finally:
            # Clean up with proper error handling
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                os.rmdir(temp_dir)
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")

        # Create and log plots
        if plot and self.cfg.log_plots:
            self._log_prediction_plots(y_true, y_pred, dataset_name)

        logger.info(f"Logged predictions for {dataset_name} dataset")

    def _log_prediction_plots(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str):
        """Create and log prediction visualization plots."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Actual vs Predicted scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{dataset_name.title()} Set: Actual vs Predicted")

            plot1_path = os.path.join(temp_dir, f"{dataset_name}_actual_vs_predicted.png")
            plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot1_path, f"plots/{dataset_name}_actual_vs_predicted.png")
            plt.close()

            # Residuals plot
            residuals = y_true - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.title(f"{dataset_name.title()} Set: Residuals Plot")

            plot2_path = os.path.join(temp_dir, f"{dataset_name}_residuals.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot2_path, f"plots/{dataset_name}_residuals.png")
            plt.close()

            # Time series plot (if data looks like time series)
            if len(y_true) > 10:
                plt.figure(figsize=(12, 6))
                plt.plot(y_true, label="Actual", alpha=0.8)
                plt.plot(y_pred, label="Predicted", alpha=0.8)
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title(f"{dataset_name.title()} Set: Time Series Comparison")
                plt.legend()

                plot3_path = os.path.join(temp_dir, f"{dataset_name}_time_series.png")
                plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
                mlflow.log_artifact(plot3_path, f"plots/{dataset_name}_time_series.png")
                plt.close()

        except Exception as e:
            logger.warning(f"Failed to create prediction plots: {e}")

        finally:
            # Clean up temporary directory and files
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

    def log_feature_importance(self, model: Any, feature_names: Optional[list] = None):
        """
        Log feature importance if available.

        Args:
            model: Trained model
            feature_names: Names of features
        """
        if not self.cfg.log_feature_importance:
            return

        try:
            # Try to get feature importance
            importance = None
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_).flatten()

            if importance is not None:
                # Create feature importance dataframe
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(importance))]

                importance_df = pd.DataFrame(
                    {"feature": feature_names[: len(importance)], "importance": importance}
                ).sort_values("importance", ascending=False)

                # Log as CSV
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    importance_df.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, "feature_importance.csv")
                    os.unlink(f.name)

                # Create and log plot
                if self.cfg.log_plots:
                    plt.figure(figsize=(10, 6))
                    top_features = importance_df.head(20)  # Top 20 features
                    plt.barh(range(len(top_features)), top_features["importance"])
                    plt.yticks(range(len(top_features)), top_features["feature"])
                    plt.xlabel("Importance")
                    plt.title("Feature Importance")
                    plt.gca().invert_yaxis()

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        plt.savefig(f.name, dpi=300, bbox_inches="tight")
                        mlflow.log_artifact(f.name, "plots/feature_importance.png")
                        os.unlink(f.name)
                    plt.close()

                logger.info("Logged feature importance")

        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    def register_model(self, model_name: Optional[str] = None):
        """
        Register the logged model in MLflow Model Registry.

        Args:
            model_name: Name for registered model (auto-generated if None)
        """
        if not self.cfg.register_model:
            return

        try:
            final_model_name = model_name or self.cfg.model_name or f"forecasting_model_{self.run_id[:8]}"

            # Register model
            model_uri = f"runs:/{self.run_id}/model"
            mlflow.register_model(model_uri, final_model_name)

            logger.info(f"Registered model: {final_model_name}")

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")

    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
