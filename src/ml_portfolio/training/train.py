"""
Main Training Script with Hydra Configuration and MLflow Tracking.

This script supports both single runs and Hydra sweeper optimization (including Optuna).

Usage:

Single Training Run:
    # Run with default configuration
    python src/ml_portfolio/training/train.py

    # Override specific parameters
    python src/ml_portfolio/training/train.py dataset_factory=walmart model=arima

Multi-run with Grid Search:
    # Test multiple models
    python src/ml_portfolio/training/train.py -m model=arima,lstm,random_forest dataset_factory=walmart

Optuna Hyperparameter Optimization (Hydra Sweeper):
    # ARIMA optimization with Optuna sweeper
    python src/ml_portfolio/training/train.py -m \\
        model=arima \\
        dataset_factory=walmart \\
        hydra/sweeper=optuna \\
        hydra.sweeper.n_trials=20 \\
        'model.order=choice([1,1,1], [2,1,1], [1,1,2])' \\
        'model.seasonal_order=choice([[1,1,1,12], [0,1,1,12]])'

    # Random Forest optimization
    python src/ml_portfolio/training/train.py -m \\
        model=random_forest \\
        dataset_factory=walmart \\
        hydra/sweeper=optuna \\
        hydra.sweeper.n_trials=10 \\
        'model.n_estimators=int(interval(50, 200))' \\
        'model.max_depth=int(interval(3, 15))'

MLflow Configuration:
    # Use different MLflow configuration
    python src/ml_portfolio/training/train.py mlflow=production

Example project-specific wrapper:
    # projects/retail_sales_walmart/scripts/train.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

    from ml_portfolio.training.train import main

    if __name__ == "__main__":
        main()
"""

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ml_portfolio.training.engine import TrainingEngine

try:
    from ml_portfolio.utils.mlflow_utils import MLflowTracker

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")


# Add src and project root to path for imports
src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(cfg: DictConfig, project_name: str = "ML Portfolio Training") -> float:
    """
    Main training pipeline with Hydra configuration and MLflow tracking.

    This function trains a model and returns the primary metric value.
    When used with Hydra sweeper (like Optuna), it returns validation metrics for optimization.
    When used in single run mode, it evaluates on test set.

    Args:
        cfg: Hydra configuration object (automatically loaded from conf/)
        project_name: Name of the project for logging

    Returns:
        Primary metric value (validation for sweeper, test for single run)
    """
    logger.info("=" * 60)
    logger.info(f"{project_name}")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Detect if running in multirun/sweep mode using hydra config in cfg
    is_multirun = False
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "mode"):
        is_multirun = cfg.hydra.mode == "MULTIRUN"
    elif hasattr(cfg, "hydra") and hasattr(cfg.hydra, "job") and hasattr(cfg.hydra.job, "num"):
        # Alternative: check if job number exists (indicates multirun)
        is_multirun = cfg.hydra.job.num is not None

    # Initialize MLflow tracking
    mlflow_tracker = None
    if MLFLOW_AVAILABLE and hasattr(cfg, "mlflow"):
        # Skip MLflow for individual sweep trials to avoid clutter
        if not is_multirun:
            try:
                mlflow_tracker = MLflowTracker(cfg.mlflow)

                # Extract information directly from cfg
                model_name = cfg.model._target_.split(".")[-1]
                dataset_name = cfg.dataset_factory._target_.split(".")[-1]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Generate informative run name
                run_name = f"{model_name}_{dataset_name}_{timestamp}"

                # Start MLflow run with comprehensive tags derived from cfg
                mlflow_tracker.start_run(
                    run_name=run_name,
                    tags={
                        "model_type": model_name,
                        "dataset": dataset_name,
                        "config_resolved": "true",
                        "hydra_version": hydra.__version__,
                        "total_config_params": str(len(OmegaConf.to_container(cfg, resolve=True))),
                        "experiment_framework": "hydra_mlflow",
                        "model_target": cfg.model._target_,
                        "dataset_target": cfg.dataset_factory._target_,
                        "training_epochs": str(cfg.training.max_epochs),
                        "optimizer_used": cfg.optimizer._target_ if cfg.optimizer else "none",
                        "multirun": str(is_multirun),
                    },
                )

                # Log configuration
                mlflow_tracker.log_config(cfg)

            except Exception as e:
                logger.warning(f"Failed to initialize MLflow tracking: {e}")
                mlflow_tracker = None

    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        logger.info(f"Random seed set to {cfg.seed}")

    # ========================================================================
    # 1. Create datasets using factory pattern
    # ========================================================================
    logger.info(f"Creating dataset factory: {cfg.dataset_factory._target_}")

    # Create dataset factory and get the three dataset splits
    dataset_factory = hydra.utils.instantiate(cfg.dataset_factory)
    train_dataset, val_dataset, test_dataset = dataset_factory.create_datasets()

    logger.info(f"Created datasets - train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # Log dataset info from the train dataset (all should have same metadata)
    n_features = None
    feature_names = None
    if hasattr(train_dataset, "get_data_info"):
        info = train_dataset.get_data_info()
        logger.info(f"Dataset info: {info}")
        n_features = info.get("n_features")
        feature_names = info.get("feature_names", [])

    # ========================================================================
    # 2. instantiate dataloaders for each split
    # ========================================================================
    dataloader_train = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
    dataloader_val = hydra.utils.instantiate(cfg.dataloader, dataset=val_dataset)
    dataloader_test = hydra.utils.instantiate(cfg.dataloader, dataset=test_dataset)

    # ========================================================================
    # 3. Instantiate model from configuration
    # ========================================================================
    logger.info(f"Instantiating model: {cfg.model._target_}")
    # For models that need input_size, override with dataset info
    model_kwargs = {}
    if n_features is not None and "input_size" in cfg.model:
        model_kwargs["input_size"] = n_features
        logger.info(f"Overriding model input_size with dataset n_features: {n_features}")
    model = hydra.utils.instantiate(cfg.model, **model_kwargs)
    logger.info(f"Model created: {model.__class__.__name__}")

    # ========================================================================
    # 4. Instantiate optimizer (for PyTorch models)
    # ========================================================================
    optimizer = None
    if hasattr(cfg, "optimizer") and cfg.optimizer is not None:
        try:
            # For PyTorch models, optimizer needs model parameters
            if hasattr(model, "parameters"):
                logger.info(f"Instantiating optimizer: {cfg.optimizer._target_}")
                optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
                logger.info(f"Optimizer created: {optimizer.__class__.__name__}")
            else:
                logger.info("Model does not have parameters() - skipping optimizer (sklearn model)")
        except Exception as e:
            logger.warning(f"Failed to instantiate optimizer: {e}")

    # ========================================================================
    # 5. Instantiate scheduler (for PyTorch models)
    # ========================================================================
    scheduler = None
    if hasattr(cfg, "scheduler") and cfg.scheduler is not None and optimizer is not None:
        try:
            logger.info(f"Instantiating scheduler: {cfg.scheduler._target_}")
            scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
            logger.info(f"Scheduler created: {scheduler.__class__.__name__}")
        except Exception as e:
            logger.warning(f"Failed to instantiate scheduler: {e}")

    # ========================================================================
    # 6. Instantiate metrics from configuration
    # ========================================================================
    logger.info("Instantiating metrics...")
    metrics = {}
    for metric_name, metric_cfg in cfg.metrics.items():
        # Skip simple config values that aren't metric objects
        if metric_name in ["primary_metric", "minimize"]:
            continue

        try:
            metrics[metric_name] = hydra.utils.instantiate(metric_cfg)
            logger.info(f"  - {metric_name}: {metric_cfg._target_}")
        except Exception as e:
            logger.warning(f"  - Failed to instantiate {metric_name}: {e}")

    # ========================================================================
    # 7. Extract training parameters and create training engine with MLflow
    # ========================================================================
    logger.info("Creating TrainingEngine with datasets...")

    # Extract training parameters from config
    training_config = cfg.training if hasattr(cfg, "training") else {}
    max_epochs = training_config.get("max_epochs", 100)
    patience = training_config.get("patience", 10)
    early_stopping = training_config.get("early_stopping", True)
    verbose = training_config.get("verbose", True)
    monitor_metric = training_config.get("monitor_metric", "val_loss")
    min_delta = training_config.get("min_delta", 1e-4)

    logger.info(f"Training parameters: max_epochs={max_epochs}, early_stopping={early_stopping}, patience={patience}")

    # Create training engine
    engine = TrainingEngine(
        model=model,
        metrics=metrics,
        train_dataset=dataloader_train,
        val_dataset=dataloader_val,
        test_dataset=dataloader_test,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        patience=patience,
        early_stopping=early_stopping,
        verbose=verbose and not is_multirun,  # Reduce verbosity for sweep trials
        monitor_metric=monitor_metric,
        min_delta=min_delta,
        mlflow_tracker=mlflow_tracker,
    )

    # ========================================================================
    # 8. Train model using training engine
    # ========================================================================
    if is_multirun:
        logger.info("Training for sweep optimization (validation-based)...")
    else:
        logger.info("Starting normal training...")

    # Train the model
    training_results = engine.train()

    if is_multirun:
        # For sweep/multirun: return validation metric for optimization
        primary_metric = cfg.metrics.get("primary_metric", "mse")

        # Try to get validation metric from training results
        validation_metric = None

        # Check if training_results has the metric directly
        val_metric_key = f"val_{primary_metric}"
        if hasattr(training_results, "get") and val_metric_key in training_results:
            validation_metric = training_results[val_metric_key]
        else:
            # Try to get from history (last epoch)
            history = training_results.get("history", {})
            val_metrics_history = history.get("val_metrics", [])
            if val_metrics_history:
                last_val_metrics = val_metrics_history[-1]
                # Try both prefixed and unprefixed versions
                validation_metric = last_val_metrics.get(f"val_{primary_metric}") or last_val_metrics.get(
                    primary_metric
                )

        if validation_metric is None:
            logger.error(f"Could not find validation metric '{primary_metric}' in training results")
            available_metrics = list(last_val_metrics.keys()) if "last_val_metrics" in locals() else "None"
            logger.error(f"Available metrics in last validation: {available_metrics}")
            raise ValueError(f"Validation metric '{primary_metric}' not found in training results")

        logger.info(f"Validation {primary_metric}: {validation_metric:.6f}")

        # End MLflow run for individual trials
        if mlflow_tracker:
            mlflow_tracker.end_run()

        return validation_metric

    else:
        # For single run: evaluate on test set and return test metric
        # ========================================================================
        # 9. Evaluate on test set and log to MLflow
        # ========================================================================
        test_metrics = engine.evaluate_test()

        # Log test metrics to MLflow
        if mlflow_tracker:
            try:
                # Log test metrics with 'test_' prefix to distinguish from training/validation
                mlflow_tracker.log_metrics(test_metrics)
                logger.info(f"Logged test metrics to MLflow: {test_metrics}")
            except Exception as e:
                logger.warning(f"Failed to log test metrics to MLflow: {e}")

        # Log model to MLflow
        if mlflow_tracker and mlflow_tracker.cfg.log_model:
            try:
                # Get sample input for model signature
                sample_X, _ = next(iter(dataloader_train))

                # Log the trained model
                mlflow_tracker.log_model(
                    model=model, model_name="model", input_example=sample_X[:5] if len(sample_X) > 5 else sample_X
                )

                # Log feature importance if available
                mlflow_tracker.log_feature_importance(model, feature_names)

                # Register model if configured
                mlflow_tracker.register_model()

            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")

        # End MLflow run
        if mlflow_tracker:
            mlflow_tracker.end_run()

        # Return primary metric from test set
        primary_metric = cfg.metrics.get("primary_metric", "mse")
        return test_metrics.get(primary_metric, 0.0)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main entry point for Hydra training with sweeper support.

    This function handles both single runs and multirun/sweep optimization:

    Single Run (default):
    - Trains model on training set
    - Uses validation set for early stopping
    - Evaluates and returns test set metrics

    Multirun/Sweep (with -m flag or sweeper):
    - Trains model on training set
    - Uses validation set for optimization metric
    - Returns validation metric for sweeper optimization

    Args:
        cfg: Hydra configuration from src/ml_portfolio/conf/

    Returns:
        Primary metric value (test for single run, validation for sweep)
    """
    # Just run the training pipeline - it will automatically detect multirun mode
    return train_pipeline(cfg)


if __name__ == "__main__":
    main()
