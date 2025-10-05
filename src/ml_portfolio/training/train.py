"""
Main training pipeline with Hydra configuration.

Follows Burkov's ML Engineering principles:
- Data leakage prevention (fit on train, transform all)
- Test set isolation (never touched until final evaluation)
- Config-driven design (Hydra)
- Reproducibility (seeds, logging)
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

# Optional imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Configured logger
    """
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    # Configure root logger
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    return logger


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed}")


def train_pipeline(cfg: DictConfig) -> Dict[str, Any]:
    """
    Complete training pipeline following Burkov's ML Engineering principles.

    Phases:
        0-1: Setup (logging, seeds)
        2: Data loading + Static feature engineering (BEFORE split - safe, deterministic)
        3: Statistical preprocessing (AFTER split - fit on train, transform all)
        4: Instantiate model,
        5: Instantiate dataloaders, metrics
        6: Instantiate engine then if torchengine: optimizer, scheduler
        7: Training
        8: Validation evaluation
        9: Test evaluation (final holdout - apply preprocessing now)
        10: Save results

    Feature Engineering Flow:
        - Static (StaticTimeSeriesPreprocessingPipeline): Applied in Phase 2 by DatasetFactory
          * Date features, lags, rolling windows, cyclical encoding
          * Deterministic and backward-looking (safe before split)
        - Statistical (StatisticalPreprocessingPipeline): Applied in Phase 3 after split
          * StandardScaler, RobustScaler, MinMaxScaler, etc.
          * Fit on train only, transform all splits (Burkov's principle)

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with training results
    """
    # ========================================================================
    # PHASE 0-1: SETUP
    # ========================================================================
    setup_logging(cfg)
    set_seed(cfg.get("seed", 42))

    logger.info("Phase 0-1: Setup complete")
    # ========================================================================
    # PHASE 2: DATA LOADING + STATIC FEATURE ENGINEERING
    # ========================================================================
    logger.info("Phase 2: Loading data and applying static feature engineering...")

    # DatasetFactory handles:
    # 1. Loading raw data
    # 2. Static feature engineering (BEFORE split - safe, deterministic)
    # 3. Temporal splitting (respecting time order)
    #
    # Static features (cfg.feature_engineering.static) are passed via
    # dataset_factory.static_feature_engineer = ${feature_engineering.static}
    factory = instantiate(cfg.dataset_factory)
    train_dataset, val_dataset, test_dataset = factory.create_datasets()

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Feature dimension: {train_dataset.get_feature_dim()}")

    # ========================================================================
    # PHASE 3: STATISTICAL PREPROCESSING (Fit on train, transform all)
    # ========================================================================
    # Statistical preprocessing happens AFTER splitting (fit on train only)
    # Static feature engineering already happened in Phase 2 (DatasetFactory)

    statistical_pipeline = None
    if "feature_engineering" in cfg and cfg.feature_engineering and cfg.feature_engineering.get("statistical"):
        logger.info("Phase 3: Applying statistical preprocessing...")

        # Instantiate statistical preprocessing pipeline from config
        statistical_pipeline = instantiate(cfg.feature_engineering.statistical)

        # Fit ONLY on training data (Burkov's principle)
        statistical_pipeline.fit(train_dataset)
        logger.info("Fitted statistical preprocessing pipeline on training data")

        # Transform train and validation splits
        train_dataset = statistical_pipeline.transform(train_dataset)
        val_dataset = statistical_pipeline.transform(val_dataset)
        # Test dataset NOT transformed yet - keep isolated until Phase 8!

        logger.info("Transformed train and validation datasets")
    else:
        logger.info("Phase 3: No statistical preprocessing configured")
        statistical_pipeline = None

    # ========================================================================
    # PHASE 4: INSTANTIATE MODEL
    # ========================================================================
    logger.info("Phase 4: Instantiating model...")
    logger.info(f"Model configuration:\n{OmegaConf.to_yaml(cfg)}")
    model = instantiate(cfg.model)
    logger.info(f"Model: {type(model).__name__}")
    # ========================================================================
    # PHASE 5: INSTANTIATE DATALOADERS + INSTANTIATE METRICS
    # ========================================================================
    logger.info("Phase 5: Creating dataloaders and instantiating metrics...")

    train_loader = instantiate(cfg.dataloader, dataset=train_dataset)
    val_loader = instantiate(cfg.dataloader, dataset=val_dataset)
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset)

    logger.info(f"Train loader: {type(train_loader).__name__}")
    logger.info(f"Val loader: {type(val_loader).__name__}")

    # Instantiate metrics
    metrics_dict = {}
    if "metrics" in cfg and "metrics" in cfg.metrics:
        logger.info("Instantiating metrics...")
        for metric_cfg in cfg.metrics.metrics:
            metric_fn = instantiate(metric_cfg)
            # Extract metric name from target path
            metric_name = metric_cfg._target_.split(".")[-1]
            metrics_dict[metric_name] = metric_fn
            logger.info(f"  - {metric_name}")
    else:
        logger.warning("No metrics configured, using defaults in engine")

    # ========================================================================
    # PHASE 6: INSTANTIATE ENGINE (+ OPTIMIZER/SCHEDULER FOR PYTORCH)
    # ========================================================================
    logger.info("Phase 6: Instantiating training engine...")
    logger.info("(Model-centric config: engine specified in model defaults)")

    # Setup MLflow tracking (optional)
    mlflow_tracker = None
    if MLFLOW_AVAILABLE and cfg.get("use_mlflow", False):
        mlflow.set_experiment(cfg.get("experiment_name", "forecasting"))
        mlflow.start_run(run_name=cfg.get("run_name", None))
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Log additional metadata as tags for MLflow UI
        # Extract model name from config
        model_name = "Unknown"
        if hasattr(cfg, "model") and hasattr(cfg.model, "_target_"):
            model_target = cfg.model._target_
            model_name = model_target.split(".")[-1].replace("Forecaster", "").replace("Regressor", "")

        # Extract dataset name
        dataset_name = cfg.get("dataset_name", None)
        if not dataset_name and hasattr(cfg, "dataset_factory"):
            # Extract from data path
            data_path = cfg.dataset_factory.get("data_path", "")
            if data_path:
                dataset_name = Path(data_path).stem  # e.g., 'Walmart' from 'Walmart.csv'

        # Set description
        description = cfg.get("description", f"{model_name} forecasting on {dataset_name or 'time series'} data")

        # Detect if running under Optuna and get trial info
        trial_number = None
        if OPTUNA_AVAILABLE and cfg.get("use_optuna", False):
            try:
                from hydra.core.global_hydra import GlobalHydra

                hydra_cfg = GlobalHydra.instance().hydra
                if hydra_cfg and hasattr(hydra_cfg, "sweeper") and hasattr(hydra_cfg.sweeper, "trial"):
                    trial = hydra_cfg.sweeper.trial
                    trial_number = trial.number
            except (AttributeError, ImportError):
                pass

        # Build run name with trial number if available
        run_name_base = f"{model_name}_{dataset_name or 'run'}"
        if trial_number is not None:
            run_name_base += f"_trial_{trial_number}"

        # Log as MLflow tags (visible in UI)
        mlflow.set_tag("mlflow.runName", run_name_base)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", dataset_name or "unknown")
        mlflow.set_tag("description", description)
        mlflow.set_tag("mlflow.note.content", description)  # Shows in Description column

        # Add Optuna-specific tags if in optimization
        if trial_number is not None:
            mlflow.set_tag("optuna_trial", trial_number)
            mlflow.set_tag("optuna_optimization", "true")

        mlflow_tracker = mlflow
        logger.info("MLflow tracking enabled")

    # Create checkpoint directory
    checkpoint_dir = None
    if cfg.get("save_checkpoints", True):
        run_name = cfg.get("run_name") or "default"  # Handle None case
        checkpoint_dir = Path(get_original_cwd()) / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Instantiate optimizer and scheduler (for PyTorch models)
    optimizer = None
    scheduler = None
    if "optimizer" in cfg and cfg.optimizer is not None:
        logger.info("Instantiating optimizer...")
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        logger.info(f"Optimizer: {type(optimizer).__name__}")

        if "scheduler" in cfg and cfg.scheduler is not None:
            logger.info("Instantiating scheduler...")
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            logger.info(f"Scheduler: {type(scheduler).__name__}")

    # Create training engine
    engine_kwargs = {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "mlflow_tracker": mlflow_tracker,
        "checkpoint_dir": checkpoint_dir,
        "verbose": cfg.get("verbose", True),
        "metrics": metrics_dict,  # Pass metrics dictionary
        "preprocessing_pipeline": statistical_pipeline,  # Pass pipeline for inverse transform
    }

    # Add optimizer and scheduler for PyTorch engines
    if optimizer is not None:
        engine_kwargs["optimizer"] = optimizer
    if scheduler is not None:
        engine_kwargs["scheduler"] = scheduler

    # Get engine config (check root level first, then model level)
    if "engine" in cfg and cfg.engine is not None:
        engine_cfg = cfg.engine
    elif "engine" in cfg.model and cfg.model.engine is not None:
        engine_cfg = cfg.model.engine
    else:
        raise ValueError("No engine configured! Add engine to config defaults")

    engine = instantiate(engine_cfg, **engine_kwargs)
    logger.info(f"Engine: {type(engine).__name__}")

    # ========================================================================
    # ========================================================================
    # PHASE 7: TRAINING
    # ========================================================================
    logger.info("Phase 7: Training...")

    training_results = engine.train()

    logger.info("Training complete!")
    logger.info(f"Training time: {training_results.get('training_time', 0):.2f}s")

    # ========================================================================
    # PHASE 8: VALIDATION EVALUATION
    # ========================================================================
    logger.info("Phase 8: Validation evaluation...")

    val_metrics = training_results.get("val_metrics", {})
    if val_metrics:
        for metric_name, value in val_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    # Report to Optuna for pruning (if running under Optuna)
    if OPTUNA_AVAILABLE and hasattr(cfg, "hydra") and cfg.get("use_optuna", False):
        try:
            # Get primary metric from validation set for reporting
            primary_metric = cfg.get("primary_metric", "MAPE")
            metric_value = (
                val_metrics.get(primary_metric)
                or val_metrics.get(f"{primary_metric}Metric")
                or val_metrics.get(primary_metric.upper())
                or val_metrics.get(primary_metric.lower())
                or float("inf")
            )

            # Try to get trial from Hydra Optuna sweeper context
            from hydra.core.global_hydra import GlobalHydra

            hydra_cfg = GlobalHydra.instance().hydra

            if hydra_cfg and hasattr(hydra_cfg, "sweeper") and hasattr(hydra_cfg.sweeper, "trial"):
                trial = hydra_cfg.sweeper.trial
                trial.report(metric_value, step=0)

                # Check if trial should be pruned
                if trial.should_prune():
                    logger.info("Trial pruned by Optuna - stopping early")
                    if mlflow_tracker:
                        mlflow.set_tag("pruned", "true")
                        mlflow.end_run()
                    raise optuna.TrialPruned()
        except (AttributeError, ImportError):
            # Not running under Optuna sweeper, continue normally
            pass
        except optuna.TrialPruned:
            # Re-raise pruning signal
            raise

    # ========================================================================
    # PHASE 9: TEST EVALUATION (Final, untouched until now!)
    # ========================================================================
    logger.info("Phase 9: TEST EVALUATION (final holdout set)")

    # NOW we can preprocess test set (using fitted statistical pipeline)
    if statistical_pipeline is not None:
        test_dataset = statistical_pipeline.transform(test_dataset)
        logger.info("Applied statistical preprocessing to test set")

    # Recreate test loader with preprocessed data
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset)

    # Update engine's test loader with preprocessed version
    engine.test_loader = test_loader

    # Evaluate on test set
    test_metrics = engine.test()

    for metric_name, value in test_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    # Log to MLflow
    if mlflow_tracker:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # ========================================================================
    # PHASE 10: SAVE RESULTS
    # ========================================================================
    logger.info("Phase 10: Saving results...")

    # Save final model
    if checkpoint_dir:
        final_model_path = checkpoint_dir / "final_model.pkl"
        engine.save_checkpoint(final_model_path, is_best=True)
        logger.info(f"Saved final model to: {final_model_path}")

        # Log model to MLflow (populates "Models" column)
        if mlflow_tracker:
            try:
                # Extract model name for registration
                model_name = "Unknown"
                if hasattr(cfg, "model") and hasattr(cfg.model, "_target_"):
                    model_target = cfg.model._target_
                    model_name = model_target.split(".")[-1].replace("Forecaster", "").replace("Regressor", "")

                # Log the model artifact
                mlflow.log_artifact(str(final_model_path), artifact_path="model")

                # Register model (optional - creates versioned model in Model Registry)
                # Uncomment to enable model registry:
                # mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

                logger.info("Logged model artifact to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")

    # Compile results
    results = {
        "train_metrics": training_results.get("train_metrics", {}),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time": training_results.get("training_time", 0),
        "history": training_results.get("history", {}),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    # Close MLflow run
    if mlflow_tracker:
        mlflow.end_run()
        logger.info("MLflow run closed")

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 80)

    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training entry point with Hydra configuration.

    Returns the primary metric from VALIDATION set for Optuna hyperparameter optimization.
    Always uses validation set (never test set) to prevent overfitting.
    The specific metric (MAPE, RMSE, MAE, etc.) is configurable via primary_metric.
    Defaults to MAPE if not specified.

    Example usage:
        # Single run with default config
        python -m ml_portfolio.training.train

        # Single run with config overrides
        python -m ml_portfolio.training.train model=arima dataset=walmart

        # Multi-run experiments
        python -m ml_portfolio.training.train -m model=arima,prophet dataset=walmart

        # Optuna hyperparameter optimization
        python -m ml_portfolio.training.train --multirun hydra/sweeper=optuna

        # With hyperparameter overrides
        python -m ml_portfolio.training.train model=lstm model.hidden_size=128 training.epochs=50

        # Using config groups
        python -m ml_portfolio.training.train model=lstm dataset=walmart engine=pytorch dataloader=pytorch

    Args:
        cfg: Hydra configuration object
            primary_metric: Name of metric to optimize (default: "MAPE")
            minimize: Whether to minimize metric (default: True)

    Returns:
        Validation metric value for Optuna optimization (always from val set)
    """
    import gc

    try:
        results = train_pipeline(cfg)

        # Get primary metric from config (defaults to MAPE)
        # Note: Always uses VALIDATION set to prevent overfitting to test set
        primary_metric = cfg.get("primary_metric", "MAPE")

        # Always use validation metrics for optimization
        val_metrics = results.get("val_metrics", {})

        # Try different metric name variations (MAPE, MAPEMetric, mape)
        metric_value = (
            val_metrics.get(primary_metric)
            or val_metrics.get(f"{primary_metric}Metric")
            or val_metrics.get(primary_metric.upper())
            or val_metrics.get(primary_metric.lower())
            or float("inf")
        )

        # Handle minimize vs maximize
        minimize = cfg.get("minimize", True)
        if not minimize:
            metric_value = -metric_value  # Negate for maximization problems

        logger.info(f"Returning val_{primary_metric} for optimization: {metric_value:.4f}")
        return metric_value

    except optuna.TrialPruned if OPTUNA_AVAILABLE else Exception:
        # Re-raise pruned trials for Optuna to handle
        logger.info("Trial pruned by Optuna")
        raise
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        # Return infinity for failed trials (Optuna will skip them)
        return float("inf")
    finally:
        # Cleanup resources for Optuna trials
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
