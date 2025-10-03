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

    # Create model (model config includes dataloader and engine via defaults)
    model = instantiate(cfg.model)
    logger.info(f"Model: {type(model).__name__}")

    # ========================================================================
    # PHASE 5: CREATE DATALOADERS + INSTANTIATE METRICS
    # ========================================================================
    logger.info("Phase 5: Creating dataloaders and instantiating metrics...")

    # Instantiate dataloader class based on config
    # Model-centric: model config uses "defaults: [/dataloader: pytorch]"
    # The "/" prefix merges dataloader config at root level (cfg.dataloader)
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
        mlflow_tracker = mlflow
        logger.info("MLflow tracking enabled")

    # Create checkpoint directory
    checkpoint_dir = None
    if cfg.get("save_checkpoints", True):
        checkpoint_dir = Path(get_original_cwd()) / "checkpoints" / cfg.get("run_name", "default")
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
    }

    # Add optimizer and scheduler for PyTorch engines
    if optimizer is not None:
        engine_kwargs["optimizer"] = optimizer
    if scheduler is not None:
        engine_kwargs["scheduler"] = scheduler

    engine = instantiate(cfg.engine, **engine_kwargs)
    logger.info(f"Engine: {type(engine).__name__}")

    # ========================================================================
    # PHASE 7: TRAINING
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Phase 7: Training...")
    logger.info("=" * 80)

    training_results = engine.train()

    logger.info("Training complete!")
    logger.info(f"Training time: {training_results.get('training_time', 0):.2f}s")

    # ========================================================================
    # PHASE 8: VALIDATION EVALUATION
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Phase 8: Validation evaluation...")
    logger.info("=" * 80)

    val_metrics = training_results.get("val_metrics", {})
    if val_metrics:
        for metric_name, value in val_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    # ========================================================================
    # PHASE 9: TEST EVALUATION (Final, untouched until now!)
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Phase 9: TEST EVALUATION (final holdout set)")
    logger.info("=" * 80)

    # NOW we can preprocess test set (using fitted statistical pipeline)
    if statistical_pipeline is not None:
        test_dataset = statistical_pipeline.transform(test_dataset)
        logger.info("Applied statistical preprocessing to test set")

    # Recreate test loader with preprocessed data
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset)

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
    logger.info("=" * 80)
    logger.info("Phase 10: Saving results...")
    logger.info("=" * 80)

    # Save final model
    if checkpoint_dir:
        final_model_path = checkpoint_dir / "final_model.pkl"
        engine.save_checkpoint(final_model_path, is_best=True)
        logger.info(f"Saved final model to: {final_model_path}")

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

    # Return primary metric for hyperparameter optimization
    primary_metric = cfg.get("primary_metric", "val_mape")
    metric_value = val_metrics.get(primary_metric.replace("val_", ""), float("inf"))

    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training entry point with Hydra configuration.

    The function returns the primary validation metric for hyperparameter optimization.

    Example usage:
        # Single run with default config
        python -m ml_portfolio.training.train

        # Single run with config overrides
        python -m ml_portfolio.training.train model=arima dataset=walmart

        # Multi-run experiments
        python -m ml_portfolio.training.train -m model=arima,prophet dataset=walmart

        # With hyperparameter overrides
        python -m ml_portfolio.training.train model=lstm model.hidden_size=128 training.epochs=50

        # Using config groups
        python -m ml_portfolio.training.train model=lstm dataset=walmart engine=pytorch dataloader=pytorch

    Args:
        cfg: Hydra configuration object

    Returns:
        Primary validation metric value (for Optuna optimization)
    """
    try:
        results = train_pipeline(cfg)

        # Return primary metric for optimization
        primary_metric = cfg.get("primary_metric", "val_mape")
        val_metrics = results.get("val_metrics", {})
        metric_value = val_metrics.get(primary_metric.replace("val_", ""), float("inf"))

        logger.info(f"Returning {primary_metric}: {metric_value}")
        return metric_value

    except Exception as e:
        logger.exception(f"Training pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
