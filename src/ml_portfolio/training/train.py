"""
Training script for Walmart Sales Forecasting with Hydra configuration.

This script orchestrates the complete training pipeline:
1. Load and instantiate datasets (train/val/test splits)
2. Instantiate model from config
3. Create training engine with DataLoader support
4. Train model with validation
5. Evaluate on test set
6. Log results and save model

Usage:
    # Run with default configuration
    poetry run python projects/retail_sales_walmart/scripts/train.py

    # Override config parameters
    poetry run python projects/retail_sales_walmart/scripts/train.py model=arima dataloader=pytorch

    # Multi-run experiment sweep
    poetry run python projects/retail_sales_walmart/scripts/train.py -m model=lstm,tcn dataloader=simple,pytorch
"""

import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_portfolio.training.engine import TrainingEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training function using Hydra configuration.

    Args:
        cfg: Hydra configuration object from conf/config.yaml

    Returns:
        Primary metric value on test set (for Optuna optimization)
    """
    logger.info("=" * 60)
    logger.info("Walmart Sales Forecasting - Training Pipeline")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        np.random.seed(cfg.seed)
        import random

        random.seed(cfg.seed)
        logger.info(f"Random seed set to {cfg.seed}")

    # ========================================================================
    # 1. Create datasets for train/val/test modes
    # ========================================================================
    logger.info(f"Creating datasets: {cfg.dataset._target_}")

    train_dataset = hydra.utils.instantiate(cfg.dataset, mode="train")
    train_dataset.load()
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    val_dataset = hydra.utils.instantiate(cfg.dataset, mode="val")
    val_dataset.load()
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    test_dataset = hydra.utils.instantiate(cfg.dataset, mode="test")
    test_dataset.load()
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Log dataset info
    if hasattr(train_dataset, "get_data_info"):
        info = train_dataset.get_data_info()
        logger.info(f"Dataset info: {info}")

    # ========================================================================
    # 2. Instantiate data loaders from configuration
    # ========================================================================
    # Get dataloader config (optional)
    dataloader_train = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
    dataloader_val = hydra.utils.instantiate(cfg.dataloader, dataset=val_dataset)
    dataloader_test = hydra.utils.instantiate(cfg.dataloader, dataset=test_dataset)
    # ========================================================================
    # 3. Instantiate model from configuration
    # ========================================================================
    logger.info(f"Instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)
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
        try:
            metrics[metric_name] = hydra.utils.instantiate(metric_cfg)
            logger.info(f"  - {metric_name}: {metric_cfg._target_}")
        except Exception as e:
            logger.warning(f"  - Failed to instantiate {metric_name}: {e}")

    # ========================================================================
    # 7. Create training engine with all components
    # ========================================================================
    logger.info("Creating TrainingEngine with datasets...")
    engine = TrainingEngine(
        model=model,
        metrics=metrics,
        train_dataset=dataloader_train,
        val_dataset=dataloader_val,
        test_dataset=dataloader_test,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg.training if hasattr(cfg, "training") else {},
    )

    # ========================================================================
    # 8. Train model using training engine and validation data for parameter tuning
    # ========================================================================
    logger.info("Starting training...")
    results = engine.train()

    # Log training results
    logger.info("Training completed!")
    logger.info(f"  Best epoch: {results['best_epoch']}")
    logger.info(f"  Total time: {results.get('total_training_time', 0):.2f}s")
    logger.info(f"  Converged: {results['converged']}")

    # ========================================================================
    # 9. Evaluate on test set (engine handles internally)
    # ========================================================================
    logger.info("Evaluating on test set...")
    test_metrics = engine.evaluate_test()

    # Log test metrics
    logger.info("Test set results:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
