"""
Generic Training Script with Hydra Configuration.

This script provides a reusable training pipeline that can be called
from project-specific scripts with different configurations.

DO NOT call this directly. Use project-specific train.py scripts instead.

Example:
    # From projects/retail_sales_walmart/scripts/train.py
    from ml_portfolio.training.train import train_pipeline
    train_pipeline(cfg)
"""

import logging
import random
from typing import Dict, Any
from pathlib import Path
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from ml_portfolio.training.engine import TrainingEngine

# Setup logging
logger = logging.getLogger(__name__)


def train_pipeline(cfg: DictConfig, project_name: str = "Training") -> float:
    """
    Generic training pipeline that can be called from any project.

    Args:
        cfg: Hydra configuration object
        project_name: Name of the project for logging

    Returns:
        Primary metric value on test set (for Optuna optimization)
    """
    logger.info("=" * 60)
    logger.info(f"{project_name} - Training Pipeline")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        logger.info(f"Random seed set to {cfg.seed}")

    # ========================================================================
    # 0. load default config and overide with project-specific config
    # ========================================================================
    # Store the original project config
    project_cfg = cfg

    # Load and resolve base configuration manually
    base_config_dir = Path(__file__).parent.parent / "conf"
    base_config_path = base_config_dir / "config.yaml"
    base_cfg = OmegaConf.load(base_config_path)

    # Manually resolve the defaults by loading each component config
    if "defaults" in base_cfg:
        defaults = base_cfg.defaults
        for default in defaults:
            if isinstance(default, str):
                continue  # Skip _self_ and other special defaults
            if isinstance(default, (dict, DictConfig)):
                for key, value in default.items():
                    if value and value != "null" and value is not None:  # Skip null values
                        component_path = base_config_dir / key / f"{value}.yaml"
                        if component_path.exists():
                            component_cfg = OmegaConf.load(component_path)
                            base_cfg[key] = component_cfg
                            logger.info(f"Loaded {key}: {value}")

    # Merge base config with project-specific config
    cfg = OmegaConf.merge(base_cfg, project_cfg)
    logger.info("Configuration successfully merged with base defaults")

    # ========================================================================
    # 1. Create datasets using factory pattern
    # ========================================================================
    logger.info(f"Creating dataset factory: {cfg.dataset._target_}")

    # Create dataset factory and get the three dataset splits
    dataset_factory = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset, test_dataset = dataset_factory.create_datasets()

    logger.info(f"Created datasets - train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # Log dataset info from the train dataset (all should have same metadata)
    if hasattr(train_dataset, "get_data_info"):
        info = train_dataset.get_data_info()
        logger.info(f"Dataset info: {info}")

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
        # Skip simple config values that aren't metric objects
        if metric_name in ["primary_metric", "minimize"]:
            continue

        try:
            metrics[metric_name] = hydra.utils.instantiate(metric_cfg)
            logger.info(f"  - {metric_name}: {metric_cfg._target_}")
        except Exception as e:
            logger.warning(f"  - Failed to instantiate {metric_name}: {e}")

    # ========================================================================
    # 7. Extract training parameters and create training engine
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
        verbose=verbose,
        monitor_metric=monitor_metric,
        min_delta=min_delta,
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

    # Return primary metric for optimization
    return test_metrics.get(cfg.get("primary_metric", "mse"), 0.0)
