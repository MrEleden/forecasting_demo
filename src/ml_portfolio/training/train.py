"""
Main Training Script with Hydra Configuration.

This is the main entry point for training ML forecasting models.
All configurations live in src/ml_portfolio/conf/

Usage:
    # Run with default configuration
    python src/ml_portfolio/training/train.py

    # Override specific parameters
    python src/ml_portfolio/training/train.py dataset_factory=walmart model=arima

    # Multi-run experiments
    python src/ml_portfolio/training/train.py -m model=random_forest,lstm,arima dataset_factory=walmart

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
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ml_portfolio.training.engine import TrainingEngine

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
    Main training pipeline with Hydra configuration.

    Args:
        cfg: Hydra configuration object (automatically loaded from conf/)
        project_name: Name of the project for logging

    Returns:
        Primary metric value on test set (for Optuna optimization)
    """
    logger.info("=" * 60)
    logger.info(f"{project_name}")
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
    # 1. Create datasets using factory pattern
    # ========================================================================
    logger.info(f"Creating dataset factory: {cfg.dataset_factory._target_}")

    # Create dataset factory and get the three dataset splits
    dataset_factory = hydra.utils.instantiate(cfg.dataset_factory)
    train_dataset, val_dataset, test_dataset = dataset_factory.create_datasets()

    logger.info(f"Created datasets - train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # Log dataset info from the train dataset (all should have same metadata)
    n_features = None
    if hasattr(train_dataset, "get_data_info"):
        info = train_dataset.get_data_info()
        logger.info(f"Dataset info: {info}")
        n_features = info.get("n_features")

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
    engine.train()
    # Training completion is logged by engine

    # ========================================================================
    # 9. Evaluate on test set (engine handles internally)
    # ========================================================================
    test_metrics = engine.evaluate_test()
    # Test evaluation is logged by engine

    # Return primary metric for optimization
    primary_metric = cfg.metrics.get("primary_metric", "mse")
    return test_metrics.get(primary_metric, 0.0)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main entry point for Hydra training.

    This function is called when running:
        python -m ml_portfolio.training.train

    Args:
        cfg: Hydra configuration from src/ml_portfolio/conf/

    Returns:
        Primary metric value for optimization
    """
    return train_pipeline(cfg)


if __name__ == "__main__":
    main()
