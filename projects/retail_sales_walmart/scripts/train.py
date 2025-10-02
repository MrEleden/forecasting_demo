"""
Walmart Sales Forecasting Training Script.

This script loads Walmart-specific configuration and calls the generic training pipeline.

Usage:
    # Run with default configuration (Random Forest)
    python projects/retail_sales_walmart/scripts/train.py

    # Override with LSTM model
    python projects/retail_sales_walmart/scripts/train.py model=lstm dataloader=pytorch training.max_epochs=100

    # Multi-run experiment
    python projects/retail_sales_walmart/scripts/train.py -m model=random_forest,lstm,arima
"""

import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Import generic training pipeline from src
from ml_portfolio.training.train import train_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Walmart training entry point.

    Loads Walmart-specific configuration from ../conf/config.yaml,
    which imports base configs from src/ml_portfolio/conf/ and
    overrides with Walmart-specific settings.

    Args:
        cfg: Hydra configuration from projects/retail_sales_walmart/conf/

    Returns:
        Primary metric value on test set
    """
    # Call generic training pipeline with Walmart configuration
    return train_pipeline(cfg, project_name="Walmart Sales Forecasting")


if __name__ == "__main__":
    main()
