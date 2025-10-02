#!/usr/bin/env python3
"""
Train individual M5 competition time series forecasting models.

This script demonstrates how to train individual M5-inspired models
using the centralized training framework.

Usage examples:
    # Train seasonal naive model
    python scripts/train_m5_individual.py model=seasonal_naive_standalone

    # Train ETS model
    python scripts/train_m5_individual.py model=ets_standalone

    # Train Fourier+ARIMA model
    python scripts/train_m5_individual.py model=fourier_arima_standalone

    # Train all models in multi-run (line broken for length)
    python scripts/train_m5_individual.py -m model=seasonal_naive_standalone,ets_standalone,\\
        fourier_arima_standalone,stlf_arima_standalone,svd_ets_standalone
"""

import sys
from pathlib import Path

# Add src and project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from ml_portfolio.training.train import main  # noqa: E402

if __name__ == "__main__":
    # Set default config path to point to our M5 configurations

    # Override default config directory to include M5 model configs
    config_dir = str(PROJECT_ROOT / "src" / "ml_portfolio" / "conf")

    print("🏪 Walmart M5 Individual Model Training")
    print("=" * 50)
    print(f"📁 Config directory: {config_dir}")
    print("📊 Available models:")
    print("   - seasonal_naive_standalone")
    print("   - ets_standalone")
    print("   - svd_ets_standalone")
    print("   - svd_stlf_standalone")
    print("   - fourier_arima_standalone")
    print("   - stlf_arima_standalone")
    print()

    # Call main training function
    main()
