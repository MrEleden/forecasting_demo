"""
ML Portfolio - Time Series Forecasting Library

A comprehensive library for time series forecasting with classical and deep learning models.
Designed for retail sales, rideshare demand, inventory, and economic indicator forecasting.
"""

__version__ = "0.1.0"
__author__ = "ML Portfolio"
__email__ = "contact@ml-portfolio.com"

# Core modules
from . import data
from . import models
from . import training
from . import evaluation
from . import pipelines
from . import utils

# Quick access imports
from .data import TimeSeriesDataset, TimeSeriesScaler
from .models import ModelRegistry
from .training import TrainingEngine
from .evaluation import TimeSeriesBacktester, plot_forecast
from .pipelines import create_time_series_pipeline, create_hybrid_pipeline
from .utils import ConfigManager, DataCache

__all__ = [
    # Modules
    "data",
    "models",
    "training",
    "evaluation",
    "pipelines",
    "utils",
    # Quick access
    "TimeSeriesDataset",
    "TimeSeriesScaler",
    "ModelRegistry",
    "TrainingEngine",
    "TimeSeriesBacktester",
    "plot_forecast",
    "create_time_series_pipeline",
    "create_hybrid_pipeline",
    "ConfigManager",
    "DataCache",
]
