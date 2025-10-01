"""
Models module for ML Portfolio

Contains forecasting models, loss functions, metrics, and model management utilities.
"""

from . import blocks, forecasting
from .losses import *
from .metrics import *
from .wrappers import *
from .registry import *

# Try to import statistical models (optional dependencies)
try:
    from .statistical import ARIMAWrapper, ProphetWrapper

    STATISTICAL_AVAILABLE = True
except ImportError:
    STATISTICAL_AVAILABLE = False
