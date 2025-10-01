"""
Models module for ML Portfolio

Contains forecasting models, loss functions, metrics, and model management utilities.
"""

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

# Try to import deep learning models (optional PyTorch dependency)
try:
    from .lstm import LSTMForecaster
    from .transformer import TransformerForecaster
    from .tcn import TCNForecaster
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
