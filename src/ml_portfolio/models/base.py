"""
Base model classes for the ML Portfolio forecasting system.

This module defines the minimal abstract base classes that all forecasting models should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

try:
    import torch.nn as nn
except ImportError:
    nn = None


class BaseForecaster(ABC):
    """
    Minimal abstract base class for all forecasting models.

    This class defines only the essential interface that all forecasting models must implement.
    """

    def __init__(self, **kwargs):
        """
        Initialize base forecaster with data loading and training strategies.

        Args:
            dataloader_class: Class responsible for creating data loaders/handling data
            training_class: Class responsible for training logic
            **kwargs: Additional parameters for subclasses
        """

        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "BaseForecaster":
        """
        Fit the forecasting model.

        Args:
            X: Input features
            y: Target values

        Returns:
            self: Fitted estimator
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass


class PyTorchForecaster(BaseForecaster, nn.Module):
    """
    Base class for PyTorch-based forecasting models.

    This class inherits from both BaseForecaster and nn.Module to provide
    PyTorch functionality while maintaining the forecasting interface.
    """

    def __init__(self, device: str = "auto", **kwargs):
        """
        Initialize PyTorch forecaster.

        Args:
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        # Initialize both parent classes
        BaseForecaster.__init__(self)
        nn.Module.__init__(self)

        if device == "auto":
            self.device = "cuda" if self._is_cuda_available() else "cpu"
        else:
            self.device = device

    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @abstractmethod
    def forward(self, x):
        """
        Define the forward pass of the PyTorch model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "BaseForecaster":
        """
        Fit the PyTorch model by calling forward during training.

        This method should be overridden by subclasses to implement specific training logic,
        but it ensures that forward() is called during the training process.

        Args:
            X: Input features
            y: Target values

        Returns:
            self: Fitted estimator
        """
        # Convert inputs to tensors
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X_tensor = torch.FloatTensor(X).to(self.device)
        _ = torch.FloatTensor(y).to(self.device)  # y_tensor for future use

        # Call forward pass (this is where subclasses will implement their training logic)
        _ = self.forward(X_tensor)  # outputs for future use

        # This is a basic implementation - subclasses should override this method
        # to implement proper training loops, loss calculation, backpropagation, etc.
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions by calling forward in evaluation mode.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        import torch

        self.eval()  # Set to evaluation mode

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X_tensor)

        return outputs.cpu().numpy()


class StatisticalForecaster(BaseForecaster):
    """
    Base class for statistical forecasting models.
    """

    def __init__(self, **kwargs):
        """Initialize statistical forecaster."""
        pass
