"""
Base model classes for the ML Portfolio forecasting system.

This module defines the minimal abstract base classes that all forecasting models should inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


class BaseForecaster(ABC):
    """
    Base class for all forecasting models.

    Models own their training logic in fit():
    - Statistical models: single pass over data
    - PyTorch models: multiple epochs with backprop
    """

    def __init__(self, **kwargs):
        """Initialize base forecaster."""
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_loader, val_loader=None, **kwargs):
        """
        Train the model using dataloaders.

        Model iterates over loaders internally and controls training logic.

        Args:
            train_loader: DataLoader for training data (yields batches of X, y)
            val_loader: Optional DataLoader for validation data
            **kwargs: Model-specific training parameters (epochs, learning_rate, etc.)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Input features (numpy array)

        Returns:
            Predictions (numpy array)
        """
        pass

    def save(self, path: Path):
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: Path):
        """
        Load model from file.

        Args:
            path: Path to load model from
        """
        import pickle

        with open(path, "rb") as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)


class StatisticalForecaster(BaseForecaster):
    """
    Base class for statistical forecasting models (sklearn-like interface).

    These models typically fit in a single pass over the data.
    """

    def __init__(self, **kwargs):
        """Initialize statistical forecaster."""
        super().__init__(**kwargs)

    def fit(self, train_loader, val_loader=None, **kwargs):
        """
        Fit statistical model with single pass over data.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            **kwargs: Additional fitting parameters
        """
        # Get data from loader (single iteration for statistical models)
        for X_train, y_train in train_loader:
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, (pd.DataFrame, pd.Series)):
                y_train = y_train.values

            # Call internal fit method
            self._fit(X_train, y_train)
            self.is_fitted = True
            break  # Statistical models only need one pass

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Internal fit method (model-specific implementation).

        Args:
            X: Training features
            y: Training targets
        """
        pass

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        from ml_portfolio.evaluation.metrics import mae, mape, rmse

        # Flatten arrays
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        return {
            "MAPE": float(mape(y_true, y_pred)),
            "RMSE": float(rmse(y_true, y_pred)),
            "MAE": float(mae(y_true, y_pred)),
        }


class PyTorchForecaster(BaseForecaster):
    """
    Base class for PyTorch-based forecasting models.

    This class provides PyTorch training logic with multi-epoch support.
    Subclasses must inherit from both PyTorchForecaster and nn.Module.
    """

    def __init__(self, device: str = "auto", **kwargs):
        """
        Initialize PyTorch forecaster.

        Args:
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        super().__init__(**kwargs)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchForecaster. Install with: pip install torch")

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Fit PyTorch model with multi-epoch training.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            verbose: Whether to print training progress
            **kwargs: Additional training parameters
        """
        # This method should be overridden by subclasses
        # but provides a template for PyTorch training
        raise NotImplementedError("Subclasses must implement fit() method")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the PyTorch model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions in evaluation mode.

        Args:
            X: Input features (numpy array)

        Returns:
            Predictions (numpy array)
        """
        # This should be overridden by subclasses with proper shape handling
        raise NotImplementedError("Subclasses must implement predict() method")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        from ml_portfolio.evaluation.metrics import mae, mape, rmse

        # Flatten arrays
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        return {
            "MAPE": float(mape(y_true, y_pred)),
            "RMSE": float(rmse(y_true, y_pred)),
            "MAE": float(mae(y_true, y_pred)),
        }

    def save(self, path: Path):
        """
        Save PyTorch model state dict.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save state dict and model config
        save_dict = {
            "state_dict": self.state_dict(),
            "config": self.get_params() if hasattr(self, "get_params") else {},
            "device": self.device,
        }
        torch.save(save_dict, path)

    def load(self, path: Path):
        """
        Load PyTorch model state dict.

        Args:
            path: Path to load model from
        """
        save_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(save_dict["state_dict"])
        if "device" in save_dict:
            self.device = save_dict["device"]
            self.to(self.device)
