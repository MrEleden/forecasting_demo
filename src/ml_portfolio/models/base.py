"""
Base model classes for the ML Portfolio forecasting system.

This module defines the minimal abstract base classes that all forecasting models should inherit from.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

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
        learning_rate: float = None,
        verbose: bool = True,
        grad_clip: float = None,
        val_interval: int = 1,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
        **kwargs,
    ):
        """
        Fit PyTorch model with multi-epoch training.

        Subclasses typically only need to implement ``forward`` and optionally override
        helper hooks (``_prepare_batch``, ``_training_step``, ``_validation_step``,
        ``_get_loss_fn``, ``_get_optimizer``).

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            verbose: Whether to print training progress
            grad_clip: Gradient clipping value (L2 norm). Disabled when ``None``.
            val_interval: Validate every N epochs (defaults to every epoch)
            **kwargs: Additional training parameters for subclasses
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchForecaster. Install with: pip install torch")

        if train_loader is None:
            raise ValueError("train_loader must be provided for PyTorch models")

        if learning_rate is None:
            learning_rate = getattr(self, "learning_rate", 0.001)

        if grad_clip is None:
            grad_clip = getattr(self, "default_grad_clip", None)

        monitor_mode = monitor_mode.lower()
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be either 'min' or 'max'")

        val_interval = max(val_interval, 1)

        supported_monitors = {"val_loss", "train_loss"}
        if early_stopping and monitor_metric not in supported_monitors:
            if verbose:
                print(f"Early stopping monitor '{monitor_metric}' is not supported; " "disabling early stopping.")
            early_stopping = False

        if early_stopping:
            patience = max(1, patience)

        if early_stopping and monitor_metric.startswith("val") and val_loader is None:
            if verbose:
                print("Early stopping requested but no validation loader supplied; disabling early stopping.")
            early_stopping = False

        self.train()

        criterion = self._get_loss_fn()
        optimizer = self._get_optimizer(learning_rate)
        scheduler = self._get_scheduler(optimizer)

        log_interval = kwargs.get("log_interval", 10)

        best_metric = None
        best_state_dict = None
        best_epoch = -1
        epochs_without_improvement = 0
        total_epochs = epochs

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                inputs, targets = self._prepare_batch(batch_X, batch_y, training=True)

                optimizer.zero_grad()
                loss, _ = self._training_step(inputs, targets, criterion)

                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            if scheduler is not None:
                scheduler.step()

            if verbose and (epoch + 1) % max(log_interval, 1) == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

            monitored_metric = None

            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_loss = self._run_validation(val_loader, criterion)
                if verbose:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, " f"Val Loss: {val_loss:.4f}"
                    )

                if monitor_metric == "val_loss":
                    monitored_metric = val_loss

            if monitored_metric is None and monitor_metric == "train_loss":
                monitored_metric = avg_train_loss

            if early_stopping and monitored_metric is not None:
                is_better = False
                if best_metric is None:
                    is_better = True
                elif monitor_mode == "min":
                    is_better = monitored_metric < (best_metric - min_delta)
                else:
                    is_better = monitored_metric > (best_metric + min_delta)

                if is_better:
                    best_metric = monitored_metric
                    best_state_dict = deepcopy(self.state_dict())
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if verbose:
                            target = f"{monitor_metric}"
                            print(
                                "Early stopping triggered at epoch "
                                f"{epoch + 1} (no improvement in {target} for {patience} checks)."
                            )
                        total_epochs = epoch + 1
                        break

        if early_stopping and best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        self.is_fitted = True
        self.trained_epochs = total_epochs
        self.best_epoch = best_epoch if best_epoch >= 0 else total_epochs - 1
        self.best_metric = best_metric
        return self

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

    def _get_loss_fn(self) -> Any:
        """Return loss function used during training."""

        return nn.MSELoss()

    def _get_optimizer(self, learning_rate: float) -> Any:
        """Instantiate optimizer for training."""

        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _get_scheduler(self, optimizer: Any):
        """Optional learning-rate scheduler hook (default: None)."""

        return None

    def _prepare_batch(self, X: Any, y: Any, training: bool = True) -> Tuple[Any, Any]:
        """Move batch to device and ensure tensor format."""

        return self._to_tensor(X), self._to_tensor(y)

    def _training_step(
        self,
        inputs: Any,
        targets: Any,
        criterion: Any,
    ) -> Tuple[Any, Any]:
        """Perform forward pass and compute loss for a single batch."""

        outputs = self.forward(inputs)
        loss = criterion(outputs, targets)
        return loss, outputs

    def _validation_step(
        self,
        inputs: Any,
        targets: Any,
        criterion: Any,
    ) -> Any:
        """Compute validation loss for a single batch."""

        outputs = self.forward(inputs)
        loss = criterion(outputs, targets)
        return loss

    def _run_validation(self, loader, criterion: Any) -> float:
        """Evaluate validation loss over an entire loader."""

        self.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                inputs, targets = self._prepare_batch(batch_X, batch_y, training=False)
                loss = self._validation_step(inputs, targets, criterion)
                total_loss += loss.item()
                n_batches += 1

        self.train()
        return total_loss / max(n_batches, 1)

    def _to_tensor(self, data: Any) -> Any:
        """Convert numpy/pandas/array-like data to float tensor on device."""

        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=torch.float32)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            array = data.values
        elif isinstance(data, np.ndarray):
            array = data
        else:
            array = np.asarray(data)

        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

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
