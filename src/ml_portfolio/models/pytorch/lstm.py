"""
LSTM forecasting model for time series.

Clean implementation using PyTorchForecaster base class.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..base import PyTorchForecaster


class LSTMForecaster(PyTorchForecaster, nn.Module):
    """
    LSTM-based time series forecaster.

    Long Short-Term Memory network for sequential forecasting.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_size: Number of output features
        dropout: Dropout rate for regularization
        bidirectional: Use bidirectional LSTM
        device: Device to use ('cpu', 'cuda', 'auto')
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        device: str = "auto",
        **kwargs,
    ):
        PyTorchForecaster.__init__(self, device=device, **kwargs)
        nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM network."""
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the fitted model."""
        tensor = self._prepare_inputs(X)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(tensor)
            predictions = outputs.cpu().numpy()

        if predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions

    def _prepare_batch(self, X: Any, y: Any, training: bool = True):
        """Convert batches to device tensors with correct shapes."""

        inputs = self._prepare_inputs(X)
        targets = self._prepare_targets(y)
        return inputs, targets

    def _to_device_tensor(self, data: Any) -> torch.Tensor:
        """Convert arbitrary input to a float tensor on the configured device."""

        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=torch.float32)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            array = data.values
        elif isinstance(data, np.ndarray):
            array = data
        else:
            array = np.asarray(data)

        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _prepare_inputs(self, X: Any) -> torch.Tensor:
        """Ensure input batches are shaped (batch, seq, features)."""

        tensor = self._to_device_tensor(X)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 3:
            # Assume batch-first (batch, seq, features)
            tensor = tensor.contiguous()
        elif tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported input shape for LSTM: {tensor.shape}")

        return tensor

    def _prepare_targets(self, y: Any) -> torch.Tensor:
        """Ensure targets are shaped (batch, output_size)."""

        tensor = self._to_device_tensor(y)

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 0:
            tensor = tensor.view(1, 1)
        elif tensor.dim() > 2:
            tensor = tensor.view(tensor.shape[0], -1)

        return tensor

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R^2 score on test data."""
        from sklearn.metrics import r2_score

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if len(y.shape) > 1:
            y = y.ravel()

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "bidirectional": self.bidirectional,
            "device": self.device,
        }

    def set_params(self, **params) -> "LSTMForecaster":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LSTMForecaster(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
            f"output_size={self.output_size})"
        )
