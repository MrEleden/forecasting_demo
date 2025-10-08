"""
Temporal Convolutional Network (TCN) for time series forecasting.

TCN uses dilated causal convolutions for efficient long-sequence modeling.
Excellent for capturing temporal dependencies with parallel computation.
"""

from typing import Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..base import PyTorchForecaster


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with residual connection.

    Uses dilated causal convolutions to expand receptive field.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Store padding for chopping
        self.padding = padding

        # Two conv layers with same dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # Activations and regularization
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through temporal block."""
        # Main path
        out = self.conv1(x)
        # Chop padding to maintain causality and match input size
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        # Chop padding again
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network architecture.

    Stack of temporal blocks with increasing dilation.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Padding to maintain causality
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through TCN."""
        # x: (batch, features, time)
        out = self.network(x)
        # Padding is already handled in TemporalBlock, so just return
        return out


class TCNForecaster(PyTorchForecaster, nn.Module):
    """
    TCN-based time series forecaster.

    Temporal Convolutional Networks use dilated causal convolutions for long dependencies.
    Faster than RNNs with parallel computation and flexible receptive field.

    Args:
        input_size: Number of input features
        output_size: Number of output features (forecast horizon)
        num_channels: List of channel sizes for each TCN level
        kernel_size: Convolutional kernel size
        dropout: Dropout rate
        device: Device to use ('cpu', 'cuda', 'auto')
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = "auto",
        **kwargs,
    ):
        PyTorchForecaster.__init__(self, device=device, **kwargs)
        nn.Module.__init__(self)

        if num_channels is None:
            num_channels = [32, 64, 128]

        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Output layer
        self.fc = nn.Linear(num_channels[-1], output_size)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the TCN network."""
        # x: (batch, time, features)
        x = x.transpose(1, 2)  # -> (batch, features, time)

        # TCN
        out = self.tcn(x)  # -> (batch, channels, time)

        # Take last timestep and predict
        out = out[:, :, -1]  # -> (batch, channels)
        out = self.fc(out)  # -> (batch, output_size)

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the fitted model."""
        inputs = self._prepare_inputs(X)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            predictions = outputs.cpu().numpy()

        if predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions

    def _prepare_batch(self, X: Any, y: Any, training: bool = True):
        """Convert batch to device tensors with causal-friendly shapes."""

        inputs = self._prepare_inputs(X)
        targets = self._prepare_targets(y)
        return inputs, targets

    def _to_device_tensor(self, data: Any) -> torch.Tensor:
        """Convert arbitrary input to float tensor on device."""

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
        """Ensure inputs have shape (batch, time, features)."""

        tensor = self._to_device_tensor(X)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 3:
            tensor = tensor.contiguous()
        elif tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported input shape for TCN: {tensor.shape}")

        return tensor

    def _prepare_targets(self, y: Any) -> torch.Tensor:
        """Ensure targets have shape (batch, output_size)."""

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
            "output_size": self.output_size,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate,
            "device": self.device,
        }

    def set_params(self, **params) -> "TCNForecaster":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TCNForecaster(input_size={self.input_size}, "
            f"num_channels={self.num_channels}, kernel_size={self.kernel_size}, "
            f"output_size={self.output_size})"
        )
