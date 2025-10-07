"""
Temporal Convolutional Network (TCN) for time series forecasting.

TCN uses dilated causal convolutions for efficient long-sequence modeling.
Excellent for capturing temporal dependencies with parallel computation.
"""

from typing import List

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

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        verbose: bool = True,
        **kwargs,
    ) -> "TCNForecaster":
        """Fit the TCN model using dataloaders."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Iterate over training dataloader
            for batch_X, batch_y in train_loader:
                # Convert to tensors if needed
                if isinstance(batch_X, np.ndarray):
                    batch_X = torch.FloatTensor(batch_X).to(self.device)
                if isinstance(batch_y, np.ndarray):
                    batch_y = torch.FloatTensor(batch_y).to(self.device)

                # Ensure correct shapes
                if len(batch_X.shape) == 2:
                    batch_X = batch_X.unsqueeze(1)
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)

                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            # Validation
            if val_loader is not None and (epoch + 1) % 10 == 0:
                self.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        if isinstance(batch_X, np.ndarray):
                            batch_X = torch.FloatTensor(batch_X).to(self.device)
                        if isinstance(batch_y, np.ndarray):
                            batch_y = torch.FloatTensor(batch_y).to(self.device)

                        if len(batch_X.shape) == 2:
                            batch_X = batch_X.unsqueeze(1)
                        if len(batch_y.shape) == 1:
                            batch_y = batch_y.unsqueeze(1)

                        val_outputs = self.forward(batch_X)
                        val_loss += criterion(val_outputs, batch_y).item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                self.train()

                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the fitted model."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            predictions = outputs.cpu().numpy()

        if predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions

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
