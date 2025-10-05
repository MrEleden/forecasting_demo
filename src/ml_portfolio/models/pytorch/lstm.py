"""
LSTM forecasting model for time series.

Clean implementation using PyTorchForecaster base class.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import PyTorchForecaster


class LSTMForecaster(PyTorchForecaster):
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
        super().__init__(device=device, **kwargs)

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True,
        **kwargs,
    ) -> "LSTMForecaster":
        """Fit the LSTM model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device) if n_val > 0 else None
        y_val_tensor = torch.FloatTensor(y_val).to(self.device) if n_val > 0 else None

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            if X_val_tensor is not None and (epoch + 1) % 10 == 0:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                self.train()

                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
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
