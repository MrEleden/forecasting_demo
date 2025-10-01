"""
LSTM and Seq2Seq implementations for time series forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


class LSTMCore(nn.Module):
    """Core LSTM network."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class LSTMForecaster:
    """
    LSTM forecaster with unified .fit() and .predict() interface.

    This wrapper makes PyTorch models compatible with the unified training system.
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=4,
        dropout=0.2,
        lr=0.001,
        epochs=100,
        batch_size=32,
        device=None,
    ):
        """
        Initialize LSTM forecaster.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            output_size: Number of output features (forecast horizon)
            dropout: Dropout rate
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run on (auto-detected if None)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Auto-detect device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = LSTMCore(input_size, hidden_size, num_layers, output_size, dropout)
        self.model.to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _prepare_sequences(self, X, y=None):
        """
        Convert flattened features back to sequences for LSTM.

        Args:
            X: Flattened features (batch_size, sequence_length)
            y: Targets (batch_size, forecast_horizon) - optional

        Returns:
            X_sequences: (batch_size, sequence_length, 1)
            y_sequences: (batch_size, forecast_horizon) - if y provided
        """
        # Reshape X back to sequences
        X_sequences = X.reshape(X.shape[0], -1, self.input_size)

        if y is not None:
            return X_sequences, y
        return X_sequences

    def fit(self, X, y):
        """
        Fit the LSTM model.

        Args:
            X: Training features (batch_size, sequence_length)
            y: Training targets (batch_size, forecast_horizon)
        """
        print(f"Training LSTM on device: {self.device}")

        # Prepare data
        X_seq, y_targets = self._prepare_sequences(X, y)

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_seq.reshape(-1, self.input_size)).reshape(X_seq.shape)
        y_scaled = self.scaler_y.fit_transform(y_targets.reshape(-1, 1)).reshape(y_targets.shape)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:3d}/{self.epochs}: Loss = {loss.item():.6f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions with the LSTM model.

        Args:
            X: Features (batch_size, sequence_length)

        Returns:
            predictions: (batch_size, forecast_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare data
        X_seq = self._prepare_sequences(X)

        # Scale data
        X_scaled = self.scaler_X.transform(X_seq.reshape(-1, self.input_size)).reshape(X_seq.shape)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

        # Convert back to numpy and inverse transform
        predictions_scaled = outputs.cpu().numpy()
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(
            predictions_scaled.shape
        )

        return predictions

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
