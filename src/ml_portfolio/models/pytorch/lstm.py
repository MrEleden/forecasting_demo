"""
LSTM implementation for time series forecasting.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..base import PyTorchForecaster


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


class LSTMForecaster(PyTorchForecaster):
    """LSTM forecaster inheriting from PyTorchForecaster base class."""

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        lr=0.001,
        epochs=100,
        batch_size=32,
        device="auto",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.scaler_ = StandardScaler()
        self.is_fitted_ = False

    def _validate_input(self, X, y=None):
        """Validate and convert input data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            if isinstance(y, list):
                y = np.array(y)
            return X.astype(np.float32), y.astype(np.float32)

        return X.astype(np.float32)

    def _prepare_data(self, X, y=None):
        """Convert numpy arrays to PyTorch tensors."""
        X_tensor = torch.FloatTensor(X).to(self.device)

        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)
            return X_tensor, y_tensor

        return X_tensor

    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before making predictions")

    def _create_model(self):
        """Create the LSTM model."""
        model = LSTMCore(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def fit(self, X, y):
        """Fit the LSTM model."""
        X, y = self._validate_input(X, y)

        if len(X.shape) == 3:
            self.input_size = X.shape[2]
        else:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            self.input_size = X.shape[2]

        X_scaled = self.scaler_.fit_transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)

        self.model_ = self._create_model()
        self.criterion_ = nn.MSELoss()
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        X_tensor, y_tensor = self._prepare_data(X_scaled, y)

        self.model_.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                batch_y = y_tensor[i : i + self.batch_size]

                outputs = self.model_(batch_X)
                loss = self.criterion_(outputs, batch_y)

                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()

            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Make predictions using the fitted LSTM model."""
        self._check_is_fitted()

        X = self._validate_input(X)

        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        X_scaled = self.scaler_.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)

        X_tensor = self._prepare_data(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor)
            predictions = predictions.cpu().numpy()

        return predictions.squeeze()
