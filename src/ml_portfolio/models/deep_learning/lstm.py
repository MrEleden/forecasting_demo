"""
LSTM-based forecasting model with probabilistic outputs.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """
    LSTM model for time series forecasting with probabilistic outputs.

    Features:
    - Bidirectional LSTM
    - Multiple layers with dropout
    - Quantile regression for prediction intervals
    - Attention mechanism (optional)

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int, default=128
        LSTM hidden dimension
    num_layers : int, default=2
        Number of LSTM layers
    dropout : float, default=0.2
        Dropout rate between LSTM layers
    bidirectional : bool, default=False
        Use bidirectional LSTM
    use_attention : bool, default=False
        Add attention mechanism
    quantiles : List[float], optional
        Quantiles for probabilistic forecasting
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False,
        quantiles: Optional[List[float]] = None,
    ):
        super(LSTMForecaster, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention mechanism
        if use_attention:
            lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.attention = nn.Linear(lstm_output_size, 1)

        # Output layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size

        # Point forecast
        self.fc_point = nn.Linear(fc_input_size, 1)

        # Quantile forecasts
        self.fc_quantiles = nn.ModuleDict({f"q_{int(q*100)}": nn.Linear(fc_input_size, 1) for q in self.quantiles})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'point' and quantile predictions
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention if enabled
        if self.use_attention:
            # Compute attention weights
            attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
            # Weighted sum of LSTM outputs
            context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context = hidden[-1]

        # Point forecast
        point_pred = self.fc_point(context)

        # Quantile forecasts
        quantile_preds = {}
        for q in self.quantiles:
            key = f"q_{int(q*100)}"
            quantile_preds[key] = self.fc_quantiles[key](context)

        return {"point": point_pred, **quantile_preds}

    def quantile_loss(self, pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
        """
        Quantile loss (pinball loss).

        Parameters
        ----------
        pred : torch.Tensor
            Predictions
        target : torch.Tensor
            Target values
        quantile : float
            Quantile level (0-1)

        Returns
        -------
        torch.Tensor
            Quantile loss
        """
        error = target - pred
        loss = torch.where(error >= 0, quantile * error, (quantile - 1) * error)
        return loss.mean()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ):
        """
        Train the LSTM model.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size
        learning_rate : float, default=0.001
            Learning rate
        validation_split : float, default=0.2
            Fraction of data for validation
        early_stopping_patience : int, default=10
            Patience for early stopping
        verbose : bool, default=True
            Print training progress
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Split train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
        y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(batch_X)

                # Compute losses
                point_loss = nn.MSELoss()(outputs["point"], batch_y)

                quantile_losses = []
                for q in self.quantiles:
                    key = f"q_{int(q*100)}"
                    quantile_losses.append(self.quantile_loss(outputs[key], batch_y, q))

                total_loss = point_loss + sum(quantile_losses)

                # Backward pass
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.forward(batch_X)
                    point_loss = nn.MSELoss()(outputs["point"], batch_y)

                    quantile_losses = []
                    for q in self.quantiles:
                        key = f"q_{int(q*100)}"
                        quantile_losses.append(self.quantile_loss(outputs[key], batch_y, q))

                    total_loss = point_loss + sum(quantile_losses)
                    val_loss += total_loss.item()

            val_loss /= len(val_loader)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], " f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,)
        """
        self.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X_tensor)
            predictions = outputs["point"].cpu().numpy().squeeze()

        return predictions

    def predict_quantiles(self, X: np.ndarray, quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Make probabilistic predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        quantiles : List[float], optional
            Quantiles to predict. Uses model's quantiles if None.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping quantile strings to predictions
        """
        self.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)

        quantiles = quantiles or self.quantiles

        with torch.no_grad():
            outputs = self.forward(X_tensor)

            results = {}
            for q in quantiles:
                key = f"q_{int(q*100)}"
                if key in outputs:
                    results[str(q)] = outputs[key].cpu().numpy().squeeze()

            # Add point prediction as median if not present
            if "0.5" not in results:
                results["0.5"] = outputs["point"].cpu().numpy().squeeze()

        return results


class SequenceLSTMForecaster(LSTMForecaster):
    """
    LSTM for sequence-to-sequence forecasting.

    Extends LSTMForecaster to handle sequential inputs and multi-step outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        sequence_length: int = 10,
        forecast_horizon: int = 7,
        **kwargs,
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, bidirectional, **kwargs)

        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Override output layer for multi-step forecasting
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc_point = nn.Linear(fc_input_size, forecast_horizon)

        self.fc_quantiles = nn.ModuleDict(
            {f"q_{int(q*100)}": nn.Linear(fc_input_size, forecast_horizon) for q in self.quantiles}
        )
