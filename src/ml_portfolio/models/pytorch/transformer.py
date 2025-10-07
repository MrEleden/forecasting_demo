"""
Transformer model for time series forecasting.

Uses self-attention mechanisms to capture long-range dependencies.
Excellent for complex temporal patterns and multivariate series.
"""

import math
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..base import PyTorchForecaster


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series Transformer.

    Adds position information to input embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerForecaster(PyTorchForecaster, nn.Module):
    """
    Transformer-based time series forecaster.

    Uses self-attention mechanisms to capture long-range dependencies.
    Great for complex temporal patterns and multivariate series.

    Args:
        input_size: Number of input features
        output_size: Number of output features
        d_model: Dimension of transformer embeddings
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        device: Device to use ('cpu', 'cuda', 'auto')
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: str = "auto",
        **kwargs,
    ):
        PyTorchForecaster.__init__(self, device=device, **kwargs)
        nn.Module.__init__(self)

        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.learning_rate = learning_rate

        # Store any additional kwargs for later use
        self.kwargs = kwargs

        # Build model
        self.model = None
        self.optimizer = None
        self.criterion = None
        self._build_model()

    def _build_model(self):
        """Build Transformer architecture."""
        # Input embedding
        self.input_embedding = nn.Linear(self.input_size, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=False,  # (seq, batch, feature)
        )

        # Output layer
        self.fc_out = nn.Linear(self.d_model, self.output_size)

        # Move to device
        self.input_embedding.to(self.device)
        self.pos_encoder.to(self.device)
        self.transformer.to(self.device)
        self.fc_out.to(self.device)

        # Optimizer and loss
        params = (
            list(self.input_embedding.parameters())
            + list(self.transformer.parameters())
            + list(self.fc_out.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        """
        Forward pass.

        Args:
            src: Source sequence (seq_len, batch, features)
            tgt: Target sequence (tgt_len, batch, features)
        """
        # Embed inputs
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Generate causal mask for decoder
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(self.device)

        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Output projection
        output = self.fc_out(output)

        return output

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        verbose: bool = True,
        **kwargs,
    ) -> "TransformerForecaster":
        """Fit the Transformer model using dataloaders."""
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

                # For simplicity, use same input for src and tgt
                outputs = self.forward(batch_X, batch_X)

                # Take last timestep prediction
                if len(outputs.shape) == 3:
                    outputs = outputs[-1]  # (batch, output_size)

                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
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

                        val_outputs = self.forward(batch_X, batch_X)
                        if len(val_outputs.shape) == 3:
                            val_outputs = val_outputs[-1]

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

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features dataframe
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Predicted values
        """
        self.input_embedding.eval()
        self.transformer.eval()
        self.fc_out.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)

            # Reshape if needed
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)

            outputs = self.forward(X_tensor, X_tensor)
            predictions = outputs[-1].cpu().numpy().squeeze()

        return predictions

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            **self.kwargs,
        }

    def set_params(self, **params) -> "TransformerForecaster":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        # Rebuild if architecture params changed
        arch_params = ["input_size", "d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward"]
        if any(k in params for k in arch_params):
            self._build_model()
        return self
