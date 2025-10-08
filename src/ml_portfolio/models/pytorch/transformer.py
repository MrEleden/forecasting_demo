"""
Transformer model for time series forecasting.

Uses self-attention mechanisms to capture long-range dependencies.
Excellent for complex temporal patterns and multivariate series.
"""

import math
from typing import Any, Dict, Optional

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
        self.default_grad_clip = 0.5

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

        # Default optimizer/loss will be created during training via base class hooks

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

    def predict(self, X: Any, **kwargs) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features as DataFrame, numpy array, or tensor-like
            **kwargs: Additional prediction parameters

        Returns:
            predictions: Predicted values
        """
        self.eval()

        with torch.no_grad():
            X_tensor = self._prepare_inputs(X)

            outputs = self.forward(X_tensor, X_tensor)
            predictions = outputs[-1].detach().cpu().numpy()

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        elif predictions.ndim == 0:
            predictions = np.array([predictions])

        return predictions

    def _prepare_batch(self, X: Any, y: Any, training: bool = True):
        """Convert batches to seq-first tensors for transformer."""

        inputs = self._prepare_inputs(X)
        batch_size = inputs.shape[1]
        targets = self._prepare_targets(y, expected_batch=batch_size)
        return inputs, targets

    def _training_step(self, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module):
        """Custom training step using transformer decoder structure."""

        outputs = self.forward(inputs, inputs)
        predictions = outputs[-1]
        loss = criterion(predictions, targets)
        return loss, predictions

    def _validation_step(self, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module):
        """Validation step mirroring the training computation."""

        outputs = self.forward(inputs, inputs)
        predictions = outputs[-1]
        return criterion(predictions, targets)

    def _to_device_tensor(self, data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert arbitrary input to a tensor on the configured device."""

        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=dtype)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            array = data.values
        elif isinstance(data, np.ndarray):
            array = data
        else:
            array = np.asarray(data)

        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def _prepare_inputs(self, X: Any) -> torch.Tensor:
        """Ensure inputs have shape (seq_len, batch, features)."""

        tensor = self._to_device_tensor(X)

        if tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            # assume batch-first (batch, seq, features)
            tensor = tensor.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"Unsupported input shape for Transformer: {tensor.shape}")

        return tensor

    def _prepare_targets(self, y: Any, expected_batch: Optional[int] = None) -> torch.Tensor:
        """Ensure targets have shape (batch, output_size)."""

        tensor = self._to_device_tensor(y)

        if tensor.dim() == 0:
            tensor = tensor.view(1, 1)
        elif tensor.dim() == 1:
            tensor = tensor.view(-1, 1)
        else:
            tensor = tensor.reshape(tensor.shape[0], -1)

        if expected_batch is not None and tensor.shape[0] != expected_batch:
            tensor = tensor.reshape(expected_batch, -1)

        return tensor

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "learning_rate": self.learning_rate,
            **self.kwargs,
        }

        if hasattr(self, "batch_size"):
            params["batch_size"] = self.batch_size
        if hasattr(self, "num_epochs"):
            params["num_epochs"] = self.num_epochs

        return params

    def set_params(self, **params) -> "TransformerForecaster":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        # Rebuild if architecture params changed
        arch_params = ["input_size", "d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward"]
        if any(k in params for k in arch_params):
            self._build_model()
        return self
