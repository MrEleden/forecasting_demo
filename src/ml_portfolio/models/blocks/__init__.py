"""
Reusable neural network building blocks for time series forecasting.

This module contains common layers and components used across different
forecasting architectures.
"""

import numpy as np
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TemporalConvBlock(nn.Module):
        """
        Temporal Convolutional Block with dilated convolutions.
        """

        def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1
        ):
            """
            Initialize TemporalConvBlock.

            Args:
                in_channels: Number of input channels
                out_channels: Number of output channels
                kernel_size: Size of convolutional kernel
                dilation: Dilation rate for dilated convolution
                dropout: Dropout probability
            """
            super().__init__()

            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2
            )
            self.norm = nn.BatchNorm1d(out_channels)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            # Residual connection
            self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        def forward(self, x):
            """Forward pass."""
            residual = x if self.residual is None else self.residual(x)

            out = self.conv(x)
            out = self.norm(out)
            out = self.activation(out)
            out = self.dropout(out)

            return out + residual

    class MultiHeadAttention(nn.Module):
        """
        Multi-head attention mechanism for time series.
        """

        def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
            """
            Initialize MultiHeadAttention.

            Args:
                d_model: Model dimension
                n_heads: Number of attention heads
                dropout: Dropout probability
            """
            super().__init__()

            assert d_model % n_heads == 0

            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def forward(self, query, key, value, mask=None):
            """Forward pass."""
            batch_size, seq_len, _ = query.size()

            # Linear transformations
            Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # Attention
            attention_output = self._attention(Q, K, V, mask)

            # Concatenate heads
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

            return self.w_o(attention_output)

        def _attention(self, Q, K, V, mask=None):
            """Compute scaled dot-product attention."""
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            return torch.matmul(attention_weights, V)

    class PositionalEncoding(nn.Module):
        """
        Positional encoding for transformer-based models.
        """

        def __init__(self, d_model: int, max_len: int = 5000):
            """
            Initialize PositionalEncoding.

            Args:
                d_model: Model dimension
                max_len: Maximum sequence length
            """
            super().__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer("pe", pe)

        def forward(self, x):
            """Add positional encoding to input."""
            return x + self.pe[: x.size(0), :]

else:
    # Placeholder classes when PyTorch is not available
    class TemporalConvBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TemporalConvBlock")

    class MultiHeadAttention:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MultiHeadAttention")

    class PositionalEncoding:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PositionalEncoding")
