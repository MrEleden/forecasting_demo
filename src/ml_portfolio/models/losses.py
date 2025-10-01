"""
Loss functions optimized for time series forecasting.

This module provides specialized loss functions for forecasting tasks including
quantile losses, pinball loss, and SMAPE.
"""

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE loss
    """
    if TORCH_AVAILABLE and torch.is_tensor(y_true):
        return F.mse_loss(y_pred, y_true)
    else:
        return np.mean((y_true - y_pred) ** 2)


def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error loss.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE loss
    """
    if TORCH_AVAILABLE and torch.is_tensor(y_true):
        return F.l1_loss(y_pred, y_true)
    else:
        return np.mean(np.abs(y_true - y_pred))


def smape_loss(y_true, y_pred, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        SMAPE loss
    """
    if TORCH_AVAILABLE and torch.is_tensor(y_true):
        numerator = torch.abs(y_true - y_pred)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
        return torch.mean(numerator / denominator) * 100
    else:
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100


def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Quantile loss for probabilistic forecasting.

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level (0.5 for median)

    Returns:
        Quantile loss
    """
    if TORCH_AVAILABLE and torch.is_tensor(y_true):
        errors = y_true - y_pred
        loss = torch.max(quantile * errors, (quantile - 1) * errors)
        return torch.mean(loss)
    else:
        errors = y_true - y_pred
        loss = np.maximum(quantile * errors, (quantile - 1) * errors)
        return np.mean(loss)


def pinball_loss(y_true, y_pred, quantile=0.5):
    """
    Pinball loss (same as quantile loss, alternative name).

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level

    Returns:
        Pinball loss
    """
    return quantile_loss(y_true, y_pred, quantile)


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss (robust to outliers).

    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for switching between MSE and MAE

    Returns:
        Huber loss
    """
    if TORCH_AVAILABLE and torch.is_tensor(y_true):
        return F.huber_loss(y_pred, y_true, delta=delta)
    else:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * np.abs(error) - 0.5 * delta**2
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))


if TORCH_AVAILABLE:

    class QuantileLoss(nn.Module):
        """
        PyTorch module for quantile loss.
        """

        def __init__(self, quantile=0.5):
            """
            Initialize QuantileLoss.

            Args:
                quantile: Quantile level
            """
            super().__init__()
            self.quantile = quantile

        def forward(self, y_pred, y_true):
            """Forward pass."""
            return quantile_loss(y_true, y_pred, self.quantile)

    class SMAPELoss(nn.Module):
        """
        PyTorch module for SMAPE loss.
        """

        def __init__(self, epsilon=1e-8):
            """
            Initialize SMAPELoss.

            Args:
                epsilon: Small value to avoid division by zero
            """
            super().__init__()
            self.epsilon = epsilon

        def forward(self, y_pred, y_true):
            """Forward pass."""
            return smape_loss(y_true, y_pred, self.epsilon)

    class MultiQuantileLoss(nn.Module):
        """
        Multi-quantile loss for probabilistic forecasting.
        """

        def __init__(self, quantiles=[0.1, 0.5, 0.9]):
            """
            Initialize MultiQuantileLoss.

            Args:
                quantiles: List of quantile levels
            """
            super().__init__()
            self.quantiles = quantiles

        def forward(self, y_pred, y_true):
            """
            Forward pass.

            Args:
                y_pred: Predictions with shape (batch, time, n_quantiles)
                y_true: True values with shape (batch, time)

            Returns:
                Combined quantile loss
            """
            total_loss = 0

            for i, q in enumerate(self.quantiles):
                pred_q = y_pred[:, :, i]
                loss_q = quantile_loss(y_true, pred_q, q)
                total_loss += loss_q

            return total_loss / len(self.quantiles)

else:
    # Placeholder classes when PyTorch is not available
    class QuantileLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for QuantileLoss")

    class SMAPELoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SMAPELoss")

    class MultiQuantileLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MultiQuantileLoss")
