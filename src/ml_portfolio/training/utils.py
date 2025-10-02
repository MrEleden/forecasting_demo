"""
Training utilities for seed management, device handling, and logging.
"""

import random
import numpy as np
import logging
from typing import Optional

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device for training.

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def setup_logging(level=logging.INFO, format_str=None):
    """
    Set up logging configuration.

    Args:
        level: Logging level
        format_str: Custom format string
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=format_str)


class TrainingLogger:
    """
    Logger for training progress and metrics.
    """

    def __init__(self, name: str = "training"):
        """
        Initialize TrainingLogger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.metrics_history = {}

    def log_epoch(self, epoch: int, metrics: dict):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        # Log to console
        metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metric_str}")

        # Store in history
        for metric, value in metrics.items():
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append(value)

    def get_history(self) -> dict:
        """Get metrics history."""
        return self.metrics_history


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: Model to count parameters for

    Returns:
        Number of trainable parameters
    """
    if TORCH_AVAILABLE and isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # For non-PyTorch models, return 0 or implement other counting logic
    return 0


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    if not TORCH_AVAILABLE:
        # For non-PyTorch models, implement alternative saving
        return

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load training checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Epoch and loss from checkpoint
    """
    if not TORCH_AVAILABLE:
        # For non-PyTorch models, implement alternative loading
        return 0, float("inf")

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]
