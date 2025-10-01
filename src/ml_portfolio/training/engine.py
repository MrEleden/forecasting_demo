"""
Training engine with train/validate loops and early stopping.
"""

import time
from typing import Optional, Dict, Any


class TrainingEngine:
    """
    Simple training engine for forecasting models.
    """

    def __init__(self, model, optimizer=None, loss_fn=None):
        """
        Initialize TrainingEngine.

        Args:
            model: Model to train
            optimizer: Optimizer for training
            loss_fn: Loss function
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        # Placeholder implementation
        train_loss = 0.1
        return train_loss

    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        # Placeholder implementation
        val_loss = 0.1
        return val_loss

    def fit(self, train_loader, val_loader=None, epochs=100, early_stopping_patience=10):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history
