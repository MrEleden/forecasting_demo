"""
Training callbacks for checkpoints, scheduling, and monitoring.
"""

from pathlib import Path


class Callback:
    """Base callback class."""

    def on_epoch_start(self, epoch, logs=None):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        pass

    def on_training_start(self, logs=None):
        """Called at the start of training."""
        pass

    def on_training_end(self, logs=None):
        """Called at the end of training."""
        pass


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""

    def __init__(self, filepath, monitor="val_loss", save_best_only=True):
        """
        Initialize ModelCheckpoint.

        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: Whether to save only the best model
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if conditions are met."""
        if logs is None:
            logs = {}

        current_score = logs.get(self.monitor, float("inf"))

        if not self.save_best_only or current_score < self.best_score:
            self.best_score = current_score
            # Save model logic would go here
            print(f"Checkpoint saved at epoch {epoch}")


class EarlyStopping(Callback):
    """Stop training when monitored metric stops improving."""

    def __init__(self, monitor="val_loss", patience=10, min_delta=0):
        """
        Initialize EarlyStopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """Check if training should stop."""
        if logs is None:
            logs = {}

        current_score = logs.get(self.monitor, float("inf"))

        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(f"Early stopping at epoch {epoch}")
                return True  # Signal to stop training

        return False


class LearningRateScheduler(Callback):
    """Adjust learning rate during training."""

    def __init__(self, schedule_fn):
        """
        Initialize LearningRateScheduler.

        Args:
            schedule_fn: Function that takes epoch and returns learning rate
        """
        self.schedule_fn = schedule_fn

    def on_epoch_start(self, epoch, logs=None):
        """Update learning rate."""
        new_lr = self.schedule_fn(epoch)
        print(f"Learning rate updated to {new_lr}")
        # Update optimizer learning rate here
