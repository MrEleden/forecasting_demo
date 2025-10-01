"""
Training Engine for ML Portfolio Forecasting.

This module provides a unified training interface that works with any model
implementing .fit() and .predict() methods. It handles training loops,
validation, metrics computation, and testing.
"""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class TrainingEngine:
    """
    Unified training engine for forecasting models.

    Handles training, validation, and testing with support for:
    - Any model with .fit() and .predict() methods
    - DataLoader-based batch training
    - Metric computation and logging
    - Early stopping (for iterative models)

    Usage:
        # Initialize engine with model, metrics, and dataloaders
        engine = TrainingEngine(
            model=model,
            metrics=metrics,
            train_dataset=train_loader,
            val_dataset=val_loader,
            test_dataset=test_loader
        )

        # Train model
        results = engine.train()

        # Evaluate on test set
        test_metrics = engine.evaluate_test()
    """

    def __init__(
        self,
        model,
        metrics: Optional[Dict[str, Any]] = None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        optimizer=None,
        scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TrainingEngine.

        Args:
            model: Model with .fit(X, y) and .predict(X) methods
            metrics: Dictionary of metric functions {name: metric_fn}
            train_dataset: Training dataloader (iterator yielding batches)
            val_dataset: Validation dataloader (iterator yielding batches)
            test_dataset: Test dataloader (iterator yielding batches)
            optimizer: PyTorch optimizer (optional, for deep learning models)
            scheduler: Learning rate scheduler (optional, for deep learning models)
            config: Training configuration (optional)
        """
        self.model = model
        self.metrics = metrics or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}

        # Training parameters
        self.max_epochs = self.config.get("max_epochs", 100)
        self.patience = self.config.get("patience", 10)
        self.early_stopping = self.config.get("early_stopping", True)
        self.verbose = self.config.get("verbose", True)
        self.monitor_metric = self.config.get("monitor_metric", "val_loss")
        self.min_delta = self.config.get("min_delta", 1e-4)

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": [], "epoch_times": []}

        # Early stopping state
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

        if self.verbose:
            logger.info(f"TrainingEngine initialized for {self.model.__class__.__name__}")

    def train(self) -> Dict[str, Any]:
        """
        Train the model using training dataloader.

        Returns:
            Dictionary with training results:
                - history: Training history
                - best_epoch: Best epoch number
                - converged: Whether training converged
                - total_training_time: Total training time
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"Training {self.model.__class__.__name__}")
            logger.info(f"Max epochs: {self.max_epochs}")
            logger.info("=" * 60)

        start_time = time.time()

        # Unified training (epochs from config)
        results = self._train()

        total_time = time.time() - start_time
        results["total_training_time"] = total_time

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"Training completed in {total_time:.2f}s")
            logger.info(f"Best epoch: {results['best_epoch']}")
            logger.info(f"Converged: {results['converged']}")
            logger.info("=" * 60)

        return results

    def _train(self) -> Dict[str, Any]:
        """
        Unified training function for all model types.

        Training behavior is controlled by config:
        - Single-shot (sklearn): max_epochs=1, batch_size=inf (dataloader gives all data)
        - Iterative (PyTorch): max_epochs>1, batch_size=32/64/etc (dataloader yields mini-batches)

        The dataloader handles batching internally. Engine just loops through epochs and batches.

        Returns:
            Training results dictionary
        """
        best_val_metric = float("inf")

        # Train for specified number of epochs
        for epoch in range(self.max_epochs):
            epoch_start = time.time()

            if self.verbose:
                logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")

            # Train on all batches from dataloader
            for X_batch, y_batch in self.train_dataset:
                # Dataloader provides X_batch, y_batch as numpy arrays
                self.model.fit(X_batch, y_batch)

            epoch_time = time.time() - epoch_start

            # Compute metrics after epoch
            train_metrics = self._evaluate_loader(self.train_dataset, prefix="train_")
            val_metrics = self._evaluate_loader(self.val_dataset, prefix="val_") if self.val_dataset else {}

            # Store history
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)
            self.history["epoch_times"].append(epoch_time)

            if self.verbose:
                self._log_metrics(epoch, train_metrics, val_metrics, epoch_time)

            # Learning rate scheduler step (for PyTorch models)
            if self.scheduler is not None:
                # Check if scheduler needs metric (ReduceLROnPlateau)
                if hasattr(self.scheduler, "step") and "metrics" in self.scheduler.step.__code__.co_varnames:
                    # ReduceLROnPlateau needs validation metric
                    current_metric = val_metrics.get(self.monitor_metric, float("inf"))
                    self.scheduler.step(current_metric)
                else:
                    # Regular schedulers (StepLR, CosineAnnealingLR, etc.)
                    self.scheduler.step()

                if self.verbose and hasattr(self.optimizer, "param_groups"):
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(f"Learning rate: {current_lr:.6f}")

            # Early stopping check
            if self.early_stopping and self.val_dataset is not None:
                current_metric = val_metrics.get(self.monitor_metric, float("inf"))

                if current_metric < best_val_metric - self.min_delta:
                    best_val_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    if self.verbose:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        logger.info(f"Best {self.monitor_metric}: {best_val_metric:.4f} at epoch {self.best_epoch + 1}")
                    break

        # Final best metric
        final_metric = (
            best_val_metric
            if self.early_stopping
            else val_metrics.get(self.monitor_metric, float("inf")) if val_metrics else float("inf")
        )

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "converged": self.patience_counter < self.patience,
            "best_metric": final_metric,
        }

    def _evaluate_loader(self, dataloader, prefix: str = "") -> Dict[str, float]:
        """
        Evaluate model on a dataloader.

        Args:
            dataloader: DataLoader to evaluate on
            prefix: Prefix for metric names (e.g., "train_", "val_")

        Returns:
            Dictionary of computed metrics
        """
        if dataloader is None:
            return {}

        y_true_list = []
        y_pred_list = []

        # Collect predictions
        for X_batch, y_batch in dataloader:
            y_pred_batch = self.model.predict(X_batch)
            y_true_list.append(y_batch)
            y_pred_list.append(y_pred_batch)

        # Concatenate all batches
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Compute metrics
        return self._compute_metrics(y_true, y_pred, prefix)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Compute all configured metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names

        Returns:
            Dictionary of computed metrics
        """
        results = {}

        for metric_name, metric_fn in self.metrics.items():
            try:
                value = metric_fn(y_true, y_pred)
                results[f"{prefix}{metric_name}"] = float(value)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                results[f"{prefix}{metric_name}"] = float("nan")

        return results

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """
        Log metrics for current epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch_time: Time taken for epoch
        """
        logger.info(f"Epoch {epoch + 1}/{self.max_epochs} - {epoch_time:.2f}s")

        # Log training metrics
        for metric_name, value in train_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Log validation metrics
        for metric_name, value in val_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    def evaluate_test(self) -> Dict[str, float]:
        """
        Evaluate model on test dataset.

        Returns:
            Dictionary of test metrics
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        if self.verbose:
            logger.info("Evaluating on test set...")

        test_metrics = self._evaluate_loader(self.test_dataset, prefix="test_")

        if self.verbose:
            logger.info("Test Results:")
            for metric_name, value in sorted(test_metrics.items()):
                logger.info(f"  {metric_name}: {value:.4f}")

        return test_metrics

    def predict(self, dataloader) -> np.ndarray:
        """
        Generate predictions for a dataloader.

        Args:
            dataloader: DataLoader to predict on

        Returns:
            Array of predictions
        """
        predictions = []

        for X_batch, _ in dataloader:
            y_pred = self.model.predict(X_batch)
            predictions.append(y_pred)

        return np.concatenate(predictions, axis=0)

    def get_history(self) -> Dict[str, list]:
        """
        Get training history.

        Returns:
            Dictionary containing training history
        """
        return self.history
