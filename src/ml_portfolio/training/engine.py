import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ml_portfolio.data.loaders import BaseDataLoader, SimpleDataLoader
from ml_portfolio.evaluation.metrics import mae, mape, rmse
from ml_portfolio.models.base import BaseForecaster, StatisticalForecaster

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """
    Abstract base class for all training engines.

    Expects DataLoaders that implement __iter__ for data iteration.
    Both PyTorch and Statistical engines use the same iteration pattern:
    `for batch_x, batch_y in loader:`

    The difference is in what constitutes a "batch":
    - PyTorchEngine: Many small batches (mini-batch training)
    - StatisticalEngine: One large batch (full dataset)
    """

    def __init__(
        self,
        model: BaseForecaster,
        train_loader: BaseDataLoader = None,
        val_loader: BaseDataLoader = None,
        test_loader: BaseDataLoader = None,
        mlflow_tracker: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize base engine with model and data loaders.

        Args:
            model: Model to train
            train_loader: Training DataLoader (must implement __iter__)
            val_loader: Validation DataLoader (must implement __iter__)
            test_loader: Test DataLoader (must implement __iter__)
            mlflow_tracker: MLflow tracker for experiment logging
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to log training progress
            metrics: Dictionary of metric functions to compute
            **kwargs: Additional parameters for subclasses
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mlflow_tracker = mlflow_tracker
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.metrics = metrics or {}
        self.preprocessing_pipeline = kwargs.get("preprocessing_pipeline", None)

        # Training history
        self.history = {"train_metrics": [], "val_metrics": [], "test_metrics": [], "epoch_times": []}

        # Store additional parameters
        self.engine_params = kwargs

        if self.verbose:
            logger.info(f"Engine initialized for {self.model.__class__.__name__}")
            logger.info(f"Train loader: {self._describe_loader(train_loader)}")
            if val_loader:
                logger.info(f"Val loader: {self._describe_loader(val_loader)}")
            if test_loader:
                logger.info(f"Test loader: {self._describe_loader(test_loader)}")

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the model using the provided data loaders.

        Implementation varies significantly:
        - PyTorchEngine: Multiple epochs, optimizer steps, early stopping, callbacks
        - StatisticalEngine: Single pass, direct model.fit()

        Returns:
            Dictionary with training results
        """
        pass

    @abstractmethod
    def evaluate(self, loader: BaseDataLoader) -> Dict[str, float]:
        """
        Evaluate model on a data loader.

        Should iterate over the loader:
        `for batch_x, batch_y in loader:`

        Args:
            loader: DataLoader to evaluate on

        Returns:
            Dictionary of metrics
        """
        pass

    def test(self) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Test metrics dictionary
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided")

        if self.verbose:
            logger.info("Evaluating on test set...")

        metrics = self.evaluate(self.test_loader)
        self.history["test_metrics"].append(metrics)

        if self.verbose:
            logger.info("Test metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def _describe_loader(self, loader: Any) -> str:
        """Get description of data loader."""
        if loader is None:
            return "None"

        # Check for common attributes
        descriptions = []

        # Check if it has length
        if hasattr(loader, "__len__"):
            try:
                descriptions.append(f"{len(loader)} batches")
            except Exception:
                pass

        # Check if it's a known type
        loader_type = type(loader).__name__
        descriptions.append(loader_type)

        # Check for batch_size attribute
        if hasattr(loader, "batch_size"):
            descriptions.append(f"batch_size={loader.batch_size}")

        # Check for dataset size
        if hasattr(loader, "n_samples"):
            descriptions.append(f"n_samples={loader.n_samples}")
        elif hasattr(loader, "dataset"):
            if hasattr(loader.dataset, "__len__"):
                descriptions.append(f"dataset_size={len(loader.dataset)}")

        return ", ".join(descriptions) if descriptions else "Unknown loader type"

    def _log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = ""):
        """Log metrics to console and MLflow."""
        if not self.verbose:
            return

        # Log to console
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if phase:
            logger.info(f"Epoch {epoch + 1} [{phase}]: {metric_str}")
        else:
            logger.info(f"Epoch {epoch + 1}: {metric_str}")

        # Log to MLflow if available
        if self.mlflow_tracker:
            try:
                if phase:
                    prefixed_metrics = {f"{phase}_{k}": v for k, v in metrics.items()}
                else:
                    prefixed_metrics = metrics
                self.mlflow_tracker.log_metrics(prefixed_metrics, step=epoch)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None and self.checkpoint_dir:
            if is_best:
                filename = "best_model.pkl"  # Use .pkl as default for compatibility
            else:
                epoch_num = len(self.history["train_metrics"])
                filename = f"checkpoint_epoch_{epoch_num}.pkl"
            path = self.checkpoint_dir / filename

        if path:
            path = Path(path)
            try:
                self._save_model_state(path)
                if self.verbose:
                    logger.info(f"Saved checkpoint to {path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    @abstractmethod
    def _save_model_state(self, path: Path):
        """
        Save model state to file.

        Implementation depends on model type:
        - PyTorch: Save state_dict
        - Statistical: Pickle entire model
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path):
        """
        Load model state from file.

        Implementation depends on model type:
        - PyTorch: Load state_dict
        - Statistical: Unpickle model
        """
        pass

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.

        Returns:
            Dictionary with training summary statistics
        """
        summary = {
            "model_type": self.model.__class__.__name__,
            "total_epochs": len(self.history["train_metrics"]),
            "total_training_time": sum(self.history["epoch_times"]),
        }

        # Add best metrics if available
        if self.history["train_metrics"]:
            # Get last epoch metrics
            summary["final_train_metrics"] = self.history["train_metrics"][-1]

        if self.history["val_metrics"]:
            # Get best validation metrics (assuming lower is better for loss)
            val_losses = [m.get("loss", float("inf")) for m in self.history["val_metrics"]]
            if val_losses:
                best_idx = np.argmin(val_losses)
                summary["best_val_metrics"] = self.history["val_metrics"][best_idx]
                summary["best_epoch"] = best_idx + 1

        if self.history["test_metrics"]:
            summary["test_metrics"] = self.history["test_metrics"][-1]

        return summary


class StatisticalEngine(BaseEngine):
    """
    Training engine for statistical models (ARIMA, sklearn).
    Handles models that fit in a single pass rather than iterative training.
    Expects SimpleDataLoader that yields full dataset in one iteration.
    """

    def __init__(
        self,
        model: StatisticalForecaster,
        train_loader: SimpleDataLoader = None,
        val_loader: SimpleDataLoader = None,
        test_loader: SimpleDataLoader = None,
        mlflow_tracker: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize statistical engine.

        Args:
            model: Statistical model with fit/predict interface
            train_loader: Training data loader (SimpleDataLoader expected)
            val_loader: Validation data loader
            test_loader: Test data loader
            mlflow_tracker: MLflow tracker for experiment logging
            checkpoint_dir: Directory to save model
            verbose: Whether to log progress
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            mlflow_tracker=mlflow_tracker,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
            **kwargs,
        )

    def train(self) -> Dict[str, Any]:
        """
        Train statistical model with single fit call.

        Returns:
            Dictionary with training results
        """
        if self.train_loader is None:
            raise ValueError("No training data loader provided")

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"Training {self.model.__class__.__name__}")
            logger.info("=" * 60)

        start_time = time.time()

        # Pass dataloaders to model - model handles iteration internally
        self.model.fit(self.train_loader, self.val_loader)

        # Compute training metrics after fitting
        train_metrics = {}
        for X_train, y_train in self.train_loader:
            y_pred_train = self.model.predict(X_train)
            train_metrics = self._compute_metrics(y_train, y_pred_train, prefix="train_")
            self.history["train_metrics"].append(train_metrics)
            break  # Single iteration for statistical models

        # Compute validation metrics if available
        val_metrics = {}
        if self.val_loader is not None:
            for X_val, y_val in self.val_loader:
                y_pred_val = self.model.predict(X_val)
                val_metrics = self._compute_metrics(y_val, y_pred_val, prefix="val_")
                self.history["val_metrics"].append(val_metrics)
                break  # Single iteration for statistical models

        # Training time
        training_time = time.time() - start_time
        self.history["epoch_times"].append(training_time)

        # Log metrics
        if self.verbose:
            logger.info(f"Training completed in {training_time:.2f}s")
            for metric_name, value in train_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            for metric_name, value in val_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

        # Log to MLflow
        if self.mlflow_tracker:
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["training_time"] = training_time
            try:
                self.mlflow_tracker.log_metrics(all_metrics)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        # Save checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(is_best=True)

        return {
            "history": self.history,
            "training_time": training_time,
            "converged": True,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def test(self) -> Dict[str, float]:
        """
        Evaluate model on test set with inverse transform for interpretable metrics.

        Returns:
            Test metrics dictionary
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided")

        if self.verbose:
            logger.info("Evaluating on test set...")

        # Iterate over loader (single iteration for full data)
        for X, y in self.test_loader:
            # Make predictions on scaled data
            y_pred = self.model.predict(X)

            # Apply inverse transform if preprocessing pipeline is available
            if self.preprocessing_pipeline is not None:
                try:
                    if self.verbose:
                        logger.info(
                            f"Before inverse: y range=[{y.min():.4f}, {y.max():.4f}], "
                            f"y_pred range=[{y_pred.min():.4f}, {y_pred.max():.4f}]"
                        )

                    y_true_original = self.preprocessing_pipeline.inverse_transform_target(y)
                    y_pred_original = self.preprocessing_pipeline.inverse_transform_target(y_pred)

                    if self.verbose:
                        logger.info(
                            f"After inverse: y range=[{y_true_original.min():.2f}, "
                            f"{y_true_original.max():.2f}], "
                            f"y_pred range=[{y_pred_original.min():.2f}, "
                            f"{y_pred_original.max():.2f}]"
                        )
                        logger.info("Applied inverse transform to predictions and targets")

                    # Compute metrics on original scale
                    metrics = self._compute_metrics(y_true_original, y_pred_original)
                except Exception as e:
                    logger.warning(f"Failed to apply inverse transform: {e}. Using scaled values.")
                    # Fallback to scaled metrics
                    metrics = self._compute_metrics(y, y_pred)
            else:
                # No preprocessing pipeline, compute on current scale
                metrics = self._compute_metrics(y, y_pred)

            self.history["test_metrics"].append(metrics)

            if self.verbose:
                logger.info("Test metrics:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")

            return metrics

        # Should never reach here if loader is properly configured
        return {}

    def evaluate(self, loader: BaseDataLoader) -> Dict[str, float]:
        """
        Evaluate statistical model on data.

        Args:
            loader: DataLoader to evaluate on

        Returns:
            Dictionary of metrics
        """
        # Iterate over loader (single iteration for full data)
        for X, y in loader:
            # Make predictions
            y_pred = self.model.predict(X)

            # Compute metrics
            return self._compute_metrics(y, y_pred)

        # Should never reach here if loader is properly configured
        return {}

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Compute metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names

        Returns:
            Dictionary of computed metrics
        """
        # Use configured metrics if available, otherwise use defaults
        if self.metrics:
            metrics_dict = self.metrics
        else:
            # Fallback to default metrics
            metrics_dict = {"mape": mape, "rmse": rmse, "mae": mae}

        # Ensure same shape
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        results = {}
        for metric_name, metric_fn in metrics_dict.items():
            try:
                value = metric_fn(y_true, y_pred)
                results[f"{prefix}{metric_name}"] = float(value)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                results[f"{prefix}{metric_name}"] = float("nan")

        return results

    def _save_model_state(self, path: Path):
        """
        Save statistical model via pickle.

        Args:
            path: Path to save model
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "history": self.history,
                },
                f,
            )

    def load_checkpoint(self, path: Path):
        """
        Load statistical model from pickle file.

        Args:
            path: Path to load model from
        """
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.model = checkpoint["model"]
        self.history = checkpoint.get("history", self.history)

        if self.verbose:
            logger.info(f"Loaded model from {path}")


class PyTorchEngine(BaseEngine):
    """
    Training engine for PyTorch-based deep learning models.
    Handles multi-epoch training with optimization, validation, and checkpointing.
    """

    def __init__(
        self,
        model,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        mlflow_tracker: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize PyTorch engine.

        Args:
            model: PyTorch model (inherits from PyTorchForecaster)
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            mlflow_tracker: MLflow tracker for experiment logging
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to log progress
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            mlflow_tracker=mlflow_tracker,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
            **kwargs,
        )

    def train(self, epochs: int = 100, learning_rate: float = 0.001, **kwargs) -> Dict[str, Any]:
        """
        Train PyTorch model for multiple epochs.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results
        """
        if self.train_loader is None:
            raise ValueError("No training data loader provided")

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"Training {self.model.__class__.__name__}")
            logger.info(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
            logger.info("=" * 60)

        start_time = time.time()

        # Model owns the training logic - just pass the loaders
        self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=self.verbose,
            **kwargs,
        )

        # Compute metrics after training
        training_time = time.time() - start_time

        # Evaluate on training set
        train_metrics = self.evaluate(self.train_loader, prefix="train_")
        self.history["train_metrics"].append(train_metrics)

        # Evaluate on validation set if available
        val_metrics = {}
        if self.val_loader is not None:
            val_metrics = self.evaluate(self.val_loader, prefix="val_")
            self.history["val_metrics"].append(val_metrics)

        self.history["epoch_times"].append(training_time)

        # Log to MLflow
        if self.mlflow_tracker:
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["training_time"] = training_time
            try:
                self.mlflow_tracker.log_metrics(all_metrics)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        # Save checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(is_best=True)

        if self.verbose:
            logger.info(f"Training completed in {training_time:.2f}s")
            for metric_name, value in train_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            for metric_name, value in val_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

        return {
            "history": self.history,
            "training_time": training_time,
            "converged": True,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def test(self) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Test metrics dictionary
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided")

        if self.verbose:
            logger.info("Evaluating on test set...")

        return self.evaluate(self.test_loader, prefix="test_")

    def evaluate(self, data_loader, prefix: str = "val_") -> Dict[str, float]:
        """
        Evaluate model on a data loader.

        Args:
            data_loader: DataLoader to evaluate
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_y_true = []
        all_y_pred = []

        # Collect predictions
        for X_batch, y_batch in data_loader:
            y_pred = self.model.predict(X_batch)
            all_y_true.append(y_batch)
            all_y_pred.append(y_pred)

        # Concatenate all batches
        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, prefix=prefix)

        return metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        from ml_portfolio.evaluation.metrics import mae, mape, rmse

        # Flatten arrays
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        results = {}
        try:
            results[f"{prefix}MAPE"] = float(mape(y_true, y_pred))
        except (ValueError, ZeroDivisionError):
            results[f"{prefix}MAPE"] = float("nan")

        try:
            results[f"{prefix}RMSE"] = float(rmse(y_true, y_pred))
        except (ValueError, ZeroDivisionError):
            results[f"{prefix}RMSE"] = float("nan")

        try:
            results[f"{prefix}MAE"] = float(mae(y_true, y_pred))
        except (ValueError, ZeroDivisionError):
            results[f"{prefix}MAE"] = float("nan")

        return results

    def _save_model_state(self, path: Path):
        """
        Save PyTorch model state dict.

        Args:
            path: Path to save model
        """
        import torch

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: Path):
        """
        Load PyTorch model from checkpoint.

        Args:
            path: Path to load model from
        """
        import torch

        checkpoint = torch.load(path, map_location=self.model.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)

        if self.verbose:
            logger.info(f"Loaded model from {path}")
