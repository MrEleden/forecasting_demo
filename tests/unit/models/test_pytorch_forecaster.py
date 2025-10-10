"""Tests for the shared PyTorchForecaster training loop."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - pytest skip handles absence
    torch = None
    nn = None

from ml_portfolio.data.datasets import TimeSeriesDataset
from ml_portfolio.data.loaders import PyTorchDataLoader
from ml_portfolio.models.base import PyTorchForecaster

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")


if torch is not None:

    class LinearRegressionForecaster(PyTorchForecaster, nn.Module):
        """Minimal PyTorch forecaster exercising the base hooks."""

        def __init__(self, input_size: int = 2, device: str = "cpu"):
            PyTorchForecaster.__init__(self, device=device)
            nn.Module.__init__(self)

            self.linear = nn.Linear(input_size, 1)
            self.learning_rate = 0.2
            self.default_grad_clip = 1.0

            # Move parameters to the configured device
            self.to(self.device)

        def forward(self, inputs):
            return self.linear(inputs)

        def predict(self, X: np.ndarray) -> np.ndarray:
            tensor = self._to_tensor(X)
            self.eval()
            with torch.no_grad():
                outputs = self.forward(tensor)
            return outputs.detach().cpu().numpy().squeeze(-1)

        def _get_scheduler(self, optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    class ProgrammableForecaster(PyTorchForecaster, nn.Module):
        """PyTorch forecaster with programmable validation metrics for testing."""

        def __init__(self, input_size: int = 2, device: str = "cpu"):
            PyTorchForecaster.__init__(self, device=device)
            nn.Module.__init__(self)

            self.linear = nn.Linear(input_size, 1)
            self.learning_rate = 0.1
            self._val_iter = None

            self.to(self.device)

        def forward(self, inputs):
            return self.linear(inputs)

        def predict(self, X: np.ndarray) -> np.ndarray:
            tensor = self._to_tensor(X)
            self.eval()
            with torch.no_grad():
                outputs = self.forward(tensor)
            return outputs.detach().cpu().numpy().squeeze(-1)

        def set_validation_losses(self, losses):
            if losses is None:
                self._val_iter = None
            else:
                self._val_iter = iter(float(value) for value in losses)

        def _run_validation(self, loader, criterion):  # type: ignore[override]
            if self._val_iter is None:
                return super()._run_validation(loader, criterion)
            try:
                return next(self._val_iter)
            except StopIteration:
                return super()._run_validation(loader, criterion)

else:

    class LinearRegressionForecaster:  # pragma: no cover - placeholder for type checkers
        pass


def _build_synthetic_dataset(num_samples: int = 32):
    rng = np.random.default_rng(42)
    feature_1 = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)
    feature_2 = np.linspace(1.0, -1.0, num_samples, dtype=np.float32)
    noise = rng.normal(scale=0.05, size=num_samples).astype(np.float32)

    X = np.stack([feature_1, feature_2], axis=1)
    y = 2.0 * feature_1 - 3.0 * feature_2 + noise
    y = y.reshape(-1, 1)

    split = num_samples * 3 // 4
    train_dataset = TimeSeriesDataset(X[:split], y[:split])
    val_dataset = TimeSeriesDataset(X[split:], y[split:])
    return train_dataset, val_dataset


def test_pytorch_forecaster_training_loop_reduces_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    train_dataset, val_dataset = _build_synthetic_dataset()

    train_loader = PyTorchDataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = PyTorchDataLoader(val_dataset, batch_size=4, shuffle=False)

    model = LinearRegressionForecaster(device="cpu")
    criterion = nn.MSELoss()

    initial_inputs, initial_targets = next(iter(train_loader))
    initial_loss = criterion(model(initial_inputs), initial_targets).item()

    model.fit(
        train_loader,
        val_loader=val_loader,
        epochs=5,
        verbose=True,
        val_interval=1,
        log_interval=1,
    )

    final_inputs, final_targets = next(iter(train_loader))
    final_loss = criterion(model(final_inputs), final_targets).item()

    assert model.is_fitted is True
    assert final_loss < initial_loss


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")
def test_early_stopping_respects_patience_and_restores_best_state():
    train_dataset, val_dataset = _build_synthetic_dataset()

    train_loader = PyTorchDataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = PyTorchDataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model = ProgrammableForecaster(device="cpu")
    model.set_validation_losses([1.0, 0.8, 0.8, 0.81, 0.82])

    model.fit(
        train_loader,
        val_loader=val_loader,
        epochs=20,
        early_stopping=True,
        patience=2,
        min_delta=0.0,
        monitor_metric="val_loss",
        monitor_mode="min",
        val_interval=1,
        verbose=False,
    )

    assert model.trained_epochs == 4
    assert model.best_epoch == 1
    assert model.best_metric == pytest.approx(0.8)


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")
def test_unsupported_monitor_disables_early_stopping():
    train_dataset, val_dataset = _build_synthetic_dataset()

    train_loader = PyTorchDataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = PyTorchDataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model = ProgrammableForecaster(device="cpu")
    model.set_validation_losses([1.0, 0.9, 0.85])

    model.fit(
        train_loader,
        val_loader=val_loader,
        epochs=5,
        early_stopping=True,
        patience=1,
        monitor_metric="val_rmse",
        monitor_mode="min",
        val_interval=1,
        verbose=False,
    )

    assert model.trained_epochs == 5
    assert model.best_metric is None


@pytest.mark.skipif(torch is None, reason="PyTorch is required for these tests")
def test_train_loss_monitor_is_supported():
    train_dataset, _ = _build_synthetic_dataset()

    train_loader = PyTorchDataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    model = ProgrammableForecaster(device="cpu")
    model.set_validation_losses(None)

    model.fit(
        train_loader,
        val_loader=None,
        epochs=3,
        early_stopping=True,
        patience=1,
        monitor_metric="train_loss",
        monitor_mode="min",
        val_interval=1,
        verbose=False,
    )

    assert model.trained_epochs == 3
    assert model.best_metric is not None
