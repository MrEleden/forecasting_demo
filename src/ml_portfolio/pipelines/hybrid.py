"""
Hybrid pipelines combining sklearn preprocessing with PyTorch models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


class SklearnTorchWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper to make PyTorch models compatible with sklearn pipelines.
    """

    def __init__(
        self,
        model_class,
        model_params: Dict[str, Any] = None,
        training_params: Dict[str, Any] = None,
        device: str = "cpu",
    ):
        """
        Initialize SklearnTorchWrapper.

        Args:
            model_class: PyTorch model class
            model_params: Parameters for model initialization
            training_params: Parameters for training
            device: Device to run model on
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.device = device
        self.model_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def fit(self, X, y):
        """
        Fit the PyTorch model.

        Args:
            X: Features
            y: Target values

        Returns:
            self
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for hybrid pipelines")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Scale features and target
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_scaled = self.scaler_X_.fit_transform(X)
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.training_params.get("batch_size", 32), shuffle=True)

        # Initialize model
        input_size = X.shape[1]
        self.model_params["input_size"] = input_size
        self.model_ = self.model_class(**self.model_params).to(self.device)

        # Set up optimizer and loss
        optimizer = optim.Adam(self.model_.parameters(), lr=self.training_params.get("learning_rate", 0.001))
        criterion = nn.MSELoss()

        # Training loop
        self.model_.train()
        epochs = self.training_params.get("epochs", 100)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions with the fitted model.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for hybrid pipelines")

        # Convert to numpy and scale
        X = np.array(X)
        X_scaled = self.scaler_X_.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Make predictions
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy()

        # Inverse transform predictions
        predictions = self.scaler_y_.inverse_transform(predictions.reshape(-1, 1)).ravel()

        return predictions


class HybridLSTM(SklearnTorchWrapper):
    """
    Hybrid LSTM model with sklearn preprocessing.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0, **kwargs):
        """
        Initialize HybridLSTM.

        Args:
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            **kwargs: Additional arguments for wrapper
        """
        try:
            from ..models.forecasting.lstm import SimpleLSTM

            model_class = SimpleLSTM
        except ImportError:
            warnings.warn("LSTM model not available, using placeholder")
            model_class = None

        model_params = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout, "output_size": 1}

        super().__init__(model_class=model_class, model_params=model_params, **kwargs)


class HybridTransformer(SklearnTorchWrapper):
    """
    Hybrid Transformer model with sklearn preprocessing.
    """

    def __init__(self, d_model: int = 64, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1, **kwargs):
        """
        Initialize HybridTransformer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            **kwargs: Additional arguments for wrapper
        """
        try:
            from ..models.forecasting.transformer import SimpleTransformer

            model_class = SimpleTransformer
        except ImportError:
            warnings.warn("Transformer model not available, using placeholder")
            model_class = None

        model_params = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "output_size": 1,
        }

        super().__init__(model_class=model_class, model_params=model_params, **kwargs)


def create_hybrid_pipeline(
    model_type: str = "lstm",
    preprocessing_steps: Optional[List] = None,
    model_params: Dict[str, Any] = None,
    training_params: Dict[str, Any] = None,
) -> Pipeline:
    """
    Create a hybrid pipeline with sklearn preprocessing and PyTorch model.

    Args:
        model_type: Type of model ('lstm', 'transformer', 'tcn')
        preprocessing_steps: Custom preprocessing steps
        model_params: Parameters for model
        training_params: Parameters for training

    Returns:
        Hybrid pipeline
    """
    if model_params is None:
        model_params = {}
    if training_params is None:
        training_params = {}

    # Default preprocessing
    if preprocessing_steps is None:
        preprocessing_steps = [("scaler", StandardScaler())]

    # Select model
    if model_type == "lstm":
        model = HybridLSTM(**model_params, training_params=training_params)
    elif model_type == "transformer":
        model = HybridTransformer(**model_params, training_params=training_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create pipeline
    steps = preprocessing_steps + [("model", model)]
    return Pipeline(steps)


class EnsembleHybrid(BaseEstimator, RegressorMixin):
    """
    Ensemble of hybrid models with different architectures.
    """

    def __init__(self, models: List[Dict[str, Any]], weights: Optional[List[float]] = None):
        """
        Initialize EnsembleHybrid.

        Args:
            models: List of model configurations
            weights: Weights for ensemble (None for equal weights)
        """
        self.models = models
        self.weights = weights
        self.fitted_models_ = []

    def fit(self, X, y):
        """
        Fit all models in the ensemble.

        Args:
            X: Features
            y: Target values

        Returns:
            self
        """
        self.fitted_models_ = []

        for model_config in self.models:
            model_type = model_config.get("type", "lstm")
            model_params = model_config.get("model_params", {})
            training_params = model_config.get("training_params", {})
            preprocessing_steps = model_config.get("preprocessing_steps", None)

            pipeline = create_hybrid_pipeline(
                model_type=model_type,
                preprocessing_steps=preprocessing_steps,
                model_params=model_params,
                training_params=training_params,
            )

            pipeline.fit(X, y)
            self.fitted_models_.append(pipeline)

        # Set equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.fitted_models_)] * len(self.fitted_models_)

        return self

    def predict(self, X):
        """
        Make ensemble predictions.

        Args:
            X: Features

        Returns:
            Ensemble predictions
        """
        if not self.fitted_models_:
            raise ValueError("Ensemble must be fitted before making predictions")

        predictions = []
        for model in self.fitted_models_:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_pred


def create_ensemble_config(
    model_types: List[str] = ["lstm", "transformer"], base_params: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Create configuration for ensemble of hybrid models.

    Args:
        model_types: Types of models to include
        base_params: Base parameters for all models

    Returns:
        List of model configurations
    """
    if base_params is None:
        base_params = {}

    configs = []

    for model_type in model_types:
        config = {
            "type": model_type,
            "model_params": base_params.get("model_params", {}),
            "training_params": base_params.get("training_params", {}),
            "preprocessing_steps": base_params.get("preprocessing_steps", None),
        }
        configs.append(config)

    return configs
