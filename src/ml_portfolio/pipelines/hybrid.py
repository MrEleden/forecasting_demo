"""
Hybrid pipelines combining sklearn preprocessing with PyTorch models.

This module provides base classes for hybrid forecasting pipelines that projects should inherit from.
Project-specific classes should extend these base classes for domain-specific customization.

Example inheritance patterns:
    - WalmartHybridLSTM(HybridLSTM)
    - OlaHybridTransformer(HybridTransformer)
    - InventoryEnsembleHybrid(EnsembleHybrid)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


class SklearnTorchWrapper(BaseEstimator, RegressorMixin):
    """
    Base wrapper to make PyTorch models compatible with sklearn pipelines.

    This is a BASE CLASS that projects should inherit from for domain-specific customization.

    Project-specific inheritance pattern:
        class WalmartLSTMWrapper(SklearnTorchWrapper):
            def __init__(self, **kwargs):
                # Walmart-specific model parameters
                model_params = {'hidden_size': 64, 'num_layers': 2}
                super().__init__(model_class=LSTMForecaster, model_params=model_params, **kwargs)

            def _preprocess_walmart_data(self, X):
                # Walmart-specific preprocessing
                return X

    Methods that projects commonly override:
        - fit(): Add domain-specific data preprocessing
        - predict(): Add domain-specific postprocessing
        - _get_model_params(): Customize model architecture for domain
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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HybridLSTM but not installed.")

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

    def _get_model_params(self):
        """
        Get model parameters. Override in project-specific classes for customization.

        Returns:
            Dict of model parameters
        """
        return self.model_params.copy()

    def _preprocess_features(self, X):
        """
        Preprocess features before model fitting/prediction.
        Override in project-specific classes for domain-specific preprocessing.

        Args:
            X: Features to preprocess

        Returns:
            Preprocessed features
        """
        return X

    def _postprocess_predictions(self, predictions):
        """
        Postprocess predictions after model prediction.
        Override in project-specific classes for domain-specific postprocessing.

        Args:
            predictions: Raw model predictions

        Returns:
            Postprocessed predictions
        """
        return predictions

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

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HybridLSTM but not installed.")

        # Apply project-specific preprocessing
        X = self._preprocess_features(X)

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

        # Apply project-specific postprocessing
        predictions = self._postprocess_predictions(predictions)

        return predictions


class HybridLSTM(SklearnTorchWrapper):
    """
    Base Hybrid LSTM model with sklearn preprocessing.

    This is a BASE CLASS for project-specific LSTM implementations.

    Project-specific inheritance examples:
        class WalmartHybridLSTM(HybridLSTM):
            def __init__(self, **kwargs):
                # Walmart-optimized parameters
                super().__init__(hidden_size=128, num_layers=3, dropout=0.2, **kwargs)

            def fit(self, X, y):
                # Add Walmart-specific feature engineering
                X_processed = self._add_holiday_features(X)
                return super().fit(X_processed, y)

        class OlaHybridLSTM(HybridLSTM):
            def __init__(self, **kwargs):
                # Rideshare-optimized parameters
                super().__init__(hidden_size=64, num_layers=2, dropout=0.1, **kwargs)
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
            from ..models.deep_learning.lstm import LSTMForecaster

            model_class = LSTMForecaster
        except ImportError:
            warnings.warn("LSTM model not available, using placeholder")
            model_class = None

        model_params = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout, "output_size": 1}

        super().__init__(model_class=model_class, model_params=model_params, **kwargs)


class HybridTransformer(SklearnTorchWrapper):
    """
    Base Hybrid Transformer model with sklearn preprocessing.

    This is a BASE CLASS for project-specific Transformer implementations.

    Project-specific inheritance examples:
        class InventoryHybridTransformer(HybridTransformer):
            def __init__(self, **kwargs):
                # Inventory-optimized parameters for longer sequences
                super().__init__(d_model=128, nhead=8, num_layers=4, **kwargs)

            def _add_inventory_features(self, X):
                # Add inventory-specific features (lead times, reorder points)
                return X

        class TSIHybridTransformer(HybridTransformer):
            def __init__(self, **kwargs):
                # Economic indicators optimized parameters
                super().__init__(d_model=64, nhead=4, num_layers=2, **kwargs)
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
            from ..models.deep_learning.transformer import TransformerForecaster

            model_class = TransformerForecaster
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
    Base ensemble of hybrid models with different architectures.

    This is a BASE CLASS for project-specific ensemble implementations.

    Project-specific inheritance examples:
        class WalmartEnsemble(EnsembleHybrid):
            def __init__(self):
                # Walmart-specific ensemble configuration
                models = [
                    {'type': 'lstm', 'model_params': {'hidden_size': 128}},
                    {'type': 'transformer', 'model_params': {'d_model': 64}},
                ]
                super().__init__(models=models, weights=[0.6, 0.4])

            def _get_walmart_models(self):
                # Return domain-specific model configurations
                pass

        class OlaEnsemble(EnsembleHybrid):
            def __init__(self):
                # Rideshare-specific ensemble for demand prediction
                models = [
                    {'type': 'lstm', 'model_params': {'hidden_size': 64}},
                    {'type': 'transformer', 'model_params': {'d_model': 32}},
                ]
                super().__init__(models=models)
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


# Factory Functions for Project-Specific Class Creation


def create_project_hybrid_class(project_name: str, model_type: str, base_class=None):
    """
    Factory function to create project-specific hybrid classes following naming conventions.

    This function creates classes that follow the pattern: {ProjectName}Hybrid{ModelType}

    Args:
        project_name: Name of the project (e.g., "Walmart", "Ola", "Inventory", "TSI")
        model_type: Type of model (e.g., "LSTM", "Transformer")
        base_class: Base class to inherit from (auto-detected if None)

    Returns:
        Project-specific hybrid class

    Example:
        WalmartHybridLSTM = create_project_hybrid_class("Walmart", "LSTM")
        OlaHybridTransformer = create_project_hybrid_class("Ola", "Transformer")
    """
    if base_class is None:
        if model_type.upper() == "LSTM":
            base_class = HybridLSTM
        elif model_type.upper() == "TRANSFORMER":
            base_class = HybridTransformer
        else:
            base_class = SklearnTorchWrapper

    class_name = f"{project_name}Hybrid{model_type}"

    class ProjectHybridClass(base_class):
        """Dynamically created project-specific hybrid class."""

        def __init__(self, **kwargs):
            # Project-specific default parameters can be set here
            super().__init__(**kwargs)
            self.project_name = project_name
            self.model_type = model_type

        def __repr__(self):
            return f"{class_name}(project={self.project_name}, model={self.model_type})"

    ProjectHybridClass.__name__ = class_name
    ProjectHybridClass.__qualname__ = class_name

    return ProjectHybridClass


def get_project_specific_params(project_name: str, model_type: str) -> Dict[str, Any]:
    """
    Get recommended parameters for project-specific hybrid models.

    Args:
        project_name: Project name
        model_type: Model type

    Returns:
        Dictionary of recommended parameters
    """
    params = {}

    if project_name.lower() == "walmart":
        if model_type.upper() == "LSTM":
            params = {"hidden_size": 128, "num_layers": 3, "dropout": 0.2}
        elif model_type.upper() == "TRANSFORMER":
            params = {"d_model": 64, "nhead": 8, "num_layers": 2}

    elif project_name.lower() == "ola" or project_name.lower() == "rideshare":
        if model_type.upper() == "LSTM":
            params = {"hidden_size": 64, "num_layers": 2, "dropout": 0.1}
        elif model_type.upper() == "TRANSFORMER":
            params = {"d_model": 32, "nhead": 4, "num_layers": 2}

    elif project_name.lower() == "inventory":
        if model_type.upper() == "LSTM":
            params = {"hidden_size": 96, "num_layers": 2, "dropout": 0.15}
        elif model_type.upper() == "TRANSFORMER":
            params = {"d_model": 128, "nhead": 8, "num_layers": 4}

    elif project_name.lower() == "tsi" or project_name.lower() == "transportation":
        if model_type.upper() == "LSTM":
            params = {"hidden_size": 64, "num_layers": 2, "dropout": 0.1}
        elif model_type.upper() == "TRANSFORMER":
            params = {"d_model": 64, "nhead": 4, "num_layers": 2}

    return params
