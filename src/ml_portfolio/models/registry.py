"""
Model registry for loading and comparing models.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


class ModelRegistry:
    """
    Registry for managing trained models and their metadata.
    """

    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Initialize ModelRegistry.

        Args:
            registry_path: Path to the registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        """Load the model registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self):
        """Save the model registry to file."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model(self, name: str, model_path: str, metadata: Dict[str, Any]):
        """
        Register a model in the registry.

        Args:
            name: Model name
            model_path: Path to the saved model
            metadata: Model metadata (metrics, hyperparameters, etc.)
        """
        self.registry[name] = {"model_path": str(model_path), "metadata": metadata}
        self._save_registry()

    def load_model(self, name: str):
        """
        Load a model from the registry.

        Args:
            name: Model name

        Returns:
            Loaded model
        """
        if name not in self.registry:
            raise ValueError(f"Model '{name}' not found in registry")

        model_path = Path(self.registry[name]["model_path"])

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a model.

        Args:
            name: Model name

        Returns:
            Model metadata
        """
        if name not in self.registry:
            raise ValueError(f"Model '{name}' not found in registry")

        return self.registry[name]["metadata"]

    def list_models(self) -> list:
        """
        List all registered models.

        Returns:
            List of model names
        """
        return list(self.registry.keys())

    def remove_model(self, name: str):
        """
        Remove a model from the registry.

        Args:
            name: Model name
        """
        if name in self.registry:
            del self.registry[name]
            self._save_registry()


# Global registry instance
_global_registry = None


def get_registry(registry_path: Optional[str] = None) -> ModelRegistry:
    """
    Get the global model registry instance.

    Args:
        registry_path: Path to registry file (optional)

    Returns:
        ModelRegistry instance
    """
    global _global_registry

    if _global_registry is None or registry_path is not None:
        path = registry_path or "models/registry.json"
        _global_registry = ModelRegistry(path)

    return _global_registry


def register_model(name: str, model_path: str, metadata: Dict[str, Any]):
    """
    Register a model using the global registry.

    Args:
        name: Model name
        model_path: Path to saved model
        metadata: Model metadata
    """
    registry = get_registry()
    registry.register_model(name, model_path, metadata)


def load_model(name: str):
    """
    Load a model using the global registry.

    Args:
        name: Model name

    Returns:
        Loaded model
    """
    registry = get_registry()
    return registry.load_model(name)
