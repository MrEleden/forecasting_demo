"""
Configuration management utilities using Hydra and OmegaConf.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import warnings
import yaml

try:
    from omegaconf import OmegaConf, DictConfig

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    # Create a type alias for fallback
    from typing import Dict as DictConfig

try:
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""

    name: str = "default"
    batch_size: int = 32
    sequence_length: int = 24
    prediction_length: int = 1
    validation_split: float = 0.2
    test_split: float = 0.2
    normalize: bool = True
    add_time_features: bool = True


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    name: str = "lstm"
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    output_size: int = 1


@dataclass
class OptimizerConfig:
    """Configuration for optimizer parameters."""

    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    name: str = "none"
    step_size: int = 30
    gamma: float = 0.1
    patience: int = 10
    factor: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    epochs: int = 100
    early_stopping_patience: int = 10
    checkpoint_every: int = 10
    device: str = "auto"
    seed: int = 42
    gradient_clip_val: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs"
    experiment_name: str = "default"


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if OMEGACONF_AVAILABLE:
        try:
            return OmegaConf.load(config_path)
        except Exception as e:
            warnings.warn(f"Failed to load config with OmegaConf: {e}")

    # Fallback to basic YAML loading
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
        OmegaConf.save(config, output_path)
    else:
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    if OMEGACONF_AVAILABLE:
        merged = OmegaConf.create({})
        for config in configs:
            merged = OmegaConf.merge(merged, config)
        return merged
    else:
        # Simple dictionary merge for fallback
        merged = {}
        for config in configs:
            if isinstance(config, dict):
                merged.update(config)
        return merged


def create_default_config() -> ExperimentConfig:
    """
    Create default experiment configuration.

    Returns:
        Default experiment configuration
    """
    return ExperimentConfig()


def config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert configuration to dictionary.

    Args:
        config: Configuration object

    Returns:
        Dictionary representation
    """
    if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, "__dict__"):
        return config.__dict__
    elif isinstance(config, dict):
        return config
    else:
        return {}


def dict_to_config(data: Dict[str, Any]) -> DictConfig:
    """
    Convert dictionary to configuration.

    Args:
        data: Dictionary data

    Returns:
        Configuration object
    """
    if OMEGACONF_AVAILABLE:
        return OmegaConf.create(data)
    else:
        return data


def validate_config(config: DictConfig, required_keys: List[str]) -> bool:
    """
    Validate that configuration contains required keys.

    Args:
        config: Configuration to validate
        required_keys: List of required keys

    Returns:
        True if valid, raises ValueError if not
    """
    missing_keys = []

    for key in required_keys:
        if "." in key:
            # Handle nested keys
            keys = key.split(".")
            current = config
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    missing_keys.append(key)
                    break
        else:
            if key not in config:
                missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    return True


def setup_hydra_config(config_dir: Union[str, Path], config_name: str = "config") -> DictConfig:
    """
    Setup Hydra configuration.

    Args:
        config_dir: Directory containing config files
        config_name: Name of main config file

    Returns:
        Hydra configuration
    """
    if not HYDRA_AVAILABLE:
        warnings.warn("Hydra not available, using basic config loading")
        config_path = Path(config_dir) / f"{config_name}.yaml"
        return load_config(config_path)

    config_dir = Path(config_dir).absolute()

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(str(config_dir), version_base=None):
            cfg = compose(config_name=config_name)
        return cfg
    except Exception as e:
        warnings.warn(f"Failed to setup Hydra config: {e}")
        # Fallback to basic loading
        config_path = config_dir / f"{config_name}.yaml"
        return load_config(config_path)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get configuration value with dot notation support.

    Args:
        config: Configuration dictionary
        key: Configuration key (supports dot notation)
        default: Default value if key not found

    Returns:
        Configuration value
    """
    if "." not in key:
        return config.get(key, default)

    keys = key.split(".")
    current = config

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current


def set_config_value(config: DictConfig, key: str, value: Any) -> None:
    """
    Set configuration value with dot notation support.

    Args:
        config: Configuration dictionary
        key: Configuration key (supports dot notation)
        value: Value to set
    """
    if "." not in key:
        config[key] = value
        return

    keys = key.split(".")
    current = config

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value


def create_config_from_args(args: Dict[str, Any]) -> ExperimentConfig:
    """
    Create configuration from command line arguments or dictionary.

    Args:
        args: Arguments dictionary

    Returns:
        Experiment configuration
    """
    config = create_default_config()

    # Map common argument names to config structure
    arg_mapping = {
        "batch_size": "dataset.batch_size",
        "learning_rate": "optimizer.lr",
        "lr": "optimizer.lr",
        "epochs": "training.epochs",
        "hidden_size": "model.hidden_size",
        "num_layers": "model.num_layers",
        "dropout": "model.dropout",
        "seed": "training.seed",
        "device": "training.device",
    }

    # Convert to dictionary for easier manipulation
    config_dict = config_to_dict(config)

    for arg_key, arg_value in args.items():
        if arg_key in arg_mapping:
            config_key = arg_mapping[arg_key]
            set_config_value(config_dict, config_key, arg_value)
        else:
            # Try to set directly if key exists in config
            try:
                set_config_value(config_dict, arg_key, arg_value)
            except:
                warnings.warn(f"Unknown configuration argument: {arg_key}")

    return dict_to_config(config_dict)


class ConfigManager:
    """
    Configuration manager for experiment tracking and parameter management.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self.config = None

    def load(self, config_name: str = "config") -> DictConfig:
        """
        Load configuration.

        Args:
            config_name: Name of configuration file

        Returns:
            Loaded configuration
        """
        if self.config_dir:
            self.config = setup_hydra_config(self.config_dir, config_name)
        else:
            self.config = dict_to_config(config_to_dict(create_default_config()))

        return self.config

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration.

        Args:
            output_path: Output file path
        """
        if self.config is not None:
            save_config(self.config, output_path)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates
        """
        if self.config is None:
            self.config = dict_to_config({})

        for key, value in updates.items():
            set_config_value(self.config, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Configuration value
        """
        if self.config is None:
            return default

        return get_config_value(self.config, key, default)
