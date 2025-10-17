"""
Unit tests for training utilities and helper functions.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from ml_portfolio.training.train import set_seed, setup_logging

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSetSeed:
    """Test seed setting functionality."""

    def test_set_seed_numpy(self):
        """Test that seed setting affects NumPy random generation."""
        set_seed(42)
        a = np.random.rand(5)

        set_seed(42)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_set_seed_different_values(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        a = np.random.rand(5)

        set_seed(123)
        b = np.random.rand(5)

        assert not np.array_equal(a, b)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_set_seed_torch(self):
        """Test that seed setting affects PyTorch random generation."""
        set_seed(42)
        a = torch.rand(5)

        set_seed(42)
        b = torch.rand(5)

        assert torch.allclose(a, b)

    def test_set_seed_reproducibility(self):
        """Test reproducibility across multiple operations."""
        set_seed(42)
        results1 = [np.random.rand() for _ in range(10)]

        set_seed(42)
        results2 = [np.random.rand() for _ in range(10)]

        np.testing.assert_array_almost_equal(results1, results2)


class TestSetupLogging:
    """Test logging setup functionality."""

    def test_setup_logging_default_level(self):
        """Test setup with default INFO level."""
        cfg = DictConfig({"log_level": "INFO", "model": {"name": "test"}, "seed": 42})

        logger = setup_logging(cfg)

        assert logger is not None
        assert logger.level == logging.INFO

    def test_setup_logging_debug_level(self):
        """Test setup with DEBUG level."""
        cfg = DictConfig({"log_level": "DEBUG", "model": {"name": "test"}})

        logger = setup_logging(cfg)

        assert logger.level == logging.DEBUG

    def test_setup_logging_warning_level(self):
        """Test setup with WARNING level."""
        cfg = DictConfig({"log_level": "WARNING", "model": {"name": "test"}})

        logger = setup_logging(cfg)

        assert logger.level == logging.WARNING

    def test_setup_logging_error_level(self):
        """Test setup with ERROR level."""
        cfg = DictConfig({"log_level": "ERROR", "model": {"name": "test"}})

        logger = setup_logging(cfg)

        assert logger.level == logging.ERROR

    def test_setup_logging_missing_level(self):
        """Test setup without log_level defaults to INFO."""
        cfg = DictConfig({"model": {"name": "test"}})

        logger = setup_logging(cfg)

        assert logger.level == logging.INFO

    def test_setup_logging_case_insensitive(self):
        """Test that log level is case insensitive."""
        cfg = DictConfig({"log_level": "debug", "model": {"name": "test"}})  # lowercase

        logger = setup_logging(cfg)

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_complex_config(self):
        """Test logging setup with complex nested config."""
        cfg = DictConfig(
            {
                "log_level": "INFO",
                "model": {"name": "lstm", "hidden_size": 128, "num_layers": 2},
                "optimizer": {"name": "adam", "lr": 0.001},
                "dataset": {"name": "walmart", "batch_size": 32},
            }
        )

        logger = setup_logging(cfg)

        assert logger is not None
        assert logger.level == logging.INFO


class TestConfigValidation:
    """Test configuration validation utilities."""

    def test_omegaconf_dict_creation(self):
        """Test OmegaConf DictConfig creation."""
        config_dict = {"model": "lstm", "batch_size": 32, "learning_rate": 0.001}

        cfg = OmegaConf.create(config_dict)

        assert cfg.model == "lstm"
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 0.001

    def test_omegaconf_nested_access(self):
        """Test accessing nested config values."""
        cfg = OmegaConf.create({"model": {"type": "lstm", "params": {"hidden_size": 128}}})

        assert cfg.model.type == "lstm"
        assert cfg.model.params.hidden_size == 128

    def test_omegaconf_to_yaml(self):
        """Test config serialization to YAML."""
        cfg = OmegaConf.create({"model": "lstm", "batch_size": 32})

        yaml_str = OmegaConf.to_yaml(cfg)

        assert "model: lstm" in yaml_str
        assert "batch_size: 32" in yaml_str

    def test_omegaconf_merge(self):
        """Test merging multiple configs."""
        base_cfg = OmegaConf.create({"a": 1, "b": 2})
        override_cfg = OmegaConf.create({"b": 3, "c": 4})

        merged = OmegaConf.merge(base_cfg, override_cfg)

        assert merged.a == 1
        assert merged.b == 3  # Overridden
        assert merged.c == 4


class TestTrainingUtilities:
    """Test training utility functions."""

    def test_seed_consistency_with_random(self):
        """Test seed affects Python's random module."""
        import random

        set_seed(42)
        a = random.random()

        set_seed(42)
        b = random.random()

        assert a == b

    def test_multiple_seed_calls(self):
        """Test multiple seed setting calls."""
        set_seed(1)
        a = np.random.rand()

        set_seed(2)
        b = np.random.rand()

        set_seed(1)
        c = np.random.rand()

        assert a == c
        assert a != b

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_deterministic_mode(self):
        """Test that torch deterministic mode is enabled."""
        set_seed(42)

        # After set_seed, torch should be in deterministic mode
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestPathUtilities:
    """Test path handling utilities."""

    def test_path_creation(self):
        """Test creating Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "models" / "checkpoints"
            path.mkdir(parents=True, exist_ok=True)

            assert path.exists()
            assert path.is_dir()

    def test_path_joining(self):
        """Test path joining operations."""
        base = Path("/tmp")
        sub = base / "models" / "lstm"

        # Use os-specific path separator for cross-platform compatibility
        import os

        expected = f"{os.sep}tmp{os.sep}models{os.sep}lstm"
        assert str(sub) == expected

    def test_path_exists_check(self):
        """Test path existence checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_path = Path(tmpdir)
            non_existing_path = Path(tmpdir) / "does_not_exist"

            assert existing_path.exists()
            assert not non_existing_path.exists()
