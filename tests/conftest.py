"""
Pytest configuration and shared fixtures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_timeseries_data():
    """Generate sample time series data for testing."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    data = pd.DataFrame(
        {
            "Date": dates,
            "Store": np.random.choice([1, 2, 3], size=100),
            "Weekly_Sales": np.random.uniform(1000, 10000, size=100),
            "Temperature": np.random.uniform(50, 90, size=100),
            "Fuel_Price": np.random.uniform(2.5, 4.0, size=100),
            "Holiday_Flag": np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
        }
    )
    return data


@pytest.fixture
def sample_arrays():
    """Generate sample numpy arrays for metric testing."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    return y_true, y_pred


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
