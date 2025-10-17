"""
Time Series Dataset classes.

Pure data containers with no preprocessing logic.
"""

import numpy as np

# Optional PyTorch support
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TimeSeriesDataset:
    """
    Pure data container for time series - no preprocessing logic.

    This class stores data in numpy format and provides basic access methods.
    Compatible with both sklearn-style and PyTorch-style training.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray = None,
        feature_names: list = None,
        metadata: dict = None,
    ):
        """
        Initialize dataset.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            timestamps: Optional timestamp array
            feature_names: List of feature names
            metadata: Dictionary of metadata
        """
        self.X = X
        self.y = y
        self.timestamps = timestamps
        self.feature_names = feature_names or []
        self.metadata = metadata or {}

    def __len__(self):
        """Return number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get a single sample."""
        return self.X[idx], self.y[idx]

    def get_feature_dim(self):
        """Get feature dimensionality."""
        return self.X.shape[1] if self.X.ndim > 1 else 1

    def get_data(self):
        """Get full X, y arrays."""
        return self.X, self.y
