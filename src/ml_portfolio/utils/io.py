"""
I/O utilities for data loading, saving, and caching.
"""

import os
import pickle
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(
    filepath: Union[str, Path], date_column: Optional[str] = None, date_format: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """
    Load CSV file with optional date parsing.

    Args:
        filepath: Path to CSV file
        date_column: Column to parse as date
        date_format: Date format string
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame
    """
    # Set default arguments
    default_kwargs = {"parse_dates": [date_column] if date_column else None, "date_format": date_format}
    default_kwargs.update(kwargs)

    try:
        df = pd.read_csv(filepath, **default_kwargs)

        # Set date column as index if specified
        if date_column and date_column in df.columns:
            df = df.set_index(date_column)
            df.index.name = "date"

        return df
    except Exception as e:
        raise IOError(f"Error loading CSV file {filepath}: {e}")


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], create_dir: bool = True, **kwargs) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        filepath: Output file path
        create_dir: Whether to create directory if it doesn't exist
        **kwargs: Additional arguments for pd.to_csv
    """
    filepath = Path(filepath)

    if create_dir:
        ensure_dir(filepath.parent)

    try:
        df.to_csv(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"Error saving CSV file {filepath}: {e}")


def load_parquet(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load parquet file.

    Args:
        filepath: Path to parquet file
        **kwargs: Additional arguments for pd.read_parquet

    Returns:
        DataFrame
    """
    try:
        return pd.read_parquet(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"Error loading parquet file {filepath}: {e}")


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path], create_dir: bool = True, **kwargs) -> None:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save
        filepath: Output file path
        create_dir: Whether to create directory if it doesn't exist
        **kwargs: Additional arguments for pd.to_parquet
    """
    filepath = Path(filepath)

    if create_dir:
        ensure_dir(filepath.parent)

    try:
        df.to_parquet(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"Error saving parquet file {filepath}: {e}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise IOError(f"Error loading pickle file {filepath}: {e}")


def save_pickle(obj: Any, filepath: Union[str, Path], create_dir: bool = True) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Output file path
        create_dir: Whether to create directory if it doesn't exist
    """
    filepath = Path(filepath)

    if create_dir:
        ensure_dir(filepath.parent)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise IOError(f"Error saving pickle file {filepath}: {e}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Error loading JSON file {filepath}: {e}")


def save_json(data: Dict[str, Any], filepath: Union[str, Path], create_dir: bool = True, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
        create_dir: Whether to create directory if it doesn't exist
        indent: JSON indentation
    """
    filepath = Path(filepath)

    if create_dir:
        ensure_dir(filepath.parent)

    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=str)
    except Exception as e:
        raise IOError(f"Error saving JSON file {filepath}: {e}")


class DataCache:
    """
    Simple data caching utility.
    """

    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        """
        Initialize DataCache.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = ensure_dir(cache_dir)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached data.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                return load_pickle(cache_path)
            except Exception:
                warnings.warn(f"Failed to load cache for key: {key}")
        return None

    def set(self, key: str, data: Any) -> None:
        """
        Set cached data.

        Args:
            key: Cache key
            data: Data to cache
        """
        cache_path = self._get_cache_path(key)
        try:
            save_pickle(data, cache_path)
        except Exception as e:
            warnings.warn(f"Failed to save cache for key {key}: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if cache exists for key.

        Args:
            key: Cache key

        Returns:
            True if cache exists
        """
        return self._get_cache_path(key).exists()

    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            key: Specific key to clear (None for all)
        """
        if key is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        else:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_data_path(project_name: str, data_type: str = "processed") -> Path:
    """
    Get data path for a specific project.

    Args:
        project_name: Name of the project
        data_type: Type of data (raw, interim, processed, external)

    Returns:
        Path to data directory
    """
    project_root = get_project_root()
    return project_root / "projects" / project_name / "data" / data_type


def list_data_files(project_name: str, data_type: str = "processed", extension: str = ".csv") -> List[Path]:
    """
    List data files in a project directory.

    Args:
        project_name: Name of the project
        data_type: Type of data directory
        extension: File extension to filter by

    Returns:
        List of file paths
    """
    data_dir = get_data_path(project_name, data_type)
    if not data_dir.exists():
        return []

    return list(data_dir.glob(f"*{extension}"))


def create_file_hash(filepath: Union[str, Path]) -> str:
    """
    Create hash of file contents for cache invalidation.

    Args:
        filepath: Path to file

    Returns:
        Hash string
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return ""

    hash_obj = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


class CachedDataLoader:
    """
    Data loader with automatic caching based on file modification time.
    """

    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        """
        Initialize CachedDataLoader.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache = DataCache(cache_dir)

    def load_data(
        self,
        filepath: Union[str, Path],
        loader_func: callable,
        cache_key: Optional[str] = None,
        force_reload: bool = False,
        **loader_kwargs,
    ) -> Any:
        """
        Load data with caching.

        Args:
            filepath: Path to data file
            loader_func: Function to load data
            cache_key: Custom cache key (uses filepath hash if None)
            force_reload: Force reload even if cached
            **loader_kwargs: Arguments for loader function

        Returns:
            Loaded data
        """
        filepath = Path(filepath)

        if cache_key is None:
            file_hash = create_file_hash(filepath)
            cache_key = f"{filepath.stem}_{file_hash}"

        # Check cache
        if not force_reload and self.cache.exists(cache_key):
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data

        # Load data
        data = loader_func(filepath, **loader_kwargs)

        # Cache data
        self.cache.set(cache_key, data)

        return data
