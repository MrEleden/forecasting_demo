"""
Data transformation utilities for time series preprocessing.

This module provides scalers, encoders, and feature engineering pipelines
optimized for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class TimeSeriesScaler:
    """
    Wrapper for scikit-learn scalers with time series specific handling.
    """

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize TimeSeriesScaler.

        Args:
            scaler_type: Type of scaler ("standard", "minmax", "robust")
        """
        self.scaler_type = scaler_type

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def fit(self, data: np.ndarray) -> "TimeSeriesScaler":
        """
        Fit the scaler to training data.

        Args:
            data: Training data to fit scaler

        Returns:
            Self for method chaining
        """
        self.scaler.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data in one step.

        Args:
            data: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.

        Args:
            data: Scaled data to inverse transform

        Returns:
            Data in original scale
        """
        return self.scaler.inverse_transform(data)


class CategoricalEncoder:
    """
    Encoder for categorical variables in time series data.
    """

    def __init__(self, encoding_type: str = "onehot"):
        """
        Initialize CategoricalEncoder.

        Args:
            encoding_type: Type of encoding ("onehot", "label", "target")
        """
        self.encoding_type = encoding_type
        self.encoders = {}

    def fit(self, data: pd.DataFrame, categorical_columns: list) -> "CategoricalEncoder":
        """
        Fit encoders to categorical columns.

        Args:
            data: DataFrame containing categorical data
            categorical_columns: List of categorical column names

        Returns:
            Self for method chaining
        """
        for col in categorical_columns:
            if self.encoding_type == "onehot":
                # Store unique values for one-hot encoding
                self.encoders[col] = data[col].unique()
            elif self.encoding_type == "label":
                # Store label mapping
                unique_vals = data[col].unique()
                self.encoders[col] = {val: i for i, val in enumerate(unique_vals)}

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using fitted encoders.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        result = data.copy()

        for col, encoder in self.encoders.items():
            if col in result.columns:
                if self.encoding_type == "onehot":
                    # Create one-hot encoded columns
                    for val in encoder:
                        result[f"{col}_{val}"] = (result[col] == val).astype(int)
                    result = result.drop(columns=[col])
                elif self.encoding_type == "label":
                    # Apply label encoding
                    result[col] = result[col].map(encoder).fillna(-1)

        return result


def create_feature_pipeline(
    numerical_scaler: str = "standard", categorical_encoding: str = "onehot"
) -> Tuple[TimeSeriesScaler, CategoricalEncoder]:
    """
    Create a feature preprocessing pipeline.

    Args:
        numerical_scaler: Type of numerical scaler
        categorical_encoding: Type of categorical encoding

    Returns:
        Tuple of (scaler, encoder)
    """
    scaler = TimeSeriesScaler(numerical_scaler)
    encoder = CategoricalEncoder(categorical_encoding)

    return scaler, encoder
