"""
Data validation schemas using Pandera.

This module defines validation schemas for all datasets in the portfolio.
Schemas enforce data types, ranges, and business rules.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Check, DataFrameSchema, Field
from pandera.typing import Series

# =============================================================================
# Walmart Sales Dataset Schema
# =============================================================================


class WalmartSalesSchema(pa.DataFrameModel):
    """
    Validation schema for Walmart sales data.

    Business Rules:
    - 45 stores total (Store 1-45)
    - Weekly sales must be non-negative
    - Temperature in Fahrenheit (-50 to 150)
    - Fuel price must be positive
    - Holiday flag is binary
    """

    Store: Series[int] = Field(ge=1, le=45, description="Store number (1-45)")

    Date: Series[pd.Timestamp] = Field(description="Date of observation")

    Weekly_Sales: Series[float] = Field(ge=0, description="Weekly sales in dollars", nullable=False)

    Temperature: Series[float] = Field(ge=-50, le=150, description="Average temperature in Fahrenheit")

    Fuel_Price: Series[float] = Field(gt=0, le=10.0, description="Fuel price per gallon")

    Holiday_Flag: Series[int] = Field(isin=[0, 1], description="Whether week contains a holiday (0=no, 1=yes)")

    class Config:
        """Schema configuration."""

        strict = False  # Allow extra columns
        coerce = True  # Coerce types if possible


# =============================================================================
# Time Series Features Schema (After Feature Engineering)
# =============================================================================


class TimeSeriesFeaturesSchema(pa.DataFrameModel):
    """
    Validation schema for engineered time series features.

    Validates data after static feature engineering has been applied.
    """

    # Original features
    Store: Series[int] = Field(ge=1, le=45)
    Weekly_Sales: Series[float] = Field(ge=0)

    # Lag features (example - will be extended dynamically)
    Weekly_Sales_lag_1: Series[float] = Field(nullable=True)
    Weekly_Sales_lag_7: Series[float] = Field(nullable=True)

    # Date features
    year: Series[int] = Field(ge=2000, le=2100, nullable=True)
    month: Series[int] = Field(ge=1, le=12, nullable=True)
    dayofweek: Series[int] = Field(ge=0, le=6, nullable=True)

    class Config:
        strict = False  # Allow additional lag/rolling features
        coerce = True


# =============================================================================
# Model Input Schema (Final Features Before Training)
# =============================================================================


def create_model_input_schema(n_features: int, feature_names: Optional[list] = None) -> DataFrameSchema:
    """
    Create a schema for model input features.

    Args:
        n_features: Expected number of features
        feature_names: Optional list of feature names

    Returns:
        DataFrameSchema for validation
    """
    checks = [
        Check.lambda_check(lambda df: df.shape[1] == n_features, error=f"Expected {n_features} features"),
        Check.lambda_check(lambda df: not df.isnull().any().any(), error="No null values allowed in model input"),
        Check.lambda_check(
            lambda df: not df.isin([float("inf"), float("-inf")]).any().any(), error="No infinite values allowed"
        ),
    ]

    if feature_names:
        checks.append(
            Check.lambda_check(
                lambda df: all(col in df.columns for col in feature_names),
                error=f"Missing required features: {feature_names}",
            )
        )

    return DataFrameSchema(checks=checks, coerce=True, strict=False)


# =============================================================================
# Prediction Output Schema
# =============================================================================


class PredictionSchema(pa.DataFrameModel):
    """
    Validation schema for model predictions.

    Ensures predictions are valid before serving.
    """

    predictions: Series[float] = Field(
        description="Point predictions",
        ge=0,
        nullable=False,
    )

    class Config:
        strict = False


# =============================================================================
# API Request Validation
# =============================================================================


class APIRequestSchema(pa.DataFrameModel):
    """
    Validation schema for API request data.

    Validates incoming prediction requests.
    """

    Store: Series[int] = Field(ge=1, le=45)
    Date: Series[pd.Timestamp]

    # Optional features - columns can be missing

    class Config:
        strict = False
        coerce = True
        # Allow missing columns (all non-required columns are optional)
        # Only Store and Date are required


# =============================================================================
# Validation Functions
# =============================================================================


def validate_walmart_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate Walmart sales data.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame

    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    return WalmartSalesSchema.validate(df)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate engineered features.

    Args:
        df: DataFrame with features to validate

    Returns:
        Validated DataFrame

    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    return TimeSeriesFeaturesSchema.validate(df, lazy=True)


def validate_model_input(df: pd.DataFrame, n_features: int, feature_names: Optional[list] = None) -> pd.DataFrame:
    """
    Validate model input data.

    Args:
        df: DataFrame to validate
        n_features: Expected number of features
        feature_names: Optional list of expected feature names

    Returns:
        Validated DataFrame

    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    schema = create_model_input_schema(n_features, feature_names)
    return schema.validate(df)


def validate_predictions(predictions: pd.Series) -> pd.Series:
    """
    Validate model predictions.

    Args:
        predictions: Series of predictions to validate

    Returns:
        Validated predictions

    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    df = pd.DataFrame({"predictions": predictions})
    validated = PredictionSchema.validate(df)
    return validated["predictions"]


def validate_api_request(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate API request data.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame

    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    return APIRequestSchema.validate(df)


# =============================================================================
# Data Quality Report
# =============================================================================


def generate_data_quality_report(df: pd.DataFrame, schema: pa.DataFrameModel) -> dict:
    """
    Generate a data quality report.

    Args:
        df: DataFrame to analyze
        schema: Pandera schema to validate against

    Returns:
        Dictionary with data quality metrics
    """
    try:
        schema.validate(df)
        is_valid = True
        errors = []
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        is_valid = False
        # Handle both single SchemaError and SchemaErrors
        if hasattr(e, "failure_cases"):
            errors = e.failure_cases.to_dict("records")
        else:
            errors = [{"error": str(e)}]

    return {
        "is_valid": is_valid,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Validate Walmart data
    sample_data = pd.DataFrame(
        {
            "Store": [1, 2, 3],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "Weekly_Sales": [24000.50, 30000.25, 28000.00],
            "Temperature": [45.5, 50.2, 48.1],
            "Fuel_Price": [2.5, 2.6, 2.55],
            "Holiday_Flag": [0, 1, 0],
        }
    )

    try:
        validated_data = validate_walmart_data(sample_data)
        print("✅ Data validation passed!")
        print(f"Validated {len(validated_data)} rows")
    except pa.errors.SchemaError as e:
        print("❌ Data validation failed!")
        print(e)
