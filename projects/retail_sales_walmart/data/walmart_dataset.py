"""
Walmart-specific dataset implementation following the project inheritance pattern.

This demonstrates how to create project-specific dataset classes that inherit from
the base classes in src/ml_portfolio/data/datasets.py.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# Add src to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_portfolio.data.datasets import TimeSeriesDataset, MultiSeriesDataset


class WalmartTimeSeriesDataset(TimeSeriesDataset):
    """
    Walmart-specific time series dataset with retail domain preprocessing.

    Inherits from TimeSeriesDataset and adds Walmart-specific features:
    - Holiday effects
    - Temperature impact
    - Fuel price correlation
    - Economic indicators (CPI, unemployment)
    - Store-specific characteristics
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        store_id: Optional[int] = None,
        aggregate_stores: bool = True,
        include_economic_features: bool = True,
        include_weather_features: bool = True,
        **kwargs,
    ):
        """
        Initialize WalmartTimeSeriesDataset.

        Args:
            data_path: Path to Walmart.csv file (auto-detected if None)
            store_id: Specific store to focus on (None for all stores)
            aggregate_stores: Whether to sum across all stores
            include_economic_features: Include CPI, unemployment
            include_weather_features: Include temperature, fuel price
            **kwargs: Additional arguments for parent TimeSeriesDataset
        """
        self.store_id = store_id
        self.aggregate_stores = aggregate_stores
        self.include_economic_features = include_economic_features
        self.include_weather_features = include_weather_features

        # Load Walmart data
        if data_path is None:
            # Auto-detect data path relative to this file
            project_dir = Path(__file__).parent.parent
            data_path = project_dir / "data" / "raw" / "Walmart.csv"
        else:
            # Convert relative paths to be relative to this project directory
            data_path = Path(data_path)
            if not data_path.is_absolute():
                project_dir = Path(__file__).parent.parent
                data_path = project_dir / data_path

        # Load and preprocess Walmart data
        raw_data = self._load_walmart_data(data_path)
        processed_data = self._preprocess_walmart_data(raw_data)

        # Set default parameters for Walmart forecasting
        kwargs.setdefault("target_column", "Weekly_Sales")
        kwargs.setdefault("lookback_window", 52)  # 1 year of weekly data
        kwargs.setdefault("forecast_horizon", 4)  # 4 weeks ahead

        # Initialize parent class
        super().__init__(processed_data, **kwargs)

    def _load_walmart_data(self, data_path: str) -> pd.DataFrame:
        """Load raw Walmart dataset."""
        df = pd.read_csv(data_path)

        # Parse date column
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

        # Sort by date and store
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        print(f"Loaded Walmart dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Stores: {df['Store'].nunique()} unique stores")

        return df

    def _preprocess_walmart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Walmart-specific preprocessing."""
        df = df.copy()

        # Store selection logic
        if self.store_id is not None:
            df = df[df["Store"] == self.store_id].copy()
            print(f"Filtered to Store {self.store_id}: {len(df)} records")
        elif self.aggregate_stores:
            # Aggregate across all stores by date
            agg_dict = {"Weekly_Sales": "sum"}

            # Include additional features in aggregation
            if self.include_economic_features:
                agg_dict.update({"CPI": "mean", "Unemployment": "mean"})

            if self.include_weather_features:
                agg_dict.update({"Temperature": "mean", "Fuel_Price": "mean"})

            # Always include holiday flag (max ensures if any store has holiday, it's marked)
            agg_dict["Holiday_Flag"] = "max"

            df = df.groupby("Date").agg(agg_dict).reset_index()
            print(f"Aggregated to total sales: {len(df)} time periods")

        # Add Walmart-specific time features
        df = self._add_walmart_time_features(df)

        # Add economic impact features
        if self.include_economic_features:
            df = self._add_economic_features(df)

        # Add weather/fuel impact features
        if self.include_weather_features:
            df = self._add_weather_features(df)

        return df

    def _add_walmart_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add retail-specific time features."""
        df = df.copy()

        # Basic time features
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["week_of_year"] = df["Date"].dt.isocalendar().week
        df["quarter"] = df["Date"].dt.quarter

        # Retail calendar features
        df["is_holiday"] = df["Holiday_Flag"].astype(bool)

        # Seasonal shopping patterns
        df["is_back_to_school"] = ((df["month"] == 8) | (df["month"] == 9)).astype(int)
        df["is_holiday_season"] = ((df["month"] == 11) | (df["month"] == 12)).astype(int)
        df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)

        # Week patterns
        df["is_month_end"] = (df["Date"].dt.day >= 25).astype(int)
        df["days_to_month_end"] = df["Date"].dt.days_in_month - df["Date"].dt.day

        return df

    def _add_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic impact features."""
        if not self.include_economic_features:
            return df

        df = df.copy()

        # Economic trends
        df["cpi_change"] = df["CPI"].pct_change(periods=4)  # Quarterly change
        df["unemployment_change"] = df["Unemployment"].pct_change(periods=4)

        # Economic conditions
        df["high_unemployment"] = (df["Unemployment"] > df["Unemployment"].median()).astype(int)
        df["inflation_period"] = (df["cpi_change"] > 0.02).astype(int)  # >2% quarterly inflation

        return df

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather and fuel impact features."""
        if not self.include_weather_features:
            return df

        df = df.copy()

        # Temperature impact on shopping
        df["temp_comfortable"] = ((df["Temperature"] >= 60) & (df["Temperature"] <= 80)).astype(int)
        df["temp_extreme"] = ((df["Temperature"] < 32) | (df["Temperature"] > 90)).astype(int)

        # Fuel price impact
        df["fuel_price_change"] = df["Fuel_Price"].pct_change(periods=4)
        df["high_fuel_prices"] = (df["Fuel_Price"] > df["Fuel_Price"].median()).astype(int)

        return df

    def get_walmart_insights(self) -> Dict[str, Any]:
        """Get Walmart-specific dataset insights."""
        insights = self.get_data_info()

        # Add Walmart-specific insights
        if hasattr(self, "processed_data"):
            df = pd.DataFrame(self.processed_data)

            walmart_insights = {
                "sales_statistics": {
                    "mean_weekly_sales": df.iloc[:, 0].mean() if len(df.columns) > 0 else None,
                    "sales_volatility": df.iloc[:, 0].std() if len(df.columns) > 0 else None,
                },
                "seasonal_patterns": {
                    "holiday_effect": "Available" if "is_holiday" in df.columns else "Not available",
                    "seasonal_features": "Available" if "is_holiday_season" in df.columns else "Not available",
                },
                "economic_features": {
                    "cpi_data": self.include_economic_features,
                    "unemployment_data": self.include_economic_features,
                },
                "weather_features": {
                    "temperature_data": self.include_weather_features,
                    "fuel_price_data": self.include_weather_features,
                },
            }

            insights.update(walmart_insights)

        return insights


class WalmartMultiStoreDataset(MultiSeriesDataset):
    """
    Walmart multi-store dataset for forecasting across different stores.

    Inherits from MultiSeriesDataset to handle multiple store time series.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        store_list: Optional[List[int]] = None,
        min_data_points: int = 100,
        **kwargs,
    ):
        """
        Initialize WalmartMultiStoreDataset.

        Args:
            data_path: Path to Walmart.csv file
            store_list: List of specific stores to include (None for all)
            min_data_points: Minimum data points required per store
            **kwargs: Additional arguments for parent MultiSeriesDataset
        """
        self.store_list = store_list
        self.min_data_points = min_data_points

        # Load data
        if data_path is None:
            project_dir = Path(__file__).parent.parent
            data_path = project_dir / "data" / "raw" / "Walmart.csv"

        # Load and preprocess
        raw_data = self._load_multi_store_data(data_path)

        # Set default parameters
        kwargs.setdefault("target_column", "Weekly_Sales")
        kwargs.setdefault("lookback_window", 26)  # 6 months for individual stores
        kwargs.setdefault("forecast_horizon", 2)  # 2 weeks ahead

        # Initialize parent class
        super().__init__(raw_data, series_column="Store", **kwargs)

    def _load_multi_store_data(self, data_path: str) -> pd.DataFrame:
        """Load and filter multi-store data."""
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        # Filter stores with sufficient data
        store_counts = df["Store"].value_counts()
        valid_stores = store_counts[store_counts >= self.min_data_points].index
        df = df[df["Store"].isin(valid_stores)].copy()

        # Filter to specific stores if requested
        if self.store_list:
            df = df[df["Store"].isin(self.store_list)].copy()

        print(f"Multi-store dataset: {len(valid_stores)} stores, {len(df)} total records")

        return df


# Factory function following our pattern
def create_walmart_dataset(dataset_type: str = "single", **kwargs):
    """
    Factory function to create Walmart datasets following naming conventions.

    Args:
        dataset_type: "single" for aggregated, "multi" for multi-store
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Walmart dataset instance
    """
    if dataset_type.lower() in ["single", "aggregated"]:
        return WalmartTimeSeriesDataset(**kwargs)
    elif dataset_type.lower() in ["multi", "multistore", "stores"]:
        return WalmartMultiStoreDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# Usage examples
if __name__ == "__main__":
    print("🏪 Walmart Dataset Examples")
    print("=" * 40)

    # Example 1: Aggregated dataset
    print("\n1. Creating aggregated Walmart dataset...")
    walmart_agg = WalmartTimeSeriesDataset(
        aggregate_stores=True, include_economic_features=True, include_weather_features=True
    )

    print(f"Dataset created: {len(walmart_agg)} sequences")
    print("Insights:", walmart_agg.get_walmart_insights())

    # Example 2: Single store dataset
    print("\n2. Creating single store dataset...")
    walmart_store1 = WalmartTimeSeriesDataset(store_id=1, aggregate_stores=False)

    print(f"Store 1 dataset: {len(walmart_store1)} sequences")

    # Example 3: Multi-store dataset
    print("\n3. Creating multi-store dataset...")
    walmart_multi = WalmartMultiStoreDataset(store_list=[1, 2, 3, 4, 5], min_data_points=100)  # Top 5 stores

    print(f"Multi-store dataset: {len(walmart_multi)} sequences")

    print("\n✅ All Walmart datasets created successfully!")
