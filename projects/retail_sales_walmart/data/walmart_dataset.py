"""
Walmart-specific dataset implementation.
Inherits from TimeSeriesDataset for domain-specific data loading and preprocessing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import logging

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_portfolio.data.datasets import TimeSeriesDataset

logger = logging.getLogger(__name__)


class WalmartDataset(TimeSeriesDataset):
    """
    Walmart sales forecasting dataset.

    Inherits from TimeSeriesDataset and implements:
    - load_data(): Walmart-specific CSV loading
    - preprocess_data(): Holiday features, store/dept encoding, economic indicators

    Usage:
        # Via Hydra (recommended)
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataset.load()

        # Direct instantiation
        dataset = WalmartDataset(
            data_path="data/raw/Walmart.csv",
            target_column="Weekly_Sales"
        )
        dataset.load()
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        target_column: str = "Weekly_Sales",
        include_store_features: bool = True,
        include_dept_features: bool = True,
        include_holiday_features: bool = True,
        include_economic_features: bool = True,
        **kwargs,
    ):
        """
        Initialize Walmart dataset.

        Args:
            data_path: Path to Walmart.csv (relative to project root)
            target_column: Column to predict (default: Weekly_Sales)
            include_store_features: Whether to include store ID as feature
            include_dept_features: Whether to include department ID as feature
            include_holiday_features: Whether to include holiday flags
            include_economic_features: Whether to include CPI, unemployment, etc.
            **kwargs: Additional arguments passed to TimeSeriesDataset
        """
        # Store Walmart-specific configuration
        self.include_store_features = include_store_features
        self.include_dept_features = include_dept_features
        self.include_holiday_features = include_holiday_features
        self.include_economic_features = include_economic_features

        # Set default parameters for Walmart
        kwargs.setdefault("lookback_window", 52)  # 1 year of weekly data
        kwargs.setdefault("forecast_horizon", 12)  # 12 weeks ahead
        kwargs.setdefault("train_ratio", 0.7)
        kwargs.setdefault("validation_ratio", 0.15)
        kwargs.setdefault("test_ratio", 0.15)

        # Initialize parent class
        super().__init__(data_path=data_path, target_column=target_column, **kwargs)

        logger.info(f"Initialized WalmartDataset with path: {data_path}")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load Walmart sales data from CSV.

        Args:
            data_path: Path to Walmart.csv

        Returns:
            Raw Walmart data as DataFrame
        """
        path = Path(data_path)

        # Handle relative paths from project root
        if not path.is_absolute():
            project_dir = Path(__file__).parent.parent.parent.parent
            path = project_dir / data_path

        # Check if file exists, otherwise generate synthetic data
        if not path.exists():
            logger.warning(f"Walmart data file not found: {path}")
            logger.info("Generating synthetic Walmart-like data for demonstration...")
            return self._generate_synthetic_walmart_data()

        logger.info(f"Loading Walmart data from: {path}")
        df = pd.read_csv(path)

        # Basic validation
        required_cols = ["Weekly_Sales"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in Walmart data: {missing_cols}")

        # Convert date column if exists
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            # Sort by date for time series
            if "Store" in df.columns and "Dept" in df.columns:
                df = df.sort_values(["Store", "Dept", date_cols[0]])
            else:
                df = df.sort_values(date_cols[0])
            df = df.reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walmart-specific preprocessing and feature engineering.

        Args:
            df: Raw Walmart data

        Returns:
            Preprocessed DataFrame with engineered features
        """
        # Call parent preprocessing first (handles missing values, sorting)
        df = super().preprocess_data(df)

        logger.info("Applying Walmart-specific preprocessing...")

        # Add temporal features if date column exists
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df["week_of_year"] = df[date_col].dt.isocalendar().week
            df["month"] = df[date_col].dt.month
            df["quarter"] = df[date_col].dt.quarter
            df["year"] = df[date_col].dt.year
            logger.info("Added temporal features: week_of_year, month, quarter, year")

        # Store features
        if self.include_store_features and "Store" in df.columns:
            # Keep Store as numeric feature (could be enhanced with embeddings)
            df["Store"] = df["Store"].astype(float)
            logger.info("Included Store as feature")
        elif "Store" in df.columns and not self.include_store_features:
            df = df.drop(columns=["Store"])

        # Department features
        if self.include_dept_features and "Dept" in df.columns:
            # Keep Dept as numeric feature
            df["Dept"] = df["Dept"].astype(float)
            logger.info("Included Dept as feature")
        elif "Dept" in df.columns and not self.include_dept_features:
            df = df.drop(columns=["Dept"])

        # Holiday features
        if self.include_holiday_features and "IsHoliday" in df.columns:
            df["IsHoliday"] = df["IsHoliday"].astype(float)
            logger.info("Included IsHoliday as feature")
        elif "IsHoliday" in df.columns and not self.include_holiday_features:
            df = df.drop(columns=["IsHoliday"])

        # Economic features
        if self.include_economic_features:
            economic_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
            found_economic = [col for col in economic_cols if col in df.columns]
            if found_economic:
                logger.info(f"Included economic features: {found_economic}")
                # Normalize economic features
                for col in found_economic:
                    # Simple standardization
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[f"{col}_normalized"] = (df[col] - mean) / std

        # Add lag features for target (useful for forecasting)
        if self.target_column in df.columns:
            df["sales_lag_1"] = df[self.target_column].shift(1)
            df["sales_lag_4"] = df[self.target_column].shift(4)  # 4 weeks ago
            df["sales_lag_52"] = df[self.target_column].shift(52)  # 1 year ago
            logger.info("Added lag features: lag_1, lag_4, lag_52")

        # Fill NaN values created by lagging (from parent's fillna)
        df = df.fillna(method="bfill")

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        logger.info(f"Final columns: {df.columns.tolist()}")

        return df

    def _generate_synthetic_walmart_data(self) -> pd.DataFrame:
        """
        Generate synthetic Walmart-like sales data for demonstration.

        Returns:
            Synthetic DataFrame matching Walmart schema
        """
        logger.info("Generating synthetic Walmart sales data...")

        np.random.seed(42)

        n_weeks = 200  # ~4 years of weekly data
        n_stores = 5
        n_depts = 10

        data = []
        start_date = pd.Timestamp("2020-01-01")

        for store in range(1, n_stores + 1):
            for dept in range(1, n_depts + 1):
                # Base sales varies by store and dept
                base_sales = np.random.uniform(1000, 3000)

                for week in range(n_weeks):
                    date = start_date + pd.Timedelta(weeks=week)

                    # Temporal patterns
                    time_trend = 0.3 * week / n_weeks * base_sales  # Trend
                    yearly_seasonal = 0.2 * base_sales * np.sin(2 * np.pi * week / 52)  # Yearly

                    # Holiday boost (random ~5% of time)
                    is_holiday = int(np.random.random() < 0.05)
                    holiday_boost = 0.4 * base_sales * is_holiday

                    # Noise
                    noise = 0.15 * base_sales * np.random.randn()

                    # Weekly sales
                    weekly_sales = base_sales + time_trend + yearly_seasonal + holiday_boost + noise
                    weekly_sales = max(weekly_sales, 0)  # Non-negative

                    data.append(
                        {
                            "Date": date,
                            "Store": store,
                            "Dept": dept,
                            "Weekly_Sales": weekly_sales,
                            "IsHoliday": is_holiday,
                            "Temperature": np.random.uniform(40, 95),  # Fahrenheit
                            "Fuel_Price": np.random.uniform(2.5, 4.5),  # USD per gallon
                            "CPI": 200 + 15 * np.random.randn(),  # Consumer Price Index
                            "Unemployment": 6 + 2 * np.random.randn(),  # Percentage
                        }
                    )

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic samples")
        return df
