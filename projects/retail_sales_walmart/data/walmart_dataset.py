"""
Walmart-specific dataset factory implementation.
Creates train/val/test TimeSeriesDataset splits with Walmart-specific preprocessing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_portfolio.data.datasets import TimeSeriesDataset, DatasetFactory

logger = logging.getLogger(__name__)


class WalmartFactory(DatasetFactory):
    """
    Walmart sales forecasting dataset factory.

    Creates train/val/test TimeSeriesDataset splits with Walmart-specific data loading
    and preprocessing logic.

    Usage:
        # Via Hydra (recommended)
        factory = hydra.utils.instantiate(cfg.dataset)
        train_dataset, val_dataset, test_dataset = factory.create_datasets()

        # Direct instantiation
        factory = WalmartFactory(
            data_path="data/raw/Walmart.csv",
            target_column="Weekly_Sales"
        )
        train_dataset, val_dataset, test_dataset = factory.create_datasets()
    """

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
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        include_store_features: bool = True,
        include_dept_features: bool = True,
        include_holiday_features: bool = True,
        include_economic_features: bool = True,
        dataset_class: str = "ml_portfolio.data.datasets.TimeSeriesDataset",
        **kwargs,
    ):
        """
        Initialize Walmart dataset factory.

        Args:
            data_path: Path to Walmart.csv (relative to project root)
            target_column: Column to predict (default: Weekly_Sales)
            train_ratio: Training split ratio
            validation_ratio: Validation split ratio
            test_ratio: Test split ratio
            include_store_features: Whether to include store ID as feature
            include_dept_features: Whether to include department ID as feature
            include_holiday_features: Whether to include holiday flags
            include_economic_features: Whether to include CPI, unemployment, etc.
            dataset_class: The base dataset class to instantiate
            **kwargs: Additional arguments passed to TimeSeriesDataset instances
        """
        # Store Walmart-specific configuration
        self.include_store_features = include_store_features
        self.include_dept_features = include_dept_features
        self.include_holiday_features = include_holiday_features
        self.include_economic_features = include_economic_features

        # Initialize parent DatasetFactory
        super().__init__(
            dataset_class=dataset_class,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            data_path=data_path,
            target_column=target_column,
            **kwargs,
        )
        self.kwargs = kwargs

        # Validate ratios
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + validation_ratio + test_ratio}")

        logger.info(f"Initialized WalmartFactory with path: {data_path}")

    def _extract_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Extract feature matrix X and target vector y from preprocessed data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Identify target column
        target_column = self.dataset_kwargs.get("target_column", "Weekly_Sales")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        y = df[target_column].values

        # Use all columns except target and non-numeric columns
        exclude_cols = [target_column]
        # Exclude date/time columns
        exclude_cols.extend([col for col in df.columns if "date" in col.lower() or "time" in col.lower()])

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Only keep numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            # If no features, create time index
            X = np.arange(len(df)).reshape(-1, 1)
            feature_names = ["time_index"]
        else:
            X = df[numeric_cols].values
            feature_names = numeric_cols

        return X, y, feature_names

    def _load_data(self) -> pd.DataFrame:
        """
        Load Walmart sales data from CSV.

        Args:
            data_path: Path to Walmart.csv

        Returns:
            Raw Walmart data as DataFrame
        """
        path = Path(self.data_path)

        # Handle relative paths from project root
        if not path.is_absolute():
            project_dir = Path(__file__).parent.parent.parent.parent
            path = project_dir / self.data_path

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
            # Handle different date formats (dd-mm-yyyy, mm-dd-yyyy, etc.)
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], dayfirst=True, errors="coerce")
            # Sort by date for time series
            if "Store" in df.columns and "Dept" in df.columns:
                df = df.sort_values(["Store", "Dept", date_cols[0]])
            else:
                df = df.sort_values(date_cols[0])
            df = df.reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walmart-specific preprocessing and feature engineering.

        Args:
            df: Raw Walmart data

        Returns:
            Preprocessed DataFrame with engineered features
        """
        # Basic cleaning (copied from TimeSeriesDataset.preprocess_data)
        df = df.copy()

        # Handle missing values
        if df.isnull().any().any():
            logger.warning("Missing values detected. Filling with forward fill.")
            df = df.fillna(method="ffill").fillna(method="bfill")

        # Sort by time if date column exists
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            df = df.sort_values(date_cols[0]).reset_index(drop=True)

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
        target_column = self.dataset_kwargs.get("target_column", "Weekly_Sales")
        if target_column in df.columns:
            df["sales_lag_1"] = df[target_column].shift(1)
            df["sales_lag_4"] = df[target_column].shift(4)  # 4 weeks ago
            df["sales_lag_52"] = df[target_column].shift(52)  # 1 year ago
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
