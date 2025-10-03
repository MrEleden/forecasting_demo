# projects/walmart/data/factory.py
from ml_portfolio.data.datasets import DatasetFactory
from ml_portfolio.data.feature_engineering import TimeSeriesFeatureEngineer


class WalmartDatasetFactory(DatasetFactory):
    """
    Walmart-specific factory with feature engineering.
    Uses reusable TimeSeriesFeatureEngineer for common features.
    """

    def __init__(self, **kwargs):
        super().__init__(target_column="Weekly_Sales", timestamp_column="Date", **kwargs)

        # Setup feature engineer with Walmart-specific configuration
        self.feature_engineer = TimeSeriesFeatureEngineer(
            date_column="Date",
            group_columns=["Store", "Dept"],  # Walmart has store/dept hierarchy
            target_column="Weekly_Sales",
            lag_features=[1, 4, 52],  # Weekly lags: 1 week, 1 month, 1 year
            rolling_windows=[4, 8, 52],  # Monthly, bi-monthly, yearly
            date_features=True,
            cyclical_features=["week", "month"],
        )

    def _extract_arrays(self, df: pd.DataFrame) -> tuple:
        """Add features before splitting (all safe transformations)."""
        # Add Walmart-specific binary features (deterministic)
        if "IsHoliday" in df.columns:
            df["is_holiday"] = df["IsHoliday"].astype(int)

        if "Type" in df.columns:
            # Store type encoding (deterministic)
            df["store_type"] = df["Type"].map({"A": 3, "B": 2, "C": 1})

        # Apply reusable feature engineering
        df = self.feature_engineer.engineer_features(df)

        # Drop rows with NaN from lagging (first few rows)
        df = df.dropna()

        # Now extract using parent method
        return super()._extract_arrays(df)
