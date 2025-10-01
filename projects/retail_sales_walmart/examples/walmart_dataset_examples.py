"""
Example usage of the custom WalmartTimeSeriesDataset.

This demonstrates the proper way to use project-specific datasets
following the inheritance pattern.
"""

import os
import sys
from pathlib import Path

# Add the data directory to path to import walmart_dataset
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
sys.path.insert(0, str(data_dir))

try:
    from walmart_dataset import WalmartTimeSeriesDataset, WalmartMultiStoreDataset, create_walmart_dataset
except ImportError:
    print(f"Note: Run this script from the project directory or ensure walmart_dataset.py exists in {data_dir}")
    raise


def example_aggregated_forecasting():
    """Example: Forecast total Walmart sales across all stores."""
    print("🏪 Example: Aggregated Walmart Sales Forecasting")
    print("-" * 50)

    # Create dataset with all Walmart-specific features
    dataset = WalmartTimeSeriesDataset(
        aggregate_stores=True,  # Sum across all stores
        include_economic_features=True,  # Include CPI, unemployment
        include_weather_features=True,  # Include temperature, fuel price
        lookback_window=52,  # 1 year of weekly data
        forecast_horizon=4,  # Predict 4 weeks ahead
    )

    print(f"Created dataset with {len(dataset)} sequences")
    print(f"Each sequence: {dataset.sequences[0][0].shape} → {dataset.sequences[0][1].shape}")

    # Get insights
    insights = dataset.get_walmart_insights()
    print(f"Average weekly sales: ${insights['sales_statistics']['mean_weekly_sales']:,.0f}")
    print(f"Sales volatility: ${insights['sales_statistics']['sales_volatility']:,.0f}")

    return dataset


def example_single_store_forecasting():
    """Example: Forecast a specific Walmart store."""
    print("\n🏬 Example: Single Store Forecasting (Store 1)")
    print("-" * 50)

    # Create dataset for Store 1 only
    dataset = WalmartTimeSeriesDataset(
        store_id=1,  # Focus on Store 1
        aggregate_stores=False,  # Don't aggregate
        include_economic_features=True,
        lookback_window=26,  # 6 months for individual store
        forecast_horizon=2,  # Predict 2 weeks ahead
    )

    print(f"Store 1 dataset: {len(dataset)} sequences")
    print(f"Each sequence: {dataset.sequences[0][0].shape} → {dataset.sequences[0][1].shape}")

    return dataset


def example_multi_store_forecasting():
    """Example: Forecast multiple stores simultaneously."""
    print("\n🏬 Example: Multi-Store Forecasting")
    print("-" * 50)

    # Create multi-store dataset for top stores
    dataset = WalmartMultiStoreDataset(
        store_list=[1, 2, 3, 4, 5],  # Top 5 stores
        min_data_points=100,  # Require at least 100 data points
        lookback_window=26,  # 6 months lookback
        forecast_horizon=2,  # 2 weeks ahead
    )

    print(f"Multi-store dataset: {len(dataset)} sequences")

    # Show store breakdown
    if dataset.sequences:
        sample_seq = dataset.sequences[0]
        print(f"Each sequence: {sample_seq[0].shape} → {sample_seq[1].shape}")
        print(f"Store ID for first sequence: {sample_seq[2]}")

    return dataset


def example_factory_usage():
    """Example: Using the factory function."""
    print("\n🏭 Example: Factory Function Usage")
    print("-" * 50)

    # Create datasets using factory function
    single_dataset = create_walmart_dataset("single", aggregate_stores=True)
    multi_dataset = create_walmart_dataset("multi", store_list=[1, 2, 3])

    print(f"Factory single dataset: {len(single_dataset)} sequences")
    print(f"Factory multi dataset: {len(multi_dataset)} sequences")

    return single_dataset, multi_dataset


def main():
    """Main example demonstrating all Walmart dataset usage patterns."""
    print("🎯 Walmart Custom Dataset Examples")
    print("=" * 60)

    # Run all examples
    agg_dataset = example_aggregated_forecasting()
    store_dataset = example_single_store_forecasting()
    multi_dataset = example_multi_store_forecasting()
    single_factory, multi_factory = example_factory_usage()

    print("\n✅ Summary:")
    print(f"📊 Aggregated dataset: {len(agg_dataset)} sequences (all stores combined)")
    print(f"🏬 Single store dataset: {len(store_dataset)} sequences (Store 1 only)")
    print(f"🏬 Multi-store dataset: {len(multi_dataset)} sequences (5 stores)")
    print(f"🏭 Factory datasets: {len(single_factory)} + {len(multi_factory)} sequences")

    print("\n🚀 Ready for model training!")
    print("💡 Use these datasets with any ML model (sklearn, PyTorch, etc.)")

    return {
        "aggregated": agg_dataset,
        "single_store": store_dataset,
        "multi_store": multi_dataset,
        "factory_single": single_factory,
        "factory_multi": multi_factory,
    }


if __name__ == "__main__":
    datasets = main()
