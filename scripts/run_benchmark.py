"""
Run benchmark suite comparing multiple forecasting models.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --dataset walmart --models lightgbm,catboost
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402

import pandas as pd  # noqa: E402
from ml_portfolio.evaluation.benchmark import ModelBenchmark  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402


def create_models():
    """Create dictionary of models to benchmark."""
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
    }

    # Try to import optional models
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        print("LightGBM not available, skipping...")

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False,
        )
    except ImportError:
        print("CatBoost not available, skipping...")

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        print("XGBoost not available, skipping...")

    return models


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(description="Run model benchmark suite")
    parser.add_argument(
        "--dataset",
        type=str,
        default="walmart",
        choices=["walmart", "ola", "tsi"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to test (e.g., 'lightgbm,catboost')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FORECASTING MODEL BENCHMARK SUITE")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}\n")

    # Load dataset
    print(f"Loading {args.dataset} dataset...")

    if args.dataset == "walmart":
        data_path = "projects/retail_sales_walmart/data/processed/walmart_sales.csv"
    elif args.dataset == "ola":
        data_path = "projects/rideshare_demand_ola/data/processed/ola_demand.csv"
    else:
        data_path = "projects/transportation_tsi/data/processed/tsi_data.csv"

    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        print("Please run data download/generation script first.")
        sys.exit(1)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")

    # Simple feature engineering
    target_col = "Weekly_Sales" if args.dataset == "walmart" else "demand"

    if target_col not in df.columns:
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=["number"]).columns
        target_col = numeric_cols[0]

    # Create features and target
    feature_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != target_col]

    if not feature_cols:
        print("Error: No numeric features found")
        sys.exit(1)

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")

    # Create models
    all_models = create_models()

    # Filter models if specified
    if args.models:
        model_names = [m.strip().lower() for m in args.models.split(",")]
        models = {name: model for name, model in all_models.items() if name.lower() in model_names}
        if not models:
            print(f"Error: No matching models found. Available: {list(all_models.keys())}")
            sys.exit(1)
    else:
        models = all_models

    print(f"\nModels to benchmark: {list(models.keys())}\n")

    # Run benchmark
    benchmark = ModelBenchmark(output_dir=args.output_dir)

    benchmark.run_multiple_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name=args.dataset,
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")

    df_results = benchmark.get_results_dataframe()
    print(df_results.to_string(index=False))

    # Display rankings
    print("\n" + "-" * 80)
    print("RANKINGS (by MAPE)")
    print("-" * 80 + "\n")
    print(benchmark.get_ranking("mape").to_string(index=False))

    # Save results
    benchmark.save_results()

    # Generate plots
    print("\nGenerating comparison plots...")
    benchmark.plot_comparison(metric="mape")
    benchmark.plot_comparison(metric="rmse")
    benchmark.plot_training_time_vs_accuracy(metric="mape")

    # Generate report
    report = benchmark.generate_report()
    print("\n" + report)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
