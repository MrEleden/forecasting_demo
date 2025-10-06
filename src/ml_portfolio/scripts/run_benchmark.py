"""
Run benchmark suite comparing multiple forecasting models from MLflow.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --dataset walmart --models lightgbm,catboost
    python scripts/run_benchmark.py --experiment-name "Walmart Forecasting"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402


def get_mlflow_runs(experiment_name=None, dataset_filter=None, model_filter=None):
    """
    Retrieve model runs from MLflow tracking server.

    Args:
        experiment_name: Name of MLflow experiment (optional)
        dataset_filter: Filter by dataset name (optional)
        model_filter: List of model names to filter (optional)

    Returns:
        DataFrame with benchmark results from MLflow
    """
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found.")
            return pd.DataFrame()
        experiment_ids = [experiment.experiment_id]
    else:
        # Get all experiments
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments if exp.name != "Default"]

    if not experiment_ids:
        print("No experiments found in MLflow.")
        return pd.DataFrame()

    # Search runs across experiments
    all_runs = []
    for exp_id in experiment_ids:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            order_by=["start_time DESC"],
        )
        all_runs.extend(runs)

    if not all_runs:
        print("No runs found in MLflow experiments.")
        return pd.DataFrame()

    # Extract results
    results = []
    for run in all_runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        # Get model name from tags or params
        model_name = (
            tags.get("model_name")
            or tags.get("model_type")
            or params.get("model")
            or params.get("model._target_", "").split(".")[-1]
            or "Unknown"
        )
        dataset_name = tags.get("dataset") or params.get("dataset") or params.get("dataset_name") or "Unknown"

        # Apply filters
        if dataset_filter and dataset_name.lower() != dataset_filter.lower():
            continue

        if model_filter and model_name.lower() not in [m.lower() for m in model_filter]:
            continue

        # Try different metric naming conventions
        mape = (
            metrics.get("test_mape")
            or metrics.get("mape")
            or metrics.get("test_MAPEMetric")
            or metrics.get("MAPEMetric")
        )
        rmse = (
            metrics.get("test_rmse")
            or metrics.get("rmse")
            or metrics.get("test_RMSEMetric")
            or metrics.get("RMSEMetric")
        )
        mae = metrics.get("test_mae") or metrics.get("mae") or metrics.get("test_MAEMetric") or metrics.get("MAEMetric")

        result = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "mape": mape,
            "rmse": rmse,
            "mae": mae,
            "training_time": metrics.get("training_time", 0),
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
        }

        # Only include if has at least one metric
        if any(result[m] is not None for m in ["mape", "rmse", "mae"]):
            results.append(result)

    df = pd.DataFrame(results)

    if df.empty:
        print("No matching runs found with required metrics.")
        return df

    # Sort by start_time and keep most recent run per model+dataset
    df = df.sort_values("start_time", ascending=False)
    df = df.groupby(["model_name", "dataset_name"]).first().reset_index()

    return df


def main():
    """Run benchmark suite from MLflow tracking data."""
    parser = argparse.ArgumentParser(description="Run model benchmark suite from MLflow")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset name (walmart, ola, tsi)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to include (e.g., 'lightgbm,catboost')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name of MLflow experiment to query",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FORECASTING MODEL BENCHMARK SUITE (FROM MLFLOW)")
    print("=" * 80)
    print("\nQuerying MLflow tracking server...")
    if args.dataset:
        print(f"Dataset filter: {args.dataset}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.experiment_name:
        print(f"Experiment: {args.experiment_name}")
    print(f"Output directory: {args.output_dir}\n")

    # Parse model filter
    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(",")]

    # Retrieve runs from MLflow
    print("Fetching runs from MLflow...")
    df_results = get_mlflow_runs(
        experiment_name=args.experiment_name,
        dataset_filter=args.dataset,
        model_filter=model_filter,
    )

    if df_results.empty:
        print("\n" + "=" * 80)
        print("NO RESULTS FOUND")
        print("=" * 80)
        print("\nNo matching runs found in MLflow.")
        print("\nTips:")
        print("  - Train some models first to populate MLflow")
        print("  - Check MLflow UI at http://localhost:5000")
        print("  - Use --experiment-name to specify experiment")
        print("  - Run: mlflow ui (to start MLflow server)")
        sys.exit(1)

    print(f"Found {len(df_results)} runs\n")

    # Display results
    print("=" * 80)
    print("BENCHMARK RESULTS FROM MLFLOW")
    print("=" * 80 + "\n")

    # Select columns to display
    display_cols = ["model_name", "dataset_name", "mape", "rmse", "mae", "training_time"]
    display_df = df_results[display_cols].copy()

    # Format for display
    for col in ["mape", "rmse", "mae", "training_time"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))

    # Rankings
    print("\n" + "-" * 80)
    print("RANKINGS (by MAPE)")
    print("-" * 80 + "\n")

    ranking_df = df_results[["model_name", "mape"]].copy()
    ranking_df = ranking_df.dropna(subset=["mape"])
    ranking_df = ranking_df.sort_values("mape")
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df = ranking_df[["rank", "model_name", "mape"]]
    print(ranking_df.to_string(index=False))

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_path / "mlflow_benchmark_results.json"
    df_results.to_json(json_path, orient="records", indent=2, date_format="iso")
    print(f"\nResults saved to: {json_path}")

    # Save CSV
    csv_path = output_path / "mlflow_benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Generate plots if matplotlib available
    try:
        import matplotlib.pyplot as plt

        # MAPE comparison
        if "mape" in df_results.columns and df_results["mape"].notna().any():
            plt.figure(figsize=(10, 6))
            plot_data = df_results.dropna(subset=["mape"]).sort_values("mape")
            plt.barh(plot_data["model_name"], plot_data["mape"])
            plt.xlabel("MAPE (%)")
            plt.title(f"Model Comparison - MAPE{' (' + args.dataset + ')' if args.dataset else ''}")
            plt.tight_layout()
            plot_path = output_path / "mlflow_benchmark_mape.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to: {plot_path}")

        # RMSE comparison
        if "rmse" in df_results.columns and df_results["rmse"].notna().any():
            plt.figure(figsize=(10, 6))
            plot_data = df_results.dropna(subset=["rmse"]).sort_values("rmse")
            plt.barh(plot_data["model_name"], plot_data["rmse"])
            plt.xlabel("RMSE")
            plt.title(f"Model Comparison - RMSE{' (' + args.dataset + ')' if args.dataset else ''}")
            plt.tight_layout()
            plot_path = output_path / "mlflow_benchmark_rmse.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to: {plot_path}")

    except ImportError:
        print("\nMatplotlib not available, skipping plots")

    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MLFLOW BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal Runs: {len(df_results)}")
    report_lines.append(f"Datasets: {df_results['dataset_name'].nunique()}")
    report_lines.append(f"Models: {df_results['model_name'].nunique()}")

    if not df_results.empty:
        report_lines.append("\n" + "-" * 80)
        report_lines.append("BEST PERFORMING MODELS")
        report_lines.append("-" * 80)

        for metric in ["mape", "rmse", "mae"]:
            if metric in df_results.columns and df_results[metric].notna().any():
                best_row = df_results.loc[df_results[metric].idxmin()]
                report_lines.append(f"\nBest {metric.upper()}: {best_row['model_name']}")
                report_lines.append(f"  {metric.upper()}: {best_row[metric]:.4f}")
                report_lines.append(f"  Dataset: {best_row['dataset_name']}")

    report_lines.append("\n" + "=" * 80)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    report_path = output_path / "mlflow_benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"\nResults directory: {args.output_dir}")


if __name__ == "__main__":
    main()
