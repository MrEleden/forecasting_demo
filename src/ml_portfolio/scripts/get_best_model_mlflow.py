"""
Retrieve the best model from MLflow after benchmark sweep.

This script:
1. Queries MLflow for all runs in the experiment
2. Finds the best run based on the primary metric (MAPE)
3. Displays comparison table of all models
4. Saves best model information
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from mlflow.tracking import MlflowClient


def get_all_runs(experiment_name: str) -> pd.DataFrame:
    """
    Get all runs from an MLflow experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        DataFrame with all runs and their metrics
    """
    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found!")

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_MAPEMetric ASC"],  # Order by MAPE ascending (lower is better)
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    # Extract relevant information
    results = []
    for run in runs:
        # Extract model name from the config
        model_config = run.data.params.get("model", "Unknown")
        if isinstance(model_config, str) and "_target_" in model_config:
            # Extract model class name from _target_ path
            # e.g., 'ml_portfolio.models.statistical.xgboost.XGBoostForecaster' -> 'XGBoost'
            try:
                import ast

                config_dict = ast.literal_eval(model_config)
                target = config_dict.get("_target_", "")
                model_name = target.split(".")[-1].replace("Forecaster", "").replace("Regressor", "")
            except Exception:
                model_name = model_config
        else:
            model_name = model_config

        run_data = {
            "run_id": run.info.run_id,
            "model": model_name,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "end_time": pd.to_datetime(run.info.end_time, unit="ms") if run.info.end_time else None,
        }

        # Add metrics
        for key, value in run.data.metrics.items():
            run_data[key] = value

        # Add params
        for key, value in run.data.params.items():
            if key not in run_data:  # Don't overwrite existing keys
                run_data[f"param_{key}"] = value

        results.append(run_data)

    return pd.DataFrame(results)


def display_comparison_table(df: pd.DataFrame):
    """Display comparison table of all models."""
    print("\n" + "=" * 100)
    print("WALMART SALES FORECASTING - MLflow Experiment Results")
    print("=" * 100)

    # Select relevant columns
    display_cols = ["model", "test_MAPEMetric", "test_RMSEMetric", "test_MAEMetric", "training_time"]

    # Filter columns that exist
    available_cols = [col for col in display_cols if col in df.columns]

    if not available_cols:
        print("No metrics found in runs!")
        return

    # Rename for display
    display_df = df[available_cols].copy()
    rename_map = {
        "model": "Model",
        "test_MAPEMetric": "Test MAPE (%)",
        "test_RMSEMetric": "Test RMSE ($)",
        "test_MAEMetric": "Test MAE ($)",
        "training_time": "Training Time (s)",
    }
    display_df.columns = [rename_map.get(col, col) for col in display_df.columns]

    # Sort by MAPE if available
    if "Test MAPE (%)" in display_df.columns:
        display_df = display_df.sort_values("Test MAPE (%)")

    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1  # Start from 1

    print(display_df.to_string())
    print("=" * 100)


def get_best_model(df: pd.DataFrame, metric: str = "test_MAPEMetric") -> Dict:
    """
    Get the best model based on a metric.

    Args:
        df: DataFrame with all runs
        metric: Metric to optimize (lower is better)

    Returns:
        Dictionary with best model information
    """
    # Find best run (minimum metric value)
    best_idx = df[metric].idxmin()
    best_run = df.loc[best_idx]

    return {
        "run_id": best_run["run_id"],
        "model": best_run["model"],
        "test_mape": best_run.get("test_MAPEMetric", None),
        "test_rmse": best_run.get("test_RMSEMetric", None),
        "test_mae": best_run.get("test_MAEMetric", None),
        "training_time": best_run.get("training_time", None),
        "val_mape": best_run.get("val_MAPEMetric", None),
    }


def save_best_model_info(best_model: Dict, output_path: Path):
    """Save best model information to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(best_model, f, indent=2)

    print(f"\nBest model information saved to: {output_path}")


def download_best_model_artifact(run_id: str, output_dir: Path):
    """Download the best model artifact from MLflow."""
    client = MlflowClient()

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download artifacts
        artifact_path = client.download_artifacts(run_id, "", dst_path=str(output_dir))
        print(f"Model artifacts downloaded to: {artifact_path}")
    except Exception as e:
        print(f"Could not download artifacts: {e}")


def main(experiment_name: str = "walmart_model_comparison"):
    """Main function to retrieve and display best model."""
    print(f"\nQuerying MLflow experiment: {experiment_name}")

    try:
        # Get all runs
        df = get_all_runs(experiment_name)

        print(f"\nFound {len(df)} runs")

        # Display comparison table
        display_comparison_table(df)

        # Get best model
        best_model = get_best_model(df, metric="test_MAPEMetric")

        # Extract clean model name for best model display
        model_name = best_model["model"]
        if isinstance(model_name, str) and len(model_name) > 50:
            # If it's still a long config string, extract the class name
            try:
                import ast

                config_dict = ast.literal_eval(model_name)
                target = config_dict.get("_target_", "")
                model_name = target.split(".")[-1].replace("Forecaster", "").replace("Regressor", "")
            except Exception:
                pass

        print("\n" + "=" * 100)
        print("BEST MODEL (by Test MAPE)")
        print("=" * 100)
        print(f"Model: {model_name}")
        print(f"Run ID: {best_model['run_id']}")
        if best_model["test_mape"]:
            print(f"Test MAPE: {best_model['test_mape']:.4f}%")
        if best_model["test_rmse"]:
            print(f"Test RMSE: ${best_model['test_rmse']:,.2f}")
        if best_model["test_mae"]:
            print(f"Test MAE: ${best_model['test_mae']:,.2f}")
        if best_model["training_time"]:
            print(f"Training Time: {best_model['training_time']:.2f}s")
        if best_model["val_mape"]:
            print(f"Validation MAPE: {best_model['val_mape']:.4f}%")
        print("=" * 100)

        # Save best model info
        output_path = Path("results") / "best_model_info.json"
        save_best_model_info(best_model, output_path)

        # Download artifacts
        artifact_dir = Path("results") / "best_model_artifacts"
        print("\nDownloading best model artifacts...")
        download_best_model_artifact(best_model["run_id"], artifact_dir)

        # Print MLflow UI command
        print("\n" + "=" * 100)
        print("To view all experiments in MLflow UI, run:")
        print("  mlflow ui")
        print("Then navigate to http://localhost:5000")
        print("=" * 100)

        return best_model

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. MLflow tracking is enabled (use_mlflow: true)")
        print("2. You've run the benchmark sweep:")
        print("   python src/ml_portfolio/training/train.py --config-name walmart_sweep -m")
        print("3. The experiment name matches the config")
        return None


if __name__ == "__main__":
    import sys

    experiment_name = "walmart_model_comparison"
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]

    best_model = main(experiment_name)

    if best_model:
        model_name = best_model["model"]
        if isinstance(model_name, str) and len(model_name) > 50:
            try:
                import ast

                config_dict = ast.literal_eval(model_name)
                target = config_dict.get("_target_", "")
                model_name = target.split(".")[-1].replace("Forecaster", "").replace("Regressor", "")
            except Exception:
                pass
        print(f"\nBest model: {model_name} (MAPE: {best_model.get('test_mape', 'N/A'):.4f}%)")
