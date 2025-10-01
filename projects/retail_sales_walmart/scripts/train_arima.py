"""
Train ARIMA model on Walmart dataset using Hydra configuration.
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Hydra imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")

# Add src to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_portfolio.models.statistical.statistical import ARIMAWrapper
from ml_portfolio.models.metrics import rmse, mae, mape
from ml_portfolio.utils.io import ensure_dir


def load_walmart_data(config: DictConfig) -> pd.DataFrame:
    """
    Load and preprocess Walmart dataset.

    Args:
        config: Dataset configuration

    Returns:
        Preprocessed DataFrame
    """
    # Get file path relative to project
    project_dir = Path(__file__).parent.parent
    file_path = project_dir / config.dataset.file_path

    print(f"Loading data from: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Parse date column
    df[config.dataset.date_column] = pd.to_datetime(df[config.dataset.date_column], format="%d-%m-%Y")

    # Sort by date
    df = df.sort_values(config.dataset.date_column)

    # Handle aggregation
    if config.dataset.aggregate_by == "total":
        # Aggregate all stores and departments
        df_agg = df.groupby(config.dataset.date_column)[config.dataset.target_column].sum().reset_index()
    elif config.dataset.aggregate_by == "store":
        # Aggregate by store (sum across departments)
        df_agg = (
            df.groupby([config.dataset.date_column, config.dataset.store_column])[config.dataset.target_column]
            .sum()
            .reset_index()
        )
        # For simplicity, take first store
        store_id = df_agg[config.dataset.store_column].iloc[0]
        df_agg = df_agg[df_agg[config.dataset.store_column] == store_id]
    else:
        # Take a specific store for demo
        store_id = df[config.dataset.store_column].iloc[0]
        df_agg = df[df[config.dataset.store_column] == store_id]
        print(f"Using Store {store_id}")

    # Create time series
    ts_data = df_agg.set_index(config.dataset.date_column)[config.dataset.target_column]

    # Handle missing values
    if config.dataset.fillna_method == "forward":
        ts_data = ts_data.fillna(method="ffill")
    elif config.dataset.fillna_method == "backward":
        ts_data = ts_data.fillna(method="bfill")

    # Apply log transform if specified
    if config.dataset.log_transform:
        ts_data = np.log1p(ts_data)

    print(f"Loaded time series with {len(ts_data)} observations")
    print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
    print(f"Target statistics:")
    print(ts_data.describe())

    return ts_data


def split_time_series(ts_data: pd.Series, config: DictConfig):
    """
    Split time series into train/validation/test sets.

    Args:
        ts_data: Time series data
        config: Dataset configuration

    Returns:
        Tuple of (train, val, test) series
    """
    n = len(ts_data)

    train_size = int(n * config.dataset.train_size)
    val_size = int(n * config.dataset.validation_size)

    train_data = ts_data.iloc[:train_size]
    val_data = ts_data.iloc[train_size : train_size + val_size]
    test_data = ts_data.iloc[train_size + val_size :]

    print(f"Data split:")
    print(f"  Train: {len(train_data)} observations ({train_data.index.min()} to {train_data.index.max()})")
    print(f"  Validation: {len(val_data)} observations ({val_data.index.min()} to {val_data.index.max()})")
    print(f"  Test: {len(test_data)} observations ({test_data.index.min()} to {test_data.index.max()})")

    return train_data, val_data, test_data


def train_arima_model(train_data: pd.Series, config: DictConfig):
    """
    Train ARIMA model using Hydra instantiation.

    Args:
        train_data: Training time series
        config: Configuration

    Returns:
        Fitted ARIMA model
    """
    print(f"Training ARIMA model with configuration:")
    print(f"  Order: {config.order}")
    print(f"  Seasonal order: {config.seasonal_order}")
    print(f"  Trend: {config.trend}")

    # Instantiate model using Hydra
    model_config = {
        "_target_": "ml_portfolio.models.statistical.ARIMAWrapper",
        "order": config.order,
        "seasonal_order": config.seasonal_order,
        "trend": config.trend,
        "enforce_stationarity": config.enforce_stationarity,
        "enforce_invertibility": config.enforce_invertibility,
        "concentrate_scale": config.concentrate_scale,
    }
    model = instantiate(model_config)

    # Fit the model
    print("Fitting ARIMA model...")
    try:
        model.fit(None, train_data)
        print("ARIMA model fitted successfully!")

        # Print model summary if available
        try:
            summary = model.summary()
            print("\nModel Summary:")
            print(summary)
        except:
            print("Model summary not available")

    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        raise

    return model


def evaluate_model(model, val_data: pd.Series, test_data: pd.Series):
    """
    Evaluate the trained model.

    Args:
        model: Trained model
        val_data: Validation data
        test_data: Test data
    """
    print("\nEvaluating model...")

    # Validation predictions
    print("\nValidation Set:")
    val_pred = model.predict(len(val_data))

    val_rmse = rmse(val_data.values, val_pred)
    val_mae = mae(val_data.values, val_pred)
    val_mape = mape(val_data.values, val_pred)

    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  MAE: {val_mae:.2f}")
    print(f"  MAPE: {val_mape:.2f}%")

    # Test predictions
    print("\nTest Set:")
    test_pred = model.predict(len(test_data))

    test_rmse = rmse(test_data.values, test_pred)
    test_mae = mae(test_data.values, test_pred)
    test_mape = mape(test_data.values, test_pred)

    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  MAPE: {test_mape:.2f}%")

    # Create predictions with confidence intervals
    try:
        prediction_result = model.predict_with_intervals(steps=len(test_data), alpha=0.05)
        print(f"\nPredictions with 95% confidence intervals:")
        print(f"  Mean predictions: {prediction_result['predictions'][:5]} ... (showing first 5)")
        print(f"  Lower CI: {prediction_result['lower_ci'][:5]} ... (showing first 5)")
        print(f"  Upper CI: {prediction_result['upper_ci'][:5]} ... (showing first 5)")
    except Exception as e:
        print(f"Error generating predictions with CI: {e}")

    return {
        "val_metrics": {"rmse": val_rmse, "mae": val_mae, "mape": val_mape},
        "test_metrics": {"rmse": test_rmse, "mae": test_mae, "mape": test_mape},
        "val_predictions": val_pred,
        "test_predictions": test_pred,
    }


def save_results(model, results: dict, config: DictConfig):
    """
    Save model and results.

    Args:
        model: Trained model
        results: Evaluation results
        config: Configuration
    """
    # Create output directory
    output_dir = Path("outputs") / config.experiment.name
    ensure_dir(output_dir)

    # Save configuration
    OmegaConf.save(config, output_dir / "config.yaml")

    # Save results
    results_df = pd.DataFrame(
        {
            "metric": ["val_rmse", "val_mae", "val_mape", "test_rmse", "test_mae", "test_mape"],
            "value": [
                results["val_metrics"]["rmse"],
                results["val_metrics"]["mae"],
                results["val_metrics"]["mape"],
                results["test_metrics"]["rmse"],
                results["test_metrics"]["mae"],
                results["test_metrics"]["mape"],
            ],
        }
    )
    results_df.to_csv(output_dir / "metrics.csv", index=False)

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "val_predictions": pd.Series(results["val_predictions"], dtype=float),
            "test_predictions": pd.Series(results["test_predictions"], dtype=float),
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training function using Hydra.

    Args:
        config: Hydra configuration
    """
    print("Walmart ARIMA Forecasting with Hydra")
    print("=" * 50)
    print(f"Configuration:")
    print(OmegaConf.to_yaml(config))

    try:
        # Load data
        ts_data = load_walmart_data(config)

        # Split data
        train_data, val_data, test_data = split_time_series(ts_data, config)

        # Train model
        model = train_arima_model(train_data, config)

        # Evaluate model
        results = evaluate_model(model, val_data, test_data)

        # Save results
        save_results(model, results, config)

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
