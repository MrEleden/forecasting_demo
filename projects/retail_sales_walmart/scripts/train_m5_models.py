"""
M5-Inspired Walmart Forecasting Training Script

Implements the winning strategies from the M5 forecasting competition
adapted for Walmart sales data.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ml_portfolio.models.metrics import mae, mape, rmse
from omegaconf import DictConfig, OmegaConf

# from ml_portfolio.models.statistical.walmart_ensemble import WalmartEnsembleForecaster

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


def weighted_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate Weighted Mean Absolute Error (WMAE) - M5 competition metric.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        weights: Optional weights (defaults to equal weighting)

    Returns:
        WMAE score
    """
    if weights is None:
        weights = np.ones_like(y_true)

    abs_errors = np.abs(y_true - y_pred)
    return np.sum(weights * abs_errors) / np.sum(weights)


def mean_absolute_scaled_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 52
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).

    Args:
        y_true: Actual test values
        y_pred: Predicted values
        y_train: Training values for baseline calculation
        seasonal_period: Seasonal period for naive forecast

    Returns:
        MASE score
    """
    # Calculate seasonal naive forecast error on training set
    if len(y_train) <= seasonal_period:
        # Fallback to simple naive
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_forecast = y_train[:-seasonal_period]
        naive_actual = y_train[seasonal_period:]
        naive_errors = np.abs(naive_forecast - naive_actual)

    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf if np.mean(np.abs(y_true - y_pred)) > 0 else 0

    mae_forecast = np.mean(np.abs(y_true - y_pred))
    return mae_forecast / mae_naive


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def evaluate_forecast(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, model_name: str = "Model"
) -> Dict[str, float]:
    """
    Comprehensive forecast evaluation with M5-style metrics.

    Args:
        y_true: Actual test values
        y_pred: Predicted values
        y_train: Training values (for MASE calculation)
        model_name: Name of the model being evaluated

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "model": model_name,
        "wmae": weighted_absolute_error(y_true, y_pred),
        "mase": mean_absolute_scaled_error(y_true, y_pred, y_train),
        "mape": mape(y_true, y_pred),
        "smape": symmetric_mape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100,
    }

    return metrics


def create_forecast_plots(
    results_df: pd.DataFrame, test_dates: pd.DatetimeIndex, title: str = "M5-Inspired Walmart Forecasting Results"
) -> None:
    """Create comprehensive forecast visualization."""

    # Main forecast plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot 1: Forecast vs Actual
    ax1 = axes[0, 0]
    forecast_cols = [col for col in results_df.columns if col not in ["Date", "Actual"]]

    ax1.plot(test_dates, results_df["Actual"], "k-", linewidth=2, label="Actual", alpha=0.8)

    colors = plt.cm.Set3(np.linspace(0, 1, len(forecast_cols)))
    for i, col in enumerate(forecast_cols):
        ax1.plot(test_dates, results_df[col], "--", color=colors[i], linewidth=1.5, label=col, alpha=0.7)

    ax1.set_title("Forecasts vs Actual Sales")
    ax1.set_ylabel("Weekly Sales ($)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Forecast errors
    ax2 = axes[0, 1]
    for i, col in enumerate(forecast_cols):
        errors = results_df["Actual"] - results_df[col]
        ax2.plot(test_dates, errors, color=colors[i], alpha=0.7, label=f"{col} Error")

    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.set_title("Forecast Errors Over Time")
    ax2.set_ylabel("Error (Actual - Predicted)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    error_data = []
    error_labels = []

    for col in forecast_cols:
        errors = results_df["Actual"] - results_df[col]
        error_data.append(errors)
        error_labels.append(col)

    ax3.boxplot(error_data, labels=error_labels)
    ax3.set_title("Error Distribution by Model")
    ax3.set_ylabel("Forecast Error")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Seasonal pattern comparison
    ax4 = axes[1, 1]

    # Extract week of year for seasonal analysis
    weeks = test_dates.isocalendar().week
    seasonal_actual = results_df.groupby(weeks)["Actual"].mean()

    ax4.plot(seasonal_actual.index, seasonal_actual.values, "ko-", linewidth=2, label="Actual (Weekly Avg)", alpha=0.8)

    for i, col in enumerate(forecast_cols[:3]):  # Limit to top 3 for clarity
        seasonal_forecast = results_df.groupby(weeks)[col].mean()
        ax4.plot(
            seasonal_forecast.index,
            seasonal_forecast.values,
            "--o",
            color=colors[i],
            label=f"{col} (Weekly Avg)",
            alpha=0.7,
        )

    ax4.set_title("Seasonal Pattern Comparison")
    ax4.set_xlabel("Week of Year")
    ax4.set_ylabel("Average Weekly Sales")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train_and_evaluate_models(cfg: DictConfig) -> pd.DataFrame:
    """
    Train and evaluate M5-inspired forecasting models.

    Args:
        cfg: Hydra configuration

    Returns:
        Results DataFrame with metrics for each model
    """
    print("🏪 M5-Inspired Walmart Forecasting Training")
    print("=" * 50)

    # Load and prepare data
    data_path = Path(cfg.data.path) if "data" in cfg and "path" in cfg.data else None

    if data_path is None:
        # Default path construction
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_path = project_root / "projects" / "retail_sales_walmart" / "data" / "raw" / "Walmart.csv"

    print(f"📊 Loading data from: {data_path}")

    # Load raw data for aggregated modeling
    df_raw = pd.read_csv(data_path, parse_dates=["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    # Create aggregated weekly series
    weekly_sales = (
        df_raw.groupby("Date", as_index=False)["Weekly_Sales"]
        .sum()
        .sort_values("Date")
        .set_index("Date")["Weekly_Sales"]
    )

    # Train/test split
    test_horizon = cfg.training.test_horizon
    train_series = weekly_sales.iloc[:-test_horizon]
    test_series = weekly_sales.iloc[-test_horizon:]

    print(f"📈 Training periods: {len(train_series)}")
    print(f"📉 Test periods: {len(test_series)}")
    print(f"💰 Training period sales: ${train_series.sum():,.0f}")

    # Initialize models to test
    models_to_test = [
        # Commented out until WalmartEnsembleForecaster is implemented
        # (
        #     "M5_Full_Ensemble",
        #     WalmartEnsembleForecaster(
        #         models=["seasonal_naive", "ets", "auto_arima", "svd_ets", "fourier_arima", "stlf_arima"],
        #         seasonal_period=52,
        #         enable_calendar_adjustment=True,
        #     ),
        # ),
        # (
        #     "SVD_STLF",
        #     WalmartEnsembleForecaster(models=["svd_ets", "stlf_arima"], seasonal_period=52, n_svd_components=15),
        # ),
        # (
        #     "Fourier_ARIMA",
        #     WalmartEnsembleForecaster(
        #         models=["fourier_arima", "seasonal_naive"], fourier_terms=8, seasonal_period=52
        #     ),
        # ),
        # (
        #     "Classical_Ensemble",
        #     WalmartEnsembleForecaster(models=["auto_arima", "ets", "seasonal_naive"], seasonal_period=52),
        # ),
        # ("Pure_ETS", WalmartEnsembleForecaster(models=["ets"], seasonal_period=52)),
        # (
        #     "Enhanced_Seasonal_Naive",
        #     WalmartEnsembleForecaster(models=["seasonal_naive"], seasonal_period=52, enable_calendar_adjustment=True),
        # ),
    ]

    # Train and evaluate models
    results = []
    forecast_data = {"Date": test_series.index, "Actual": test_series.values}

    for model_name, model in models_to_test:
        print(f"\n🔮 Training {model_name}...")

        try:
            # Fit model (reshape for single series)
            model.fit(train_series.values.reshape(-1, 1))

            # Generate forecast
            forecast = model.predict(steps=len(test_series), forecast_dates=test_series.index)[
                0
            ]  # Extract single series forecast

            # Evaluate
            metrics = evaluate_forecast(test_series.values, forecast, train_series.values, model_name)

            results.append(metrics)
            forecast_data[model_name] = forecast

            print(f"   ✅ WMAE: {metrics['wmae']:.2f} | MASE: {metrics['mase']:.3f} | MAPE: {metrics['mape']:.2f}%")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    forecast_df = pd.DataFrame(forecast_data)

    # Display results
    print("\n📊 Model Performance Summary")
    print("=" * 50)

    if len(results_df) > 0:
        # Sort by primary metric (WMAE)
        results_display = results_df.sort_values("wmae").round(3)
        print(results_display[["model", "wmae", "mase", "mape", "rmse", "directional_accuracy"]])

        # Create visualization
        create_forecast_plots(forecast_df, test_series.index)

        # Save results
        output_dir = Path("outputs") / "walmart_m5_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_dir / "model_performance.csv", index=False)
        forecast_df.to_csv(output_dir / "forecasts.csv", index=False)

        print(f"\n💾 Results saved to: {output_dir}")

    else:
        print("❌ No models completed successfully!")
        results_df = pd.DataFrame()

    return results_df


@hydra.main(version_base=None, config_path="conf", config_name="walmart_m5_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print("🔧 Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Train and evaluate models
    results_df = train_and_evaluate_models(cfg)

    if len(results_df) > 0:
        best_model = results_df.loc[results_df["wmae"].idxmin()]
        print(f"\n🏆 Best Model: {best_model['model']}")
        print(f"   📈 WMAE: {best_model['wmae']:.3f}")
        print(f"   📊 MASE: {best_model['mase']:.3f}")
        print(f"   📉 MAPE: {best_model['mape']:.2f}%")


if __name__ == "__main__":
    main()
