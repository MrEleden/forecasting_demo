"""
Benchmark suite for comparing forecasting models.

This module provides a comprehensive framework for comparing multiple models
on standardized metrics and datasets.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml_portfolio.evaluation.metrics import mae, mape, rmse


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark."""

    model_name: str
    dataset_name: str
    mape: float
    rmse: float
    mae: float
    training_time: float
    prediction_time: float
    n_samples: int
    n_features: int
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelBenchmark:
    """Benchmark suite for comparing forecasting models."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark for a single model.

        Args:
            model: Model instance with fit() and predict() methods
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            dataset_name: Name of the dataset
            params: Model hyperparameters

        Returns:
            BenchmarkResult with metrics
        """
        print(f"\nBenchmarking {model_name} on {dataset_name}...")

        # Training phase
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"  Training time: {training_time:.2f}s")
        except Exception as e:
            print(f"  Training failed: {e}")
            return None

        # Prediction phase
        start_time = time.time()
        try:
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            print(f"  Prediction time: {prediction_time:.2f}s")
        except Exception as e:
            print(f"  Prediction failed: {e}")
            return None

        # Calculate metrics
        try:
            mape_score = mape(y_test.values, y_pred)
            rmse_score = rmse(y_test.values, y_pred)
            mae_score = mae(y_test.values, y_pred)

            print(f"  MAPE: {mape_score:.4f}")
            print(f"  RMSE: {rmse_score:.4f}")
            print(f"  MAE: {mae_score:.4f}")

            result = BenchmarkResult(
                model_name=model_name,
                dataset_name=dataset_name,
                mape=mape_score,
                rmse=rmse_score,
                mae=mae_score,
                training_time=training_time,
                prediction_time=prediction_time,
                n_samples=len(X_test),
                n_features=X_test.shape[1],
                params=params,
            )

            self.results.append(result)
            return result

        except Exception as e:
            print(f"  Metric calculation failed: {e}")
            return None

    def run_multiple_models(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for multiple models.

        Args:
            models: Dictionary of {model_name: model_instance}
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            dataset_name: Name of the dataset

        Returns:
            List of BenchmarkResult objects
        """
        results = []
        for model_name, model in models.items():
            result = self.run_benchmark(
                model=model,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                dataset_name=dataset_name,
            )
            if result:
                results.append(result)

        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.

        Returns:
            DataFrame with all benchmark results
        """
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame([r.to_dict() for r in self.results])

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics grouped by model.

        Returns:
            DataFrame with mean and std for each metric by model
        """
        df = self.get_results_dataframe()
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby("model_name").agg(
            {
                "mape": ["mean", "std", "min"],
                "rmse": ["mean", "std", "min"],
                "mae": ["mean", "std", "min"],
                "training_time": ["mean", "std"],
                "prediction_time": ["mean", "std"],
            }
        )

        return summary

    def get_ranking(self, metric: str = "mape") -> pd.DataFrame:
        """
        Get model ranking by metric.

        Args:
            metric: Metric to rank by ('mape', 'rmse', 'mae')

        Returns:
            DataFrame with model rankings
        """
        df = self.get_results_dataframe()
        if df.empty:
            return pd.DataFrame()

        # Group by model and get mean metric
        ranking = df.groupby("model_name")[metric].mean().sort_values()
        ranking_df = pd.DataFrame(
            {
                "rank": range(1, len(ranking) + 1),
                "model_name": ranking.index,
                f"avg_{metric}": ranking.values,
            }
        )

        return ranking_df

    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save benchmark results to JSON file.

        Args:
            filename: Name of output file (default: benchmark_results.json)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "benchmark_results.json"

        output_path = self.output_dir / filename

        results_dict = {
            "results": [r.to_dict() for r in self.results],
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_models": len(set(r.model_name for r in self.results)),
            "n_datasets": len(set(r.dataset_name for r in self.results)),
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def plot_comparison(self, metric: str = "mape", save_fig: bool = True) -> Optional[plt.Figure]:
        """
        Plot model comparison.

        Args:
            metric: Metric to plot ('mape', 'rmse', 'mae')
            save_fig: Whether to save figure to disk

        Returns:
            Matplotlib figure object
        """
        df = self.get_results_dataframe()
        if df.empty:
            print("No results to plot")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Bar chart of mean metrics
        summary = df.groupby("model_name")[metric].mean().sort_values()
        summary.plot(kind="barh", ax=axes[0], color="steelblue")
        axes[0].set_xlabel(metric.upper())
        axes[0].set_ylabel("Model")
        axes[0].set_title(f"Model Comparison by {metric.upper()}")
        axes[0].grid(axis="x", alpha=0.3)

        # Plot 2: Box plot showing distribution
        df_sorted = df.sort_values(by=metric, ascending=False)
        sns.boxplot(data=df_sorted, y="model_name", x=metric, ax=axes[1])
        axes[1].set_xlabel(metric.upper())
        axes[1].set_ylabel("Model")
        axes[1].set_title(f"Distribution of {metric.upper()} Scores")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = self.output_dir / f"benchmark_comparison_{metric}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {fig_path}")

        return fig

    def plot_training_time_vs_accuracy(self, metric: str = "mape", save_fig: bool = True) -> Optional[plt.Figure]:
        """
        Plot trade-off between training time and accuracy.

        Args:
            metric: Metric to use for accuracy ('mape', 'rmse', 'mae')
            save_fig: Whether to save figure

        Returns:
            Matplotlib figure object
        """
        df = self.get_results_dataframe()
        if df.empty:
            print("No results to plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        for model_name in df["model_name"].unique():
            model_data = df[df["model_name"] == model_name]
            ax.scatter(
                model_data["training_time"],
                model_data[metric],
                label=model_name,
                s=100,
                alpha=0.6,
            )

        ax.set_xlabel("Training Time (seconds)")
        ax.set_ylabel(f"{metric.upper()}")
        ax.set_title("Training Time vs Accuracy Trade-off")
        ax.legend()
        ax.grid(alpha=0.3)

        if save_fig:
            fig_path = self.output_dir / "benchmark_tradeoff.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {fig_path}")

        return fig

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report.

        Returns:
            Formatted report string
        """
        df = self.get_results_dataframe()
        if df.empty:
            return "No benchmark results available."

        report_lines = [
            "=" * 80,
            "BENCHMARK REPORT",
            "=" * 80,
            f"\nTotal Models Tested: {len(df['model_name'].unique())}",
            f"Total Datasets: {len(df['dataset_name'].unique())}",
            f"Total Runs: {len(df)}",
            "\n" + "-" * 80,
            "RANKINGS BY MAPE",
            "-" * 80,
        ]

        # Add rankings
        ranking = self.get_ranking("mape")
        report_lines.append(ranking.to_string(index=False))

        report_lines.extend(["\n" + "-" * 80, "SUMMARY STATISTICS", "-" * 80])

        # Add summary statistics
        summary = self.get_summary_statistics()
        report_lines.append(summary.to_string())

        report_lines.extend(["\n" + "-" * 80, "DETAILED RESULTS", "-" * 80])

        # Add detailed results
        for _, row in df.iterrows():
            report_lines.extend(
                [
                    f"\nModel: {row['model_name']} | Dataset: {row['dataset_name']}",
                    f"  MAPE: {row['mape']:.4f} | RMSE: {row['rmse']:.4f} | MAE: {row['mae']:.4f}",
                    f"  Training: {row['training_time']:.2f}s | Prediction: {row['prediction_time']:.4f}s",
                ]
            )

        report_lines.append("\n" + "=" * 80)

        report = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"\nReport saved to: {report_path}")

        return report


def load_benchmark_results(filepath: str) -> List[BenchmarkResult]:
    """
    Load benchmark results from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of BenchmarkResult objects
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    results = []
    for result_dict in data["results"]:
        results.append(BenchmarkResult(**result_dict))

    return results
