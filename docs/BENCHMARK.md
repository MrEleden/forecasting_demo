---
title: Benchmark Suite Guide
description: How to run, persist, and integrate benchmarking experiments in the ML forecasting portfolio.
---

# Benchmark Suite Guide

# Benchmark Suite Guide

## Purpose

The benchmark suite provides a unified way to compare classical, gradient boosting, and deep learning forecasters on shared datasets. It standardizes metrics, captures timing information, and produces analysis artifacts that feed the dashboard and reporting layers of the portfolio.

## Capabilities at a Glance

- Run a battery of models against one or many datasets with consistent train/test splits.
- Capture accuracy metrics (MAPE, RMSE, MAE) alongside wall-clock training and inference time.
- Persist results as structured JSON, human-readable reports, and ready-to-embed visualizations.
- Plug saved benchmark results into the Streamlit dashboard or downstream analytics notebooks.

## Prerequisites

1. **Activate the project environment** (PowerShell example):

   ```powershell
   .\.venv\Scripts\activate
   ```

1. **Ensure data is available.** Download project datasets once with:

   ```powershell
   python src/ml_portfolio/scripts/download_all_data.py --dataset walmart
   ```

   Update `--dataset` to `ola`, `inventory`, `tsi`, or `all` as needed.

1. **Install optional model backends** if you want to benchmark them. Packages such as `lightgbm`, `catboost`, and `xgboost` are listed in `requirements-ml.txt` and `requirements-models.txt`.

## Workflow Overview

1. Prepare a dataset split (train/validation/test) using the reusable factories in `ml_portfolio.data` or project-specific loaders.
1. Instantiate the models you want to compare. Every model must expose `fit(X, y)` and `predict(X)`.
1. Use `ModelBenchmark` to execute single-model or multi-model runs on each dataset split.
1. Persist outputs (`benchmark_results.json`, plots, reports) to `results/benchmarks` or a custom directory.
1. Inspect the generated artifacts directly, ship them to MLflow, or surface them in the dashboard.

## Quickstart – Python API

```python
import pandas as pd

from ml_portfolio.data.dataset_factory import DatasetFactory
from ml_portfolio.evaluation.benchmark import ModelBenchmark
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
from ml_portfolio.models.statistical.random_forest import RandomForestForecaster


def dataset_to_frame(dataset, target_name="Weekly_Sales"):
	X, y = dataset.get_data()
	columns = dataset.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
	features = pd.DataFrame(X, columns=columns)
	target = pd.Series(y, name=target_name)
	return features, target


# 1. Load and split data
factory = DatasetFactory(
	data_path="projects/retail_sales_walmart/data/processed/walmart_sales.csv",
	target_column="Weekly_Sales",
	timestamp_column="Date",
)

train_ds, val_ds, test_ds = factory.create_datasets()
X_train, y_train = dataset_to_frame(train_ds)
X_val, y_val = dataset_to_frame(val_ds)
X_test, y_test = dataset_to_frame(test_ds)

# Optional: fold validation into training for the final fit
X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

# 2. Define models (remove LightGBM if the package is unavailable)
models = {
	"random_forest": RandomForestForecaster(n_estimators=300, max_depth=12),
	"lightgbm": LightGBMForecaster(n_estimators=400, learning_rate=0.05),
}

# 3. Run the benchmark
benchmark = ModelBenchmark(output_dir="results/benchmarks")
benchmark.run_multiple_models(
	models=models,
	X_train=X_train_full,
	y_train=y_train_full,
	X_test=X_test,
	y_test=y_test,
	dataset_name="walmart",
)

# 4. Analyse and persist artifacts
results_df = benchmark.get_results_dataframe()
print(results_df[["model_name", "mape", "rmse", "mae", "training_time", "prediction_time"]])

benchmark.save_results("walmart_gbdt_vs_rf.json")
benchmark.generate_report()
benchmark.plot_comparison(metric="mape")
benchmark.plot_training_time_vs_accuracy(metric="mape")
```

Notes:

- Metrics are reported as decimals (`0.082` ≈ `8.2%` MAPE). Multiply by 100 if you prefer percentage formatting.
- Each figure is written to the configured `output_dir` (default `results/benchmarks`).
- If an optional backend such as LightGBM is missing, install it or remove that entry from the `models` dictionary.

## Multi-Dataset Experiments

```python
datasets = {
	"walmart": {
		"data_path": "projects/retail_sales_walmart/data/processed/walmart_sales.csv",
		"target": "Weekly_Sales",
		"timestamp": "Date",
	},
	"tsi": {
		"data_path": "projects/transportation_tsi/data/processed/tsi_data.csv",
		"target": "target",          # Update to match your column name
		"timestamp": "timestamp",    # Update to match your column name
	},
}

for name, cfg in datasets.items():
	factory = DatasetFactory(
		data_path=cfg["data_path"],
		target_column=cfg["target"],
		timestamp_column=cfg["timestamp"],
	)

	train_ds, val_ds, test_ds = factory.create_datasets()
	X_train, y_train = dataset_to_frame(train_ds, target_name=cfg["target"])
	X_val, y_val = dataset_to_frame(val_ds, target_name=cfg["target"])
	X_test, y_test = dataset_to_frame(test_ds, target_name=cfg["target"])

	benchmark.run_multiple_models(
		models=models,
		X_train=pd.concat([X_train, X_val], ignore_index=True),
		y_train=pd.concat([y_train, y_val], ignore_index=True),
		X_test=X_test,
		y_test=y_test,
		dataset_name=name,
	)

benchmark.save_results()
```

Running multiple datasets in the same session allows `ModelBenchmark` to produce consolidated rankings and statistics across domains. Replace the placeholder `target` and `timestamp` keys with the actual column names from each dataset.

## Output Artifacts

| Artifact                            | Default location      | Description                                                                            |
| ----------------------------------- | --------------------- | -------------------------------------------------------------------------------------- |
| `benchmark_results.json`            | `results/benchmarks/` | Structured list of `BenchmarkResult` payloads with metrics, timings, and metadata.     |
| `benchmark_report.txt`              | `results/benchmarks/` | Text report containing rankings, summary statistics, and per-model breakdowns.         |
| `benchmark_comparison_<metric>.png` | `results/benchmarks/` | Horizontal bar chart and box plot for the chosen metric (`mape`, `rmse`, `mae`).       |
| `benchmark_tradeoff.png`            | `results/benchmarks/` | Scatter plot showing training time versus accuracy to highlight efficiency trade-offs. |

## Inspecting Metrics Programmatically

```python
ranking = benchmark.get_ranking(metric="mape")
summary = benchmark.get_summary_statistics()

print("Ranking by MAPE:\n", ranking)
print("\nSummary statistics:\n", summary)
```

- `get_ranking` sorts models by the average of the selected metric.
- `get_summary_statistics` returns mean, standard deviation, minimum, and maximum values per model.

## Visualization Helpers

- `benchmark.plot_comparison(metric="mape")` compares models on the selected metric and saves a bar/box plot composite.
- `benchmark.plot_training_time_vs_accuracy(metric="mape")` highlights the speed–accuracy trade-off; models close to the lower-left corner are both fast and accurate.

## Loading Saved Results Later

```python
from ml_portfolio.evaluation.benchmark import load_benchmark_results

results = load_benchmark_results("results/benchmarks/walmart_gbdt_vs_rf.json")
for entry in results:
	print(entry.model_name, entry.mape, entry.training_time)
```

The helper reconstructs `BenchmarkResult` objects from disk, allowing you to hydrate past experiments for comparison, plotting, or dashboard ingestion.

## Integrations

- **Dashboard**: The Streamlit app (`src/ml_portfolio/dashboard/app.py`) loads JSON files from `results/benchmarks` when live MLflow data is not available.
- **MLflow**: Use the workflow documented in `docs/MLFLOW_INTEGRATION.md` to sync benchmark metrics to the tracking server.
- **Hydra pipelines**: Combine `ModelBenchmark` with Hydra configs to orchestrate repeatable experiments (see `src/ml_portfolio/conf`).

### Hydra & CLI automation

To script recurring benchmark runs, wrap `ModelBenchmark` inside a small Hydra-driven entry point. Save the following as `src/ml_portfolio/scripts/run_benchmark_hydra.py` (or similar):

```python
import pandas as pd
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ml_portfolio.data.dataset_factory import DatasetFactory


@hydra.main(config_path="../conf", config_name="benchmark/lightgbm_vs_catboost", version_base="1.3")
def main(cfg: DictConfig) -> None:
		factory: DatasetFactory = instantiate(cfg.dataset_factory)
		train_ds, val_ds, test_ds = factory.create_datasets()

		def to_frame(dataset):
				X, y = dataset.get_data()
				features = pd.DataFrame(X, columns=dataset.feature_names)
				target = pd.Series(y, name=cfg.target_column)
				return features, target

		X_train, y_train = to_frame(train_ds)
		X_test, y_test = to_frame(test_ds)

		benchmark = instantiate(cfg.benchmark)
		models = {name: instantiate(model_cfg) for name, model_cfg in cfg.models.items()}
		benchmark.run_multiple_models(models, X_train, y_train, X_test, y_test, dataset_name=cfg.dataset_name)
		benchmark.save_results()
		benchmark.generate_report()


if __name__ == "__main__":
		main()
```

Example Hydra configuration (`src/ml_portfolio/conf/benchmark/lightgbm_vs_catboost.yaml`):

```yaml
defaults:
	- dataset_factory: walmart

target_column: Weekly_Sales
dataset_name: walmart

benchmark:
	_target_: ml_portfolio.evaluation.benchmark.ModelBenchmark
	output_dir: results/benchmarks/hydra

models:
	lightgbm:
		_target_: ml_portfolio.models.statistical.lightgbm.LightGBMForecaster
		n_estimators: 400
		learning_rate: 0.05
	catboost:
		_target_: ml_portfolio.models.statistical.catboost.CatBoostForecaster
		iterations: 600
		learning_rate: 0.05
```

Run the job from the project root:

```bash
python src/ml_portfolio/scripts/run_benchmark_hydra.py \
	dataset_factory.data_path=projects/retail_sales_walmart/data/raw/Walmart.csv
```

Hydra handles overrides (e.g., swapping datasets or model parameters) and captures outputs per run directory.

## Best Practices

- Keep feature engineering deterministic and backward-looking before splitting to avoid leakage.
- Log the `params` argument when calling `run_benchmark` so hyperparameters appear in the stored results.
- Version your benchmark outputs or include a timestamped filename when calling `save_results` to avoid accidental overwrites.
- Pair benchmark runs with Optuna sweeps (see `src/ml_portfolio/scripts/run_optimization.py`) when exploring large hyperparameter spaces.

## Troubleshooting

| Issue                                                | Likely cause                     | Resolution                                                                                               |
| ---------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'lightgbm'`    | Optional backend not installed   | Install missing package inside `.venv` or remove the model from `models`.                                |
| `ValueError: Target column 'Weekly_Sales' not found` | Dataset path or schema mismatch  | Verify the CSV path and ensure preprocessing created the expected column names.                          |
| Empty plots or `No results to plot`                  | No successful benchmark runs yet | Confirm that `run_benchmark` returned results and that `benchmark.results` is not empty before plotting. |
| Extremely large MAPE values                          | Zero or near-zero targets        | Switch to RMSE/MAE, add a floor to the target, or filter zero-demand periods.                            |

## Next Steps

1. Expand coverage with additional model wrappers or deep learning blocks in `src/ml_portfolio/models`.
1. Add temporal cross-validation strategies (rolling windows, blocked splits) around the benchmark entry point.
1. Wire results into automated regression checks in CI (for example, compare against a stored baseline JSON).
1. Enrich the generated report with statistical tests (Diebold–Mariano, paired t-tests) to quantify improvements.
1. Expose a thin CLI wrapper if repeated non-notebook runs are required; the current Python API supports scripting directly.

## References

- `src/ml_portfolio/evaluation/benchmark.py` – Source of `ModelBenchmark` and helper utilities.
- `docs/DASHBOARD.md` – How benchmark outputs appear in the dashboard.
- `docs/MLFLOW_INTEGRATION.md` – Steps for pushing benchmark data to MLflow.
- `src/ml_portfolio/models/statistical/` – Available baseline model wrappers used in benchmarks.
