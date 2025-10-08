---
title: Experiment Tracking Playbook
description: Standard operating procedure for logging and analysing experiments with MLflow across the forecasting portfolio.
---

# Experiment Tracking Playbook

Consistent experiment logging makes it easy to compare models across projects. This guide summarises how to use MLflow with the shared tooling in this repo.

## 1. Tracking directory layout

- Local runs are stored under `mlruns/` in the repository root.
- Each project can map to its own MLflow experiment (e.g., `walmart_sales_forecasting`, `ola_demand_forecasting`).
- Remote tracking servers are optional—point `MLFLOW_TRACKING_URI` to switch.

```powershell
# Example: log runs to a remote server
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python src/ml_portfolio/scripts/run_optimization.py --trials 20
```

## 2. Logging conventions

| Item            | Recommendation                                                                    |
| --------------- | --------------------------------------------------------------------------------- |
| Experiment name | `<project>_<objective>` (e.g., `walmart_sales_forecasting`)                       |
| Run name        | `<model>-<timestamp>` or `<model>-<hydra-job-id>`                                 |
| Parameters      | Log hyperparameters using `mlflow.log_params(cfg.model)` or explicit dictionaries |
| Metrics         | Use portfolio metrics: `mape`, `rmse`, `mae`, `directional_accuracy`              |
| Artefacts       | Save benchmark JSONs, plots, and trained models for later comparison              |

## 3. Using the optimisation script

`src/ml_portfolio/scripts/run_optimization.py` already integrates MLflow logging when `--with-mlflow` is enabled in the Hydra config.

```powershell
python src/ml_portfolio/scripts/run_optimization.py --trials 50 hydra.run.dir=outputs/optuna/walmart
```

- Override `mlflow.experiment_name` in the config to target a specific experiment.
- Optuna trials log best metrics; push final benchmark artefacts using `ModelBenchmark.save_results()`.

## 4. Linking with the benchmark suite

After running `ModelBenchmark`, log the consolidated results back to MLflow:

```python
import mlflow
from ml_portfolio.evaluation.benchmark import ModelBenchmark

mlflow.set_experiment("walmart_sales_forecasting")
with mlflow.start_run(run_name="lightgbm_vs_catboost"):
    benchmark = ModelBenchmark(output_dir="results/benchmarks/mlflow")
    # ... run benchmarks ...
    path = benchmark.save_results("walmart_lightgbm_vs_catboost.json")
    mlflow.log_artifact(path)
    mlflow.log_metric("best_mape", benchmark.get_ranking().iloc[0]["avg_mape"])
```

## 5. Dashboard integration

The Streamlit dashboard (`src/ml_portfolio/dashboard/app.py`) prioritises live MLflow data when available:

1. Select "MLflow (Live)" as the data source.
1. Provide the experiment name via the sidebar or environment variables.
1. When MLflow is unavailable, it falls back to the JSON artefacts saved under `results/benchmarks`.

## 6. Hygiene checklist

- Clear stale runs regularly: `mlflow gc --older-than 30d` for local stores.
- Record MLflow server credentials in environment variables, not config files.
- Sync benchmark reports (JSON + plots) as artefacts after major experiments so the dashboard stays current.
- Version control configuration files used for experiments to make results reproducible.

Following this playbook keeps experiment history tidy and makes it simple to compare new candidates against previous baselines.
