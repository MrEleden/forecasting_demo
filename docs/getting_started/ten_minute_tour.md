---
title: 10-Minute Quickstart
description: Guided walkthrough that takes you from cloning the repository to generating your first benchmark in ten minutes.
---

# 10-Minute Quickstart

This quickstart is purpose-built for new contributors who want to see the full workflow without reading every guide. The flow mirrors popular ML libraries (scikit-learn, PyTorch, Hugging Face) while showcasing the project's forecasting opinionated patterns.

## 0. Prerequisites (1 minute)

- Python 3.11 installed.
- Git CLI available.
- Optional: Docker Desktop running if you plan to explore containerisation later.

Clone the repository and enter the workspace:

```powershell
git clone https://github.com/MrEleden/forecasting_demo.git
cd forecasting_demo
```

## 1. Create and activate the environment (2 minutes)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements-ml.txt
pip install -e .
```

> **Tip:** Use `pip install -r requirements-dev.txt` when developing documentation or running tests locally.

## 2. Download sample data (1 minute)

```powershell
python src/ml_portfolio/scripts/download_all_data.py --dataset walmart
```

This command fetches the weekly Walmart sales data and stores it under `projects/retail_sales_walmart/data/raw/`.

## 3. Generate deterministic features (1 minute)

```python
from ml_portfolio.data.preprocessing import StaticTimeSeriesPreprocessingPipeline
from ml_portfolio.data.dataset_factory import DatasetFactory

pipeline = StaticTimeSeriesPreprocessingPipeline(
    date_column="Date",
    target_column="Weekly_Sales",
    group_columns=["Store", "Dept"],
    lag_features=[1, 2, 4, 52],
    rolling_windows=[4, 8, 52],
    cyclical_features=["month", "week"],
)

factory = DatasetFactory(
    data_path="projects/retail_sales_walmart/data/raw/Walmart.csv",
    target_column="Weekly_Sales",
    static_feature_engineer=pipeline,
)
train_ds, val_ds, test_ds = factory.create_datasets()
```

The `DatasetFactory` handles temporal splits and applies the static feature pipeline before splitting to eliminate leakage.

## 4. Train your first model (2 minutes)

```python
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
from ml_portfolio.evaluation.metrics import MAPEMetric

X_train, y_train = train_ds.get_data()
X_val, y_val = val_ds.get_data()

model = LightGBMForecaster(random_state=42)
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
mape = MAPEMetric()(y_val, val_predictions)
print(f"Validation MAPE: {mape:.2f}%")
```

You should see a baseline MAPE in the low 30s. Add more features or tune hyperparameters to improve the score.

## 5. Record the run in MLflow (2 minutes)

```python
import mlflow

mlflow.set_experiment("walmart_lightgbm_baselines")
with mlflow.start_run(run_name="lightgbm_baseline"):
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("validation_mape", float(mape))
    mlflow.sklearn.log_model(model.model, "model")
```

Launch the MLflow UI to inspect the run:

```powershell
mlflow ui --port 5000
```

Open `http://localhost:5000` in your browser to compare metrics and artefacts.

## 6. Run a benchmark sweep (1 minute)

```powershell
python -m ml_portfolio.training.train dataset=walmart -m model=lightgbm,catboost,xgboost
```

Hydra's multirun (`-m`) launches each model configuration and stores results in `outputs/`. The [benchmark guide](../BENCHMARK.md) explains how to visualise the results and export summaries.

## 7. Surface results in the dashboard (optional)

```powershell
streamlit run src/ml_portfolio/dashboard/app.py
```

Switch the sidebar data source to **JSON (Cache)** to load the benchmark artefacts generated in the previous step.

## 8. Next steps

- Follow the [model selection playbook](../guides/model_selection.md) to pick the right architecture for your dataset.
- Read the [experiment tracking guide](../guides/experiment_tracking.md) to standardise MLflow usage.
- Explore the new example notebooks under `examples/notebooks/`.
- Contribute documentation or code by following the [CONTRIBUTING](../../CONTRIBUTING.md) instructions.

You now have the end-to-end experience under your belt—happy forecasting! 🎯
