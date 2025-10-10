scripts\\demo_optuna_showcase.bat  # Windows

# � ML Forecasting Portfolio

[![CI](https://github.com/MrEleden/forecasting_demo/actions/workflows/ci.yml/badge.svg)](https://github.com/MrEleden/forecasting_demo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-Sphinx-green.svg)](docs/index.md)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)

A production-grade portfolio that showcases reusable time-series forecasting components, domain demos, and MLOps workflows. The project separates a shared library (`src/ml_portfolio/`) from project-specific assets in `projects/`, enabling fast experimentation without duplicating code.

______________________________________________________________________

## 🧭 Overview

- **Models**: Statistical baselines, gradient boosting, deep learning forecasters, and ensembles
- **Domains**: Retail sales, rideshare demand, inventory planning, transportation indices
- **Tooling**: Hydra configs, Optuna optimization, MLflow tracking, Streamlit dashboards, Dockerized deployment

Read more in the [architecture guide](docs/ARCHITECTURE_REFACTOR_COMPLETE.md).

______________________________________________________________________

## ⚡ Quick Start

### 1. Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements-ml.txt
pip install -e .
```

### 2. Fetch example data

```powershell
python src/ml_portfolio/scripts/download_all_data.py --dataset walmart
```

### 3. Train your first model

```powershell
python -m ml_portfolio.training.train dataset=walmart model=lightgbm
```

### 4. View the portfolio dashboard

```powershell
streamlit run src/ml_portfolio/dashboard/app.py
```

Open http://localhost:8501 to explore the narrative-driven portfolio with 6 tabs covering models, experiments, benchmarks, and engineering/data science perspectives.

### 5. Minimal Python snippet

```python
from ml_portfolio.data.dataset_factory import DatasetFactory
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
from ml_portfolio.evaluation.metrics import MAPEMetric

factory = DatasetFactory(
	data_path="projects/retail_sales_walmart/data/raw/Walmart.csv",
	target_column="Weekly_Sales",
)
train_ds, val_ds, _ = factory.create_datasets()

X_train, y_train = train_ds.get_data()
X_val, y_val = val_ds.get_data()

model = LightGBMForecaster(random_state=42)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)
mape = MAPEMetric()(y_val, val_pred)
print(f"Validation MAPE: {mape:.2f}%")
```

______________________________________________________________________

## 📊 Benchmark Snapshot (2025-10)

| Dataset            | Top Model          | Metric | Score     |
| ------------------ | ------------------ | ------ | --------- |
| Walmart Retail     | LightGBMForecaster | MAPE   | **7.1%**  |
| Ola Rideshare      | CatBoostForecaster | RMSE   | **162.4** |
| Inventory Planning | SARIMAXForecaster  | MAE    | **38.2**  |
| Transportation TSI | LSTMForecaster     | RMSE   | **21.5**  |

Full methodology and configuration details live in [docs/BENCHMARK.md](docs/BENCHMARK.md).

______________________________________________________________________

## 📚 Documentation Map

The documentation site can be built locally with:

```powershell
.\.venv\Scripts\sphinx-build docs docs/_build
```

______________________________________________________________________

## 🧱 Repository Layout

```
├── src/ml_portfolio/    # Shared library (data, models, training, evaluation)
├── projects/            # Domain demos with their own configs and assets
├── docs/                # Markdown documentation + Sphinx configuration
├── tests/               # Unit, integration, and regression suites
├── scripts/             # CLI entry points for training, benchmarking, ops
└── docs/conf.py        # Sphinx documentation configuration
```

______________________________________________________________________

## 🌟 Core Capabilities

1. **Config-driven experiments** with Hydra and Optuna sweeps
1. **Production telemetry** via MLflow tracking and registry integrations
1. **Reusable data tooling** for windowing, transforms, and validation
1. **Model zoo** spanning statistical, gradient boosting, and neural architectures
1. **Visualization stack** with Streamlit dashboards and Matplotlib reporting

______________________________________________________________________

## � Contributing

We welcome improvements! Review the [contribution guide](CONTRIBUTING.md) for branching strategy, quality gates, and documentation standards. See the [roadmap](CHANGELOG.md) for upcoming work.

______________________________________________________________________

## � Support & Discussion

- File issues on [GitHub](https://github.com/MrEleden/forecasting_demo/issues)
- Join discussions in the repository discussions tab (coming soon)
- Share ideas via pull requests—each one is reviewed with a focus on reproducibility and documentation

______________________________________________________________________

Made with ❤️ for showcasing end-to-end forecasting best practices.
