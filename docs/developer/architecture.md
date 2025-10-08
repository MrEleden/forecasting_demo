---
title: Architecture & Design Patterns
description: Technical overview of the shared forecasting library, project scaffolding, and cross-cutting design patterns.
---

# Architecture & Design Patterns

This document explains how the forecasting portfolio is structured and the design choices that keep shared components reusable across multiple domains.

## 1. High-level layout

```
├── src/ml_portfolio/        # Shared, reusable Python package
│   ├── data/                # Datasets, loaders, preprocessing, validation
│   ├── models/              # Statistical, ML, and deep learning forecasters
│   ├── training/            # Training engines, callbacks, tuning utilities
│   ├── evaluation/          # Metrics, losses, benchmarking helpers
│   └── utils/               # Config, IO, registry, logging utilities
├── projects/                # Domain demos (retail, rideshare, inventory, TSI)
│   ├── data/                # Project-specific raw/interim/processed datasets
│   ├── scripts/             # Data acquisition (download/generate)
│   ├── notebooks/           # Exploratory analysis and feature design
│   ├── models/              # Stored artefacts, checkpoints, mlflow exports
│   └── api/app/             # Optional serving layers (FastAPI, Streamlit)
├── docs/                    # Markdown documentation + Sphinx config
├── tests/                   # Unit/integration/regression tests
└── scripts/                 # Repo-level automation (cleanup, setup)
```

## 2. Core patterns

### 2.1 Configuration-first (Hydra)

- Every experiment is described in YAML configs under `src/ml_portfolio/conf/`.
- Overrides on the CLI (`python -m ml_portfolio.training.train model=lightgbm`) keep notebooks reproducible.
- Config composition mirrors scikit-learn's pipeline philosophy—models, datasets, and optimizers are interchangeable.

### 2.2 Deterministic preprocessing

- Static feature engineering (`StaticTimeSeriesPreprocessingPipeline`) always happens before splitting to avoid temporal leakage.
- Statistical transforms (scalers) belong to the statistical pipeline applied after splitting, ensuring train-only fitting.
- Data validation via Pandera runs immediately after ingestion for consistent schemas.

### 2.3 Model abstraction layers

- Statistical forecasters inherit from `StatisticalForecaster` and expose sklearn-like `fit/predict` APIs.
- PyTorch forecasters extend `PyTorchForecaster`, sharing callbacks, optimizer setup, and checkpointing logic.
- Registry (`ml_portfolio.models.registry`) loads models by name/version for dashboards and APIs.

### 2.4 Training orchestration

- `StatisticalEngine` performs single-step fits with optional metric logging and checkpointing.
- `PyTorchEngine` handles epoch loops, gradient clipping, early stopping, and schedules.
- Engines are configured via Hydra to toggle MLflow tracking, Optuna sweeps, and GPU/CPU selection.

### 2.5 Benchmarking workflow

- `ModelBenchmark` runs multiple models/datasets, aggregates metrics, and writes structured artefacts.
- Benchmarks feed the Streamlit dashboard or MLflow for reporting.
- CI can execute lightweight benchmarks to prevent regressions.

## 3. Cross-cutting concerns

| Concern               | Implementation                                             | Notes                                                               |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------- |
| Experiment tracking   | MLflow + Hydra loggers                                     | Environment variable `MLFLOW_TRACKING_URI` toggles local vs remote. |
| Hyperparameter tuning | Optuna integration (`run_optimization.py`)                 | Supports distributed workers via RDB storage.                       |
| Testing               | `pytest`, fixtures in `tests/conftest.py`, coverage gating | Separate unit, integration, regression suites.                      |
| Documentation         | Markdown + Sphinx + MyST                                   | Sources live next to code; CI validates builds.                     |
| Deployment            | Docker Compose + FastAPI + Streamlit                       | Multi-service stack for API, dashboard, MLflow.                     |

## 4. Extensibility guidelines

- Keep reusable code in `src/ml_portfolio/`; project-specific hacks belong in `projects/<domain>/`.
- Prefer dependency injection via configs rather than hard-coded imports.
- Write abstract base classes for recurring patterns (e.g., new data loaders or callback types).
- Use registry patterns when exposing artefacts to dashboards or APIs.

## 5. Sequence diagrams

### 5.1 Training flow

```
User CLI -> Hydra config -> DatasetFactory -> Training Engine -> Model -> Metrics/MLflow -> Artefacts
```

### 5.2 Benchmark flow

```
Benchmark config -> ModelBenchmark -> (train model -> evaluate -> collect metrics) x N -> JSON/MLflow -> Dashboard
```

## 6. Related documentation

- [Model Selection Playbook](../guides/model_selection.md)
- [Adding a New Model](adding_model.md)
- [Testing Best Practices](testing_best_practices.md)
- [10-Minute Quickstart](../getting_started/ten_minute_tour.md)

Keep this document up to date whenever new subsystems (e.g., feature stores, streaming pipelines) are added to the portfolio.
