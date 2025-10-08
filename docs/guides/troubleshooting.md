---
title: Troubleshooting Guide
description: Common issues encountered when working with the forecasting portfolio and how to resolve them quickly.
---

# Troubleshooting Guide

Use this guide when something goes wrong—whether it is installation, training, benchmarking, or documentation builds. Problems are grouped into categories with clear fix steps.

## 1. Environment and installation

### `ImportError: No module named 'ml_portfolio'`

- Ensure the repo is installed in editable mode: `pip install -e .`
- Verify you activated the correct virtual environment (`.\.venv\Scripts\Activate` on Windows).

### `ImportError: lightgbm` / `catboost` / `xgboost`

- Install optional model dependencies via `pip install -r requirements-models.txt`.
- Confirm the package versions match the requirements file to avoid ABI mismatches.

### `ModuleNotFoundError: linkify`

- Install documentation extras: `pip install -r requirements-dev.txt`.
- Re-run `sphinx-build docs docs/_build/html` to confirm the issue is resolved.

## 2. Data and preprocessing (Hydra or scripts)

### `FileNotFoundError: Walmart.csv`

- Download datasets before training: `python src/ml_portfolio/scripts/download_all_data.py --dataset walmart`.
- If running inside Docker, mount the `projects/retail_sales_walmart/data/` directory.

### Misaligned feature counts between train/val/test splits

- Check for non-deterministic transformations—static features must be created **before** splitting.
- Use the shared `StaticTimeSeriesPreprocessingPipeline`; avoid ad-hoc feature engineering inside notebooks.

### `ValueError: Found input variables with inconsistent numbers of samples`

- Confirm that you are using `dataset.get_data()` (returns aligned `(X, y)` pairs).
- Printing `len(dataset)` and the shapes of `X`/`y` helps detect slicing mistakes.

## 3. Training and benchmarking

### Hydra run fails with `Config attribute not found`

- List available configs: `python -m ml_portfolio.training.train --help`.
- Ensure you spelled overrides correctly (e.g., `dataset=walmart`, `model=lightgbm`).

### Training crashes due to GPU/Torch mismatch

- Fallback to CPU by overriding the device: `python -m ml_portfolio.training.train trainer.device=cpu`.
- Alternatively, uninstall or update the conflicting CUDA toolkit version.

### `optuna.exceptions.StorageInternalError`

- Delete the old SQLite study: remove `optuna.db` or switch storage: `hydra.run.dir=outputs/optuna/<date>`.
- When using RDB storage, verify your credentials and network connectivity.

### Benchmark script reports `No results found`

- Confirm each model wrote results to `results/benchmarks/` (check for JSON files).
- Run the benchmark again with logging enabled: `python -m ml_portfolio.training.train dataset=walmart -m model=lightgbm,catboost -p hydra.job.chdir=false`.

## 4. Dashboard and API

### Streamlit app cannot load MLflow data

- Ensure `MLFLOW_TRACKING_URI` points to the live tracking server.
- Use the **JSON (Cache)** option to load `results/benchmarks/*.json` files instead.

### FastAPI returns `503` or `Model not loaded`

- Verify the model artefact path in the project-specific `api/config.py`.
- Check model version compatibility; re-export models after major refactors.

## 5. Testing and CI

### `pytest` fails due to missing test data

- Use the synthetic fixtures in `tests/conftest.py` instead of referencing raw datasets directly.
- For deterministic seeds, run `pytest --maxfail=1 --disable-warnings` to narrow down flaky tests.

### Pre-commit formatting errors

- Run `pre-commit run --all-files` to auto-fix style issues.
- Install the hook first: `pre-commit install`.

## 6. Documentation builds

### Sphinx build warnings about missing documents

- Ensure every Markdown file appears in a toctree (update `docs/index.md`).
- Remove or update cross-references pointing at deleted docs.

### Code blocks fail to highlight (`misc.highlighting_failure`)

- Prefix code fences with the correct language (`python`, `powershell`, `yaml`).
- Avoid tab characters in YAML snippets; replace tabs with two spaces.

## 7. When all else fails

- Search existing issues: <https://github.com/MrEleden/forecasting_demo/issues>
- Ask on the repository discussions (coming soon) or open a new issue with logs, commands, and environment info.
- Tag maintainers if production-facing functionality is blocked.
