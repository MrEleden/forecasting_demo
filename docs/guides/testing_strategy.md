---
title: Testing Strategy
description: Consistent approach for unit, integration, and regression tests across the forecasting portfolio.
---

# Testing Strategy

This guide outlines the testing layers we expect for every project in the portfolio and how to wire them into the shared tooling.

## 1. Pyramid overview

| Layer             | Scope                                          | Example                                        |
| ----------------- | ---------------------------------------------- | ---------------------------------------------- |
| Unit tests        | Smallest components (data transforms, metrics) | `tests/unit/test_metrics.py`                   |
| Integration tests | Compose multiple modules, respect configs      | `tests/integration/test_benchmark_pipeline.py` |
| Regression tests  | End-to-end assertions on saved artefacts       | `tests/regression/test_dashboard_snapshots.py` |

All tests run via `pytest`. Use markers (`@pytest.mark.integration`) to separate slower suites.

## 2. Shared fixtures

`tests/conftest.py` already exposes reusable fixtures:

- `sample_config`: Hydra config overrides for synthetic data.
- `dummy_series`: Pandas `Series` with seasonal patterns.
- `tmp_model_dir`: Temporary directory for model artefacts.

Use these fixtures to avoid bespoke scaffolding in individual tests.

## 3. Data validation utilities

The `ml_portfolio.data.validation` module provides helper functions:

```python
from ml_portfolio.data.validation import validate_timeseries_frame

def test_validate_schema(dummy_frame):
    validate_timeseries_frame(dummy_frame)
```

Extend `validation.py` with project-specific checks if needed (e.g., positive demand constraints).

## 4. Benchmark pipeline tests

Integration tests should cover the benchmark lifecycle:

```python
from ml_portfolio.evaluation.benchmark import ModelBenchmark

def test_benchmark_runs(sample_config, tmp_path):
    benchmark = ModelBenchmark(output_dir=tmp_path)
    benchmark.run_from_config(sample_config)
    report = benchmark.get_ranking()
    assert not report.empty
```

- Use lightweight models (e.g., statistical baselines) to keep runtime under 30 seconds.
- For Optuna integration, limit trials via config overrides (`optimizer.max_trials=2`).

## 5. Regression safeguards

When dashboards or APIs change, capture snapshots:

1. Generate JSON responses or HTML snapshots under `tests/data/snapshots/`.
1. Compare with the latest output; update snapshots only after reviewing differences.
1. Use `pytest`'s approval plugins if full HTML diffs are needed.

## 6. Coverage expectations

- Maintain overall coverage above 85% (tracked via `coverage.xml`).
- Critical modules (`evaluation/benchmark.py`, `models/registry.py`) should exceed 90%.
- Run the suite with coverage before merging:

```powershell
pytest --cov=ml_portfolio --cov-report=term-missing
```

## 7. Continuous integration

The GitHub Actions workflow runs unit and integration suites on pull requests.

- Use `pytest -m "not slow"` locally for quick checks.
- Add a `slow` marker to long-running experiments and gate them in CI via matrix strategies.

## 8. Hygiene checklist

- Keep test data minimal and synthetic; real datasets live under `projects/<name>/data/`.
- Mock external services (MLflow, S3) using fixtures; never require live credentials in CI.
- Ensure every bug fix includes a regression test in the most appropriate layer.

Adopting this strategy ensures reliable, fast feedback loops while preserving confidence in shared forecasting components.
