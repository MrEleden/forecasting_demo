---
title: Testing Best Practices
description: Guidelines for designing, organising, and maintaining automated tests across the forecasting portfolio.
---

# Testing Best Practices

This guide complements the high-level [Testing Strategy](../guides/testing_strategy.md) by focusing on implementation details for developers.

## 1. Test layout

```
tests/
├── unit/                # Fast tests for individual modules
│   └── data/            # Example: transforms, validators
├── integration/         # Compose multiple modules (training + metrics)
├── regression/          # Snapshot/fixture-based comparisons
└── conftest.py          # Shared fixtures
```

Keep tests close to the modules they cover. Mirror the structure of `src/ml_portfolio/` when possible.

## 2. Writing deterministic tests

- Seed randomness (`np.random.seed`, `torch.manual_seed`) in fixtures.
- Use the synthetic fixtures provided in `tests/conftest.py` to avoid brittle data dependencies.
- When testing Optuna sweeps, set `n_trials=2` and `timeout=5` to keep runtime low.

## 3. Assertions and diagnostics

- Prefer `numpy.testing` helpers for array comparisons (e.g., `assert_allclose`).
- Log intermediate shapes and metrics when debugging; wrap assertions with helpful messages.
- For floating-point metrics (MAPE, RMSE), assert within tolerances rather than exact equality.

## 4. Parametrise aggressively

```python
import pytest
from ml_portfolio.evaluation.metrics import MAPEMetric, RMSEMetric

@pytest.mark.parametrize("metric_cls", [MAPEMetric, RMSEMetric])
def test_metrics_handle_numpy(metric_cls, dummy_series):
    metric = metric_cls()
    y_true = dummy_series.values
    y_pred = y_true * 0.95
    assert metric(y_true, y_pred) >= 0.0
```

Parametrisation keeps suites concise while hitting multiple code paths.

## 5. Mocking external services

- Use `unittest.mock.patch` or dedicated fixtures to stub MLflow, S3, or API clients.
- For dashboard tests, mock file IO instead of relying on real JSON artefacts.
- When testing FastAPI endpoints, use `TestClient` with in-memory registries.

## 6. Coverage expectations

- Target ≥85% overall coverage; critical modules (`benchmark.py`, `train.py`) should exceed 90%.
- Use `pytest --cov=ml_portfolio --cov-report=term-missing` before submitting PRs.
- Investigate uncovered lines; add unit tests or mark legacy code paths for refactoring.

## 7. Continuous integration hooks

- CI runs `pytest` and `pre-commit`; ensure both pass locally.
- Slow tests can be marked with `@pytest.mark.slow` and excluded via `pytest -m "not slow"` for quick iterations.
- Long-running notebooks or benchmarks should be covered by separate scheduled jobs, not the default CI pipeline.

## 8. Updating tests after refactors

- Adjust fixtures and expected artefacts when data schemas change.
- Update snapshot baselines through a dedicated commit to review differences explicitly.
- Document major test changes in `CHANGELOG.md` if they alter contributor workflows.

Investing in deterministic, well-structured tests keeps the portfolio reliable as new models and datasets are added.
