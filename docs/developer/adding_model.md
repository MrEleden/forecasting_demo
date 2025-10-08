---
title: Adding a New Model
description: Step-by-step checklist for integrating a new forecasting model into the shared library and downstream tooling.
---

# Adding a New Model

Follow this playbook to introduce a new model to the portfolio without breaking existing pipelines. The process mirrors contributions in scikit-learn and PyTorch Forecasting, emphasising tests, documentation, and configuration wiring.

## 1. Choose the right abstraction

- **Statistical / Gradient Boosting** → inherit from `ml_portfolio.models.statistical.StatisticalForecaster`.
- **PyTorch-based** → inherit from `ml_portfolio.models.pytorch.base.PyTorchForecaster`.
- **Hybrid / Ensemble** → consider extending `ml_portfolio.models.ensemble.BaseEnsembleForecaster`.

Each base class provides logging, parameter management, and compatibility with training engines.

## 2. Implement the model class

1. Create a new file under the appropriate module (e.g., `src/ml_portfolio/models/statistical/my_model.py`).
1. Implement `_fit`, `_predict`, and optional helper methods (`get_feature_importance`, `save`, `load`).
1. Register default parameters in `__init__`, exposing keyword arguments for Hydra overrides.
1. Add docstrings with parameter descriptions and usage examples.

```python
from ml_portfolio.models.statistical import StatisticalForecaster

class MyCoolForecaster(StatisticalForecaster):
    """Short description with usage example."""

    def __init__(self, hyperparam: float = 0.1, random_state: int | None = None):
        super().__init__()
        self.hyperparam = hyperparam
        self.random_state = random_state

    def _fit(self, X, y, **kwargs):
        # Train underlying model
        ...

    def _predict(self, X, **kwargs):
        # Return predictions
        ...
```

## 3. Wire Hydra configuration

- Create a config file under `src/ml_portfolio/conf/model/` (e.g., `my_cool_model.yaml`) with `_target_` pointing to your class.
- Register parameter defaults and search spaces if the model supports Optuna.
- Update composition configs (`src/ml_portfolio/conf/model/default.yaml`) if you want the model selectable via `model=my_cool_model`.

```yaml
# src/ml_portfolio/conf/model/my_cool_model.yaml
_target_: ml_portfolio.models.statistical.my_model.MyCoolForecaster
hyperparam: 0.1
random_state: 42
```

## 4. Add tests

- Unit tests validating `fit`/`predict` on synthetic data (`tests/unit/models/test_my_cool_model.py`).
- Optional integration tests ensuring compatibility with `StatisticalEngine` or `ModelBenchmark`.
- Update fixtures if new dependencies or datasets are required.

## 5. Update documentation

- Document the model in the [API reference](../api_reference/index.md) with parameter summaries and example usage.
- Mention availability in the [model selection playbook](../guides/model_selection.md) if appropriate.
- Include release notes in `CHANGELOG.md` under the next version heading.

## 6. Ensure tooling compatibility

- **Registry**: Add import in `ml_portfolio.models.registry` if the model should be loadable by name.
- **Benchmark suite**: Include the `_target_` path in benchmark configs as needed.
- **Dashboard**: Confirm the new model’s metrics appear correctly (run a quick benchmark and view in Streamlit).

## 7. Submit the contribution

- Run `pre-commit run --all-files` and `pytest`.
- Build documentation: `sphinx-build docs docs/_build/html`.
- Open a PR referencing this checklist and attach relevant screenshots or benchmark outputs.

By following this checklist, the new model will integrate seamlessly with datasets, training engines, benchmarks, and dashboards.
