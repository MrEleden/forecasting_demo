# Code Coverage Plan - Core Modules

## Current Status (October 7, 2025)

### Coverage Scope
Focused on core implementation modules only:
- `src/ml_portfolio/data/`
- `src/ml_portfolio/models/`
- `src/ml_portfolio/pipelines/`

**Excluded from coverage requirements:**
- `src/ml_portfolio/utils/` (utilities, config, io, mlflow)
- `src/ml_portfolio/training/` (training engine, scripts)
- `src/ml_portfolio/evaluation/` (benchmarking, losses)

### Current Metrics
- **Total Statements**: 1,068
- **Covered**: 85 (6.15%)
- **Target**: 90% (961 statements)
- **Gap**: 876 statements

### Module-Level Breakdown

| Module | Statements | Covered | Coverage | Status |
|--------|-----------|---------|----------|--------|
| `data/validation.py` | 66 | 58 | 84.29% | ✅ Nearly there |
| `data/datasets.py` | 21 | 10 | 47.62% | ⚠️ Needs work |
| `data/preprocessing.py` | 109 | 17 | 10.18% | ⚠️ Needs work |
| `data/dataset_factory.py` | 60 | 0 | 0.00% | ❌ Not tested |
| `data/loaders.py` | 48 | 0 | 0.00% | ❌ Not tested |
| **All models** | 764 | 0 | 0.00% | ❌ Not tested |
| `models/base.py` | 75 | 0 | 0.00% | ❌ Not tested |
| `models/ensemble/*` | 113 | 0 | 0.00% | ❌ Not tested |
| `models/statistical/*` | 575 | 0 | 0.00% | ❌ Not tested |
| `pipelines/` | 0 | 0 | N/A | Empty folder |

## Why 90% on Core Modules?

These modules contain:
1. **Reusable forecasting components** shared across projects
2. **Data transformation logic** critical for model inputs
3. **Model implementations** that need reliability
4. **Core business logic** that must be correct

The excluded modules (utils, training, evaluation) are:
- More configuration/orchestration focused
- Heavily dependent on external systems (MLflow, Hydra)
- Better tested through integration/E2E tests
- Less critical for unit test coverage

## Incremental Coverage Plan

### Phase 1: Quick Wins (Target: 20% coverage)
**Estimated effort**: 2-3 hours

Priority: Test existing validated code
- ✅ Fix `data/validation.py` edge cases (84% → 100%)
- ✅ Add `data/datasets.py` tests (47% → 90%)
  - TimeSeriesDataset creation
  - Data access methods
  - Metadata handling

### Phase 2: Data Pipeline (Target: 35% coverage)
**Estimated effort**: 4-5 hours

Priority: Data preprocessing and loading
- Test `data/preprocessing.py` (10% → 90%)
  - StatisticalPreprocessingPipeline fit/transform
  - StaticTimeSeriesPreprocessingPipeline features
  - Inverse transforms
- Test `data/dataset_factory.py` (0% → 70%)
  - Factory pattern for dataset creation
- Test `data/loaders.py` (0% → 70%)
  - Data loading utilities

### Phase 3: Model Base Classes (Target: 50% coverage)
**Estimated effort**: 3-4 hours

Priority: Model interfaces and base functionality
- Test `models/base.py` (0% → 85%)
  - BaseForecaster interface
  - StatisticalForecaster wrapper
  - PyTorchForecaster base
- Test model initialization patterns

### Phase 4: Statistical Models (Target: 75% coverage)
**Estimated effort**: 8-10 hours

Priority: Core model implementations
- Test each statistical model (0% → 80%):
  - `models/statistical/lightgbm.py`
  - `models/statistical/xgboost.py`
  - `models/statistical/catboost.py`
  - `models/statistical/random_forest.py`
  - `models/statistical/arima.py`
  - `models/statistical/prophet.py`
  - `models/statistical/sarimax.py`

Focus on:
- Model initialization with different configs
- Fit/predict workflow
- Hyperparameter validation
- Error handling

### Phase 5: Ensemble Models (Target: 90% coverage)
**Estimated effort**: 4-5 hours

Priority: Ensemble forecasting
- Test `models/ensemble/voting.py` (0% → 90%)
  - Voting ensemble with multiple base models
  - Different voting strategies (mean, median)
- Test `models/ensemble/stacking.py` (0% → 90%)
  - Stacking with meta-learner
  - Cross-validation for meta-features

## Testing Strategy

### Unit Test Structure
```python
# tests/unit/models/test_lightgbm.py
class TestLightGBMForecaster:
    def test_initialization_with_defaults(self):
        """Test model can be created with default params."""

    def test_fit_with_training_data(self):
        """Test model fitting on sample data."""

    def test_predict_returns_correct_shape(self):
        """Test predictions match expected dimensions."""

    def test_hyperparameter_validation(self):
        """Test invalid hyperparameters raise errors."""

    def test_persistence_and_loading(self):
        """Test model can be saved and loaded."""
```

### Integration Test Strategy
Use the existing optimization tests:
- Run full training pipeline for each model
- Validate outputs match expected format
- Check model persistence and reloading

### Coverage Measurement
```bash
# Run focused coverage on core modules
pytest tests/unit/ -v \
  --cov=src/ml_portfolio/data \
  --cov=src/ml_portfolio/models \
  --cov=src/ml_portfolio/pipelines \
  --cov-report=html \
  --cov-report=term-missing \
  --cov-fail-under=90
```

## Pre-commit Hook Configuration

Pre-commit hook enforces 90% coverage on core modules only:
```yaml
- id: pytest-coverage
  name: pytest with 90% coverage on core modules
  entry: pytest
  args: [
    'tests/unit/', '-v',
    '--cov=src/ml_portfolio/data',
    '--cov=src/ml_portfolio/models',
    '--cov=src/ml_portfolio/pipelines',
    '--cov-fail-under=90'
  ]
```

## Timeline Estimate

| Phase | Target Coverage | Estimated Hours | Dependencies |
|-------|----------------|-----------------|--------------|
| Phase 1 | 20% | 2-3 hours | None |
| Phase 2 | 35% | 4-5 hours | Phase 1 |
| Phase 3 | 50% | 3-4 hours | Phase 2 |
| Phase 4 | 75% | 8-10 hours | Phase 3 |
| Phase 5 | 90% | 4-5 hours | Phase 4 |
| **Total** | **90%** | **21-27 hours** | Sequential |

## Temporary Configuration

While working towards 90%, temporarily reduce threshold:

**pyproject.toml:**
```toml
[tool.coverage.report]
fail_under = 20  # Incremental: 20% → 35% → 50% → 75% → 90%
```

**Update threshold** after completing each phase.

## Success Metrics

1. **Coverage**: ≥90% on data, models, pipelines
2. **Test Quality**: All tests pass, no flaky tests
3. **CI/CD**: Coverage checks run on every PR
4. **Documentation**: All test patterns documented
5. **Maintainability**: Tests are clear and maintainable

## Notes

- Current 37 passing tests cover validation and metrics well
- 8 skipped preprocessing tests need refactoring (API mismatch)
- Focus on testing behavior, not implementation details
- Use fixtures for common test data (already in conftest.py)
- Leverage integration tests from optimization runs

---

*Last updated: October 7, 2025*
*Current coverage: 6.15% of core modules*
*Next milestone: Phase 1 - 20% coverage*
