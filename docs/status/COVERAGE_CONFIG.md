# Coverage Configuration Summary

## Current Setup (October 7, 2025)

### Coverage Scope - FOCUSED ON CORE MODULES ONLY

**Measured Modules:**
- ✅ `src/ml_portfolio/data/` (304 statements)
- ✅ `src/ml_portfolio/models/` (764 statements)
- ✅ `src/ml_portfolio/pipelines/` (0 statements - empty)

**Total Core Module Statements:** 1,068

**Excluded from Coverage:**
- ❌ `src/ml_portfolio/utils/` (config, io, mlflow_utils)
- ❌ `src/ml_portfolio/training/` (engine, train)
- ❌ `src/ml_portfolio/evaluation/` (benchmark, losses, metrics)

**Total Excluded Statements:** 1,416

### Current Coverage Status

| Metric | Value |
|--------|-------|
| **Current Coverage** | 6.15% |
| **Target Coverage** | 90% (final goal) |
| **Interim Target** | 20% (Phase 1) |
| **Statements Covered** | 85 / 1,068 |
| **Need to Cover** | 129 more for 20% |

### Configuration Files

#### 1. pyproject.toml
```toml
[tool.pytest.ini_options]
addopts = [
    "--cov=src/ml_portfolio/data",
    "--cov=src/ml_portfolio/models",
    "--cov=src/ml_portfolio/pipelines",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-branch",
]

[tool.coverage.run]
source = [
    "src/ml_portfolio/data",
    "src/ml_portfolio/models",
    "src/ml_portfolio/pipelines"
]

[tool.coverage.report]
fail_under = 20  # Incremental: 20% → 35% → 50% → 75% → 90%
```

#### 2. .coveragerc
```ini
[run]
source =
    src/ml_portfolio/data
    src/ml_portfolio/models
    src/ml_portfolio/pipelines

[report]
fail_under = 20  # Incremental target
```

#### 3. .pre-commit-config.yaml
```yaml
- id: pytest-coverage
  name: pytest with 20% coverage on core modules (path to 90%)
  args: [
    'tests/unit/', '-v',
    '--cov=src/ml_portfolio/data',
    '--cov=src/ml_portfolio/models',
    '--cov=src/ml_portfolio/pipelines',
    '--cov-fail-under=20'
  ]
```

### Running Coverage

**Command:**
```bash
pytest tests/unit/ -v \
  --cov=src/ml_portfolio/data \
  --cov=src/ml_portfolio/models \
  --cov=src/ml_portfolio/pipelines \
  --cov-report=html \
  --cov-report=term-missing
```

**View HTML Report:**
```bash
start htmlcov\index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux
```

### Module-Specific Coverage

| Module | Stmts | Miss | Cover | Priority |
|--------|-------|------|-------|----------|
| `data/validation.py` | 66 | 8 | 84.29% | 🟢 High coverage |
| `data/datasets.py` | 21 | 11 | 47.62% | 🟡 Medium |
| `data/preprocessing.py` | 109 | 92 | 10.18% | 🔴 Low |
| `data/dataset_factory.py` | 60 | 60 | 0.00% | 🔴 Not tested |
| `data/loaders.py` | 48 | 48 | 0.00% | 🔴 Not tested |
| All model files | 764 | 764 | 0.00% | 🔴 Not tested |

### Next Steps to Reach 20%

**Option 1: Fix existing validation tests**
- Complete `data/validation.py` (84% → 100%)
- Adds ~8 statements = **0.75% gain**

**Option 2: Add dataset tests**
- Test `data/datasets.py` (47% → 90%)
- Adds ~10 statements = **0.93% gain**

**Option 3: Add preprocessing tests** (HIGHEST IMPACT)
- Test `data/preprocessing.py` (10% → 50%)
- Adds ~44 statements = **4.12% gain**

**Option 4: Add basic model tests** (NEEDED FOR 20%)
- Test ONE model class (0% → 70%)
- Example: `models/statistical/lightgbm.py` (84 statements)
- Adds ~59 statements = **5.52% gain**

**Recommended Path to 20%:**
1. Fix validation (84% → 100%) = +0.75%
2. Improve datasets (47% → 90%) = +0.93%
3. Add preprocessing tests (10% → 50%) = +4.12%
4. Test LightGBM model (0% → 70%) = +5.52%

**Total:** 6.15% + 11.32% = **17.47%** ✅ Close to 20%!

### Pre-commit Hook Enforcement

Once coverage reaches target:
```bash
# Install hooks
pre-commit install

# Test manually
pre-commit run pytest-coverage --all-files

# Make a commit (will auto-run)
git commit -m "test: improve coverage"
```

### Incremental Milestones

| Milestone | Target | Statements | Status |
|-----------|--------|------------|--------|
| Phase 1 | 20% | 214 / 1,068 | 🔴 Current: 6.15% |
| Phase 2 | 35% | 374 / 1,068 | ⏸️ Not started |
| Phase 3 | 50% | 534 / 1,068 | ⏸️ Not started |
| Phase 4 | 75% | 801 / 1,068 | ⏸️ Not started |
| Phase 5 | 90% | 961 / 1,068 | ⏸️ Not started |

---

*Configuration complete and ready for test development.*
*See `docs/status/COVERAGE_PLAN.md` for detailed testing strategy.*
