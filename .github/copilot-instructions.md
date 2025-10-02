# Claude Code Guidelines - ML Forecasting Portfolio

## Overview
Professional ML portfolio showcasing time series forecasting across multiple domains (retail, rideshare, inventory, transportation). Architecture separates reusable components (`src/ml_portfolio/`) from self-contained project demos (`projects/`).

## Core Architecture
- **Shared Library**: `src/ml_portfolio/` - Reusable forecasting components
- **Self-Contained Projects**: `projects/*/` - Independent demonstrations with own data/configs
- **Config Management**: Hydra for structured configs and experiment tracking
- **Hyperparameter Tuning**: Optuna for automated optimization
- **Model Registry**: Centralized loading/comparison for dashboards and APIs

## Coding Standards

### Virtual Environment (REQUIRED)
- **ALWAYS use .venv**: All commands (pip, python, pytest, etc.) MUST run inside `.venv/`
- **Check before executing**: Verify virtual environment is active before running any Python/pip commands
- **Activation**:
  - Windows: `.venv\Scripts\activate`
  - Linux/Mac: `source .venv/bin/activate`
- **Verification**: Check `$VIRTUAL_ENV` environment variable or `which python` output

### Output Rules (STRICT)
- ✅ Emoji ONLY in `.md` files
- ❌ NO emoji in: Python code, comments, logs, output, commit messages, YAML, JSON, or any non-markdown files
- ✅ ASCII-only output for cross-platform compatibility

### Code Quality
- **Style Guide**: PEP 8 (checked with pycodestyle)
- **Formatting**: Black
- **Type Hints**: For public APIs and complex functions
- **Documentation**: Docstrings for all public functions/classes
- **Testing**: pytest for unit and integration tests

### Naming Conventions
- **Modules**: lowercase_with_underscores
- **Classes**: PascalCase
- **Functions**: snake_case
- **Configs**: Consistent keys across projects (model=..., dataset=..., optimizer=...)

## Project Structure

### Standardized Project Layout
```
projects/{project_name}/
├── data/                   # raw/, interim/, processed/
├── models/                 # artifacts/, checkpoints/
├── scripts/                # download_data.py, generate_data.py
├── notebooks/              # Exploration notebooks
├── tests/                  # Unit tests
├── api/                    # FastAPI endpoints (optional)
└── app/                    # Streamlit dashboards (optional)
```

**Notes**:
- `scripts/` is for data acquisition only (download/generate data)
- Training scripts live in `src/ml_portfolio/training/`
- Avoid project-specific `outputs/` or `docs/` - use root-level or git-ignore

### Shared Library Structure
```
src/ml_portfolio/
├── conf/                   # Hydra config templates
├── data/                   # datasets.py, loaders.py, transforms.py, timeseries.py
├── models/
│   ├── statistical/        # ARIMA, Prophet
│   ├── deep_learning/      # LSTM, TCN, Transformer
│   ├── blocks/             # Reusable layers
│   └── losses.py, metrics.py, registry.py
├── training/               # engine.py, callbacks.py, train.py, utils.py
├── evaluation/             # backtesting.py, plots.py
├── pipelines/              # classical.py, hybrid.py
└── utils/                  # config.py, io.py
```

## Development Patterns

### Class Inheritance
- Inherit from `src/ml_portfolio/` base classes
- Override only for domain-specific customization (data loading, feature engineering)
- Naming: `{ProjectName}{ModelType}` (e.g., `WalmartARIMAModel`, `OlaDemandLSTM`)

### Hydra Configuration
- Use `_target_:` for object instantiation
- Configs stored in `src/ml_portfolio/conf/` (shared library)
- Example structure:
```yaml
_target_: ml_portfolio.models.deep_learning.lstm.LSTMForecaster
hidden_size: 128
num_layers: 2
```

### Training Commands
```bash
# Single run with config overrides
python scripts/train.py model=lstm dataset=walmart optimizer=adam

# Multi-run experiments
python scripts/train.py -m model=lstm,tcn optimizer=adam,adamw

# Hyperparameter optimization
python scripts/optimize.py --config-path conf --config-name config
```

## Data Handling
- **Time Series Windowing**: Use `ml_portfolio.data.datasets` for windowing
- **Transforms**: Scalers, encoders in `transforms.py`
- **Time Series Split**: Respect temporal order, use gap periods to prevent leakage
- **Caching**: Use Parquet format for processed data

## Model Implementation
- **Base Classes**: Common interfaces in `models/wrappers.py` (sklearn compatible)
- **Losses**: Quantile, Pinball, SMAPE in `losses.py`
- **Metrics**: MAPE, RMSE, MAE, directional accuracy in `metrics.py`
- **Training**: Use engine with callbacks, early stopping, checkpointing
- **Registry**: Load models via `registry.load_model(name, version)`

## Evaluation Standards
- **Dataset-Specific Primary Metrics**:
  - Walmart: WMAE (Weighted Mean Absolute Error)
  - Rideshare/Inventory: MAPE
  - TSI/Economic: RMSE
- **Universal Metrics**: MAPE, RMSE, MAE, directional accuracy
- **Cross-Validation**: Time series split with configurable gap periods
- **Visualization**: Residual analysis, forecast vs actual plots

## Error Handling Guidelines

### File Operations
- **Read Errors**: If file doesn't exist, check alternative paths or ask user for correct location
- **Write Errors**: Verify directory exists before writing; create if missing
- **Edit Errors**: If exact string match fails, show context and ask user to verify the code section

### Code Errors
- **Import Errors**: Check if module exists in `src/ml_portfolio/` structure before suggesting fixes
- **Config Errors**: Verify Hydra config structure matches expected format
- **Runtime Errors**: When encountering errors during execution, provide diagnostic info and suggest fixes

### Recovery Actions
1. **Check file existence** before reading/editing
2. **Verify directory structure** matches expected layout
3. **Validate config syntax** before suggesting Hydra commands
4. **Test imports** in Python files match actual module structure
5. **Ask for clarification** if multiple valid solutions exist

## Best Practices
- **Config-Driven**: All hyperparameters in YAML configs
- **Reproducible**: Hydra logs configs, outputs, random seeds
- **Time Series Aware**: Respect temporal order in splits/CV
- **Model Persistence**: Save with metadata (training period, hyperparams, metrics)
- **Error Handling**: Graceful degradation in optimization trials
- **Consistent Naming**: Use same config keys across all projects

---

*Claude Code guidelines for ML forecasting portfolio development.*
