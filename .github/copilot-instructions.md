# Copilot Instructions for ML Portfolio - Forecasting Projects

## Project Overview
This is a professional ML portfolio showcasing time series forecasting across multiple domains. The architecture separates reusable forecasting components (`src/ml_portfolio/`) from self-contained project demonstrations (`projects/`). Primary showcase includes Walmart retail sales, ride-sharing demand (Ola Bike), inventory forecasting, and U.S. Transportation Services Index (TSI) with statistical (ARIMA, Prophet), deep learning (LSTM, TCN, Transformer), and hybrid approaches.

## Project Structure
```
ml-portfolio/
├── README.md                      # Portfolio overview and navigation
├── download_all_data.py           # Master data orchestration script
├── validate_structure.py          # Project structure validator (CI enforced)
├── docs/                          # Documentation and guides
│   ├── README.md                  # Documentation index
│   ├── DATA_DOWNLOAD_README.md    # Data setup guide
│   ├── KAGGLE_SETUP.md           # Kaggle API configuration
│   ├── DATASET_STATUS.md         # Current data status
│   └── PROJECT_STRUCTURE.md      # Mandatory structure guide
├── .github/
│   ├── copilot-instructions.md   # This file - AI development guidelines
│   └── workflows/
│       └── validate-structure.yml # CI structure validation
├── ci/                           # CI/CD infrastructure
│   ├── README.md                 # CI/CD documentation
│   ├── github-actions/           # GitHub Actions workflow templates
│   │   └── validate-structure.yml # Project structure validation workflow
│   ├── scripts/                  # CI/CD automation scripts
│   │   └── validate_structure.py # Project structure validation script
│   └── docker/                   # Docker configurations (future)
├── src/
│   └── ml_portfolio/             # Shared, importable forecasting library
│       ├── data/
│       │   ├── datasets.py       # PyTorch Dataset wrappers for time series
│       │   ├── loaders.py        # DataLoader factories with time series sampling
│       │   ├── transforms.py     # Scalers, encoders, feature engineering
│       │   └── timeseries.py     # Windowing, lags, calendar features
│       ├── models/
│       │   ├── blocks/           # Reusable layers (TCN, attention mechanisms)
│       │   ├── losses.py         # MSE/MAE/Quantile/Pinball/SMAPE for forecasting
│       │   ├── metrics.py        # RMSE/MAE/MAPE + time series CV helpers
│       │   ├── forecasting/
│       │   │   ├── lstm.py       # LSTM/Seq2Seq implementations
│       │   │   ├── transformer.py# Informer/Transformer for long sequences
│       │   │   └── tcn.py        # Temporal ConvNet
│       │   ├── wrappers.py       # Scikit-learn compatible interfaces
│       │   └── registry.py       # Model registry for loading/comparing models
│       ├── training/
│       │   ├── engine.py         # Train/validate loops with early stopping
│       │   ├── callbacks.py      # Checkpoints, LR scheduling, monitoring
│       │   └── utils.py          # Seed, device management, logging
│       ├── evaluation/
│       │   ├── backtesting.py    # Rolling origin evaluation, time series CV
│       │   └── plots.py          # Learning curves, forecast visualization
│       ├── pipelines/
│       │   ├── classical.py      # Scikit-learn pipelines for preprocessing
│       │   └── hybrid.py         # Sklearn preprocessing → PyTorch model
│       └── utils/
│           ├── io.py             # I/O, caching, data paths
│           └── config.py         # Hydra/OMEGACONF configuration loading
├── projects/                     # STANDARDIZED STRUCTURE (CI enforced)
│   ├── retail_sales_walmart/     # SELF-CONTAINED: Walmart sales forecasting
│   │   ├── README.md             # Project-specific documentation
│   │   ├── api/                  # FastAPI endpoints
│   │   ├── app/                  # Streamlit dashboard
│   │   ├── conf/                 # Hydra configuration management
│   │   │   ├── config.yaml       # Main project configuration
│   │   │   ├── dataset/          # Dataset configurations
│   │   │   ├── model/            # Model configurations
│   │   │   ├── optimizer/        # Optimizer configurations
│   │   │   ├── scheduler/        # LR scheduler configurations
│   │   │   └── hydra/            # Hydra runtime settings
│   │   ├── data/                 # DVC-style data layout
│   │   │   ├── external/         # External/third-party data
│   │   │   ├── interim/          # Intermediate processed data
│   │   │   ├── processed/        # Final processed data
│   │   │   └── raw/              # Raw unprocessed data (Walmart.csv)
│   │   ├── models/               # Trained models and artifacts
│   │   │   ├── artifacts/        # Model artifacts (scalers, encoders)
│   │   │   └── checkpoints/      # Model checkpoints during training
│   │   ├── notebooks/            # Jupyter notebooks for exploration
│   │   ├── reports/              # Generated analysis reports
│   │   │   └── figures/          # Generated plots and visualizations
│   │   ├── scripts/              # Python scripts
│   │   │   └── download_data.py  # Data acquisition script
│   │   └── tests/                # Unit and integration tests
│   ├── rideshare_demand_ola/     # SELF-CONTAINED: Same structure as above
│   │   ├── README.md
│   │   ├── scripts/
│   │   │   └── generate_data.py  # Synthetic data generation
│   │   └── [same folder structure as walmart]
│   ├── inventory_forecasting/    # SELF-CONTAINED: Same structure as above
│   │   ├── README.md
│   │   ├── scripts/
│   │   │   └── generate_data.py  # Synthetic data generation
│   │   └── [same folder structure as walmart]
│   └── transportation_tsi/       # SELF-CONTAINED: Same structure as above
│       ├── README.md
│       ├── scripts/
│       │   └── download_data.py  # BTS API data download
│       └── [same folder structure as walmart]
```

## Coding Standards and Guidelines

### Output and Formatting Rules
- **Emoji Usage**: Use emoji ONLY in markdown (.md) files for documentation and README files
- **Python Output**: NO emoji in Python print statements, debug messages, or logging output (ensures Windows terminal compatibility)
- **ASCII Compatibility**: All Python script output must be ASCII-only for cross-platform compatibility
- **Professional Standards**: Clean, readable output in all scripts and tools
- **Non-Markdown Files**: ALL files except .md files must be emoji-free (includes .py, .yml, .yaml, .json, .txt, .sh, etc.)

### Code Quality
- **Linting**: Use ruff for fast Python linting (replaces flake8)
- **Formatting**: Black for consistent code formatting
- **Type Hints**: Use type annotations for public APIs and complex functions
- **Documentation**: Docstrings for all public functions and classes
- **Testing**: pytest for unit and integration tests

### File Structure Guidelines
- **.gitkeep Files**: All empty folders contain .gitkeep files to maintain folder structure in Git (auto-created by validation script)
- **Config Keys**: Consistent naming across all projects (model=..., dataset=..., optimizer=...)
- **File Structure**: Follow standardized 22-folder structure enforced by validation script
- **Module Names**: Use lowercase with underscores for Python modules
- **Class Names**: PascalCase for class names, snake_case for function names

## Dependency Management and Poetry

### Poetry Project Structure
This project uses **Poetry** for modern Python dependency management, providing reproducible builds, virtual environment management, and streamlined development workflows. Poetry replaces traditional `requirements.txt` files with structured `pyproject.toml` configuration.

### Core Poetry Files
- **`pyproject.toml`**: Main project configuration defining dependencies, build settings, and tool configurations
- **`poetry.lock`**: Lock file ensuring exact dependency versions across environments (commit to Git)
- **`requirements.txt`**: Generated fallback for deployment environments without Poetry
- **`requirements-dev.txt`**: Generated development dependencies for legacy systems

### Dependency Groups and Organization
```toml
[tool.poetry.dependencies]
python = "^3.9"                    # Python version constraint
# Core time series and ML
pandas = ">=1.5.0"                 # Data manipulation
numpy = ">=1.21.0"                 # Numerical computing
scikit-learn = ">=1.0.0"           # Machine learning algorithms
matplotlib = ">=3.5.0"             # Plotting and visualization
plotly = ">=5.0.0"                 # Interactive visualizations

# Statistical forecasting (optional installs)
statsmodels = {version = ">=0.13.0", optional = true}  # ARIMA, seasonal decomposition
prophet = {version = ">=1.1.0", optional = true}       # Facebook Prophet forecasting

# Deep learning (optional installs)
torch = {version = ">=1.11.0", optional = true}        # PyTorch for LSTM/Transformer
gluonts = {version = ">=0.11.0", optional = true}      # DeepAR and advanced forecasting

# Gradient boosting (optional installs)
xgboost = {version = ">=1.6.0", optional = true}       # XGBoost for feature-based forecasting
lightgbm = {version = ">=3.3.0", optional = true}     # LightGBM alternative

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"                 # Testing framework
pytest-cov = "^4.0.0"             # Coverage reporting
ruff = "^0.1.0"                   # Fast Python linter (replaces flake8)
black = "^22.0.0"                 # Code formatting
pre-commit = "^3.0.0"             # Git hooks for quality checks
jupyter = "^1.0.0"                # Notebook development
ipykernel = "^6.0.0"              # Jupyter kernel

[tool.poetry.group.train.dependencies]
optuna = "^3.4.0"                 # Hyperparameter optimization
mlflow = "^2.8.0"                 # Experiment tracking (optional)
hydra-core = "^1.3.0"             # Configuration management
omegaconf = "^2.3.0"              # YAML/structured configs
tensorboard = "^2.14.0"           # Training visualization

[tool.poetry.group.app.dependencies]
streamlit = "^1.15.0"             # Dashboard framework
fastapi = "^0.104.0"              # API endpoints
uvicorn = "^0.24.0"               # ASGI server
pydantic = "^2.4.0"               # Request/response validation

[tool.poetry.extras]
statistical = ["statsmodels", "prophet"]
deep_learning = ["torch", "gluonts"] 
boosting = ["xgboost", "lightgbm"]
all = ["statsmodels", "prophet", "torch", "gluonts", "xgboost", "lightgbm"]
```

### Environment Setup Commands
```bash
# Initial project setup
curl -sSL https://install.python-poetry.org | python3 -  # Install Poetry
poetry install                                            # Install base dependencies
poetry install --extras "statistical deep_learning"      # Install with optional extras
poetry install --with dev,train,app                      # Install with specific groups

# Virtual environment management
poetry shell                      # Activate Poetry virtual environment
poetry env info                   # Show virtual environment information
poetry env list                   # List available virtual environments
poetry env remove <env-name>      # Remove virtual environment

# Dependency management
poetry add pandas                 # Add runtime dependency
poetry add --group dev pytest     # Add development dependency
poetry add --group train optuna   # Add training dependency
poetry add --optional statsmodels # Add optional dependency
poetry show                       # Show installed packages
poetry show --tree                # Show dependency tree
poetry check                      # Verify consistency

# Lock file and export
poetry lock                       # Update lock file
poetry export -f requirements.txt --output requirements.txt
poetry export --with dev -f requirements.txt --output requirements-dev.txt
```

### Development Workflow Integration
```bash
# Quality checks (run before commits)
poetry run ruff check src/ tests/           # Linting
poetry run black src/ tests/                # Formatting
poetry run pytest tests/ --cov=src/         # Testing with coverage

# Training and experimentation
poetry run python projects/retail_sales_walmart/scripts/train_arima.py
poetry run streamlit run projects/retail_sales_walmart/app/dashboard.py

# Production deployment
poetry build                     # Build wheel and source distribution
poetry install --only=main       # Install only production dependencies
```

### Docker Integration with Poetry
```dockerfile
# Multi-stage Docker build with Poetry
FROM python:3.9-slim as builder
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

FROM python:3.9-slim as runtime
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY src/ ./src/
```

### Dependency Strategy and Guidelines
- **Core Dependencies**: Keep minimal for maximum compatibility
- **Optional Extras**: Group heavy dependencies (PyTorch, TensorFlow) as optional installs
- **Version Pinning**: Use `^` for compatible updates, `>=` for minimum versions
- **Group Separation**: Isolate dev, training, and app dependencies to reduce production bloat
- **Regular Updates**: Use `poetry update` cautiously, test thoroughly
- **Lock File Commits**: Always commit `poetry.lock` for reproducible builds
- **Environment Isolation**: Never install packages outside Poetry in project environments

### Troubleshooting Common Issues
```bash
# Poetry installation issues
curl -sSL https://install.python-poetry.org | python3 - --uninstall  # Reinstall
poetry config virtualenvs.in-project true                           # Local .venv

# Dependency conflicts
poetry lock --no-update          # Resolve without updating
poetry install --no-dev          # Skip development dependencies
poetry show --outdated           # Check for updates

# Virtual environment problems
poetry env remove python         # Remove current environment
poetry install                   # Recreate environment
poetry cache clear pypi --all    # Clear package cache
```

### Legacy Requirements Support
For environments without Poetry support, maintain generated requirements files:
```bash
# Generate requirements for deployment
poetry export -f requirements.txt --output requirements.txt --without dev
poetry export --with dev -f requirements.txt --output requirements-dev.txt

# Install from requirements (fallback)
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development environment
```

## Architecture Principles
- **Shared Library**: `src/ml_portfolio/` contains reusable forecasting components
- **Self-Contained Projects**: Each `projects/*/` is independent with its own data, notebooks, configs
- **Hydra Configuration**: Object instantiation and experiment management via structured configs
- **Optuna Optimization**: Automated hyperparameter tuning with pruning and multi-objective support
- **Consistent Naming**: Standardized config keys (model=..., dataset=...) across all projects
- **Model Registry**: Centralized pattern for loading/comparing models in dashboards and APIs
- **Reproducible Experiments**: Hydra configs for hyperparameters, data splits, model architecture
- **Portfolio Showcase**: Multiple forecasting domains demonstrating versatility
- **Production Ready**: FastAPI endpoints, Streamlit dashboards, CI/CD pipelines
- **Time Series Focus**: Specialized components for windowing, backtesting, seasonal patterns

## Data Handling Patterns
- **PyTorch Datasets**: `ml_portfolio.data.datasets` for time series windowing and sampling
- **Configurable Transforms**: Scalers, encoders, and feature engineering pipelines
- **Calendar Features**: Holiday effects, seasonality, temporal encoding in `timeseries.py`
- **Data Splits**: Time-aware train/validation/test splits respecting temporal order
- **Caching**: Efficient I/O with parquet storage for interim/processed data
- **DVC-style Layout**: `raw/` → `interim/` → `processed/` data progression per project

## Model Implementation Conventions
- **Abstract Base Classes**: Common interfaces in `models/wrappers.py` for sklearn compatibility
- **Modular Architecture**: Reusable blocks (TCN, attention) in `models/blocks/`
- **Loss Functions**: Forecasting-specific losses (Quantile, Pinball, SMAPE) in `losses.py`
- **Training Engine**: DLwP-style train/validate loops with callbacks and early stopping
- **Model Registry**: Checkpoints and artifacts storage with metadata tracking via `registry.py`
- **Hybrid Pipelines**: Sklearn preprocessing feeding into PyTorch models
- **Time Series Metrics**: MAPE, SMAPE, directional accuracy with temporal cross-validation
- **Consistent Naming**: All models use same config keys across projects (model=lstm, model=arima)

## Key Development Commands
```bash
# Environment setup
poetry install                    # Install all dependencies and create virtual environment
poetry shell                      # Activate virtual environment

# Development dependencies
poetry add --group dev pytest black ruff pre-commit
poetry add --group train optuna mlflow hydra-core

# Hydra-configured training (single run)
poetry run python projects/retail_sales_walmart/scripts/train.py \
    model=lstm dataset=walmart optimizer=adam

# Optuna hyperparameter optimization
poetry run python projects/retail_sales_walmart/scripts/optimize.py \
    --config-path conf --config-name config \
    hydra.sweep.dir=outputs/optuna_study

# Multi-run experiments with Hydra sweeps
poetry run python projects/retail_sales_walmart/scripts/train.py -m \
    model=lstm,tcn,transformer \
    optimizer=adam,adamw \
    dataset.lookback_window=24,48,168

# Hydra config overrides
poetry run python scripts/train.py \
    model.hidden_size=128 \
    optimizer.lr=0.001 \
    dataset.batch_size=64

# Model evaluation and comparison
poetry run python scripts/evaluate.py --study-name walmart_optimization

# Dashboard with config selection
poetry run streamlit run app/dashboard.py

# Testing and code quality
poetry run pytest tests/
poetry run ruff check src/ tests/
poetry run black src/ tests/

# Docker deployment
docker build -t forecasting-demo .
docker run -p 8501:8501 -e DATASET=walmart forecasting-demo
```

## Model Evaluation Standards
- **Primary Metric**: Dataset-specific primary metrics (WMAE for Walmart, MAPE for rideshare/inventory, RMSE for TSI/economic indicators)
- **Universal Metrics**: MAPE, RMSE, MAE, directional accuracy across all datasets
- **Cross-Validation**: Time series split with configurable gap periods per dataset
- **Comparison Framework**: Statistical significance testing between models
- **Visualization**: Residual analysis, forecast vs actual plots by entity/time period

## Streamlit Dashboard Requirements
- **Dataset Selector**: Dropdown to switch between available datasets
- **Multi-Model Comparison**: Side-by-side forecast plots with confidence intervals
- **Dynamic Filters**: Entity selection based on dataset (stores/depts, regions, products, pickup zones, TSI components)
- **Performance Metrics**: Real-time metric display per model (dataset-appropriate)
- **Data Exploration**: Time series visualization with trend/seasonality decomposition
- **Model Insights**: Feature importance, residual analysis, forecast uncertainty

## Docker Deployment Patterns
- **Base Image**: `python:3.9-slim` for production efficiency
- **Multi-Stage Build**: Separate stages for dependencies and application
- **Environment Variables**: Model selection, data paths, dashboard configuration
- **Health Checks**: Endpoint for model availability and data freshness validation
- **Volume Mounts**: External data directory for dataset updates

## Critical Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.9"

# Core forecasting
pandas = ">=1.5.0"
numpy = ">=1.21.0"
statsmodels = ">=0.13.0"      # ARIMA implementation
prophet = ">=1.1.0"           # Prophet forecasting (formerly fbprophet)
tensorflow = ">=2.8.0"        # LSTM and TFP-STS
torch = ">=1.11.0"           # PyTorch LSTM and DeepAR
gluonts = ">=0.11.0"         # DeepAR implementation

# Dashboard and deployment
streamlit = ">=1.15.0"
plotly = ">=5.0.0"           # Interactive time series plots

# Model evaluation and gradient boosting
scikit-learn = ">=1.0.0"     # Metrics and preprocessing
xgboost = ">=1.6.0"          # Gradient boosting for rideshare/inventory
lightgbm = ">=3.3.0"         # Alternative gradient boosting

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
ruff = "^0.1.0"              # Fast Python linter (replaces flake8)
black = "^22.0.0"
pre-commit = "^3.0.0"
jupyter = "^1.0.0"

[tool.poetry.group.train.dependencies]
optuna = "^3.4.0"            # Hyperparameter optimization
mlflow = "^2.8.0"           # Experiment tracking (optional)
hydra-core = "^1.3.0"       # Configuration management
omegaconf = "^2.3.0"        # YAML/structured configs

[tool.poetry.group.app.dependencies]
fastapi = "^0.104.0"        # API endpoints
uvicorn = "^0.24.0"         # ASGI server
pydantic = "^2.4.0"         # Request/response validation
```

## Project-Specific Patterns
- **Hydra Object Instantiation**: Use `_target_` to instantiate models, optimizers, datasets from configs
- **Optuna Integration**: Multi-objective optimization with pruning for early stopping
- **Config-Driven Experiments**: All hyperparameters, model architectures defined in YAML
- **Reproducible Runs**: Hydra automatically logs configs, outputs, and random seeds
- **Consistent Config Naming**: All projects use same keys (model=..., dataset=..., optimizer=...)
- **Model Registry Usage**: Load models via registry for dashboards/APIs: `registry.load_model(name, version)`
- **Time Series Schema**: Follow canonical schema in `docs/forecasting_framework.md`
- **Loader Guidelines**: New datasets follow documented patterns for transforms and windowing
- **Time Series Split**: Use `TimeSeriesSplit` with gap=4 weeks to prevent data leakage
- **Model Persistence**: Save models with metadata (training period, hyperparameters, performance)
- **Error Handling**: Graceful degradation when individual optimization trials fail
- **Logging**: Track model training time, convergence, and prediction confidence levels

## Structure Validation and Enforcement
- **Automated Validation**: Run `python ci/scripts/validate_structure.py` to check all project structures
- **CI/CD Integration**: GitHub Actions automatically validates structure on commits
- **Structure Fixes**: Use `python ci/scripts/validate_structure.py --fix` to create missing folders
- **Required Structure**: All projects must have identical 22-folder structure
- **Enforcement**: CI pipeline fails if any project deviates from standard structure
- **Documentation**: Project structure is enforced and documented in validation script

---

*This file reflects the current ML portfolio forecasting implementation with validated project structure and professional coding standards. Update model-specific sections as new algorithms are added or performance benchmarks change.*