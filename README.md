# 📈 ML Portfolio: Time Series Forecasting

Professional ML portfolio showcasing time series forecasting across multiple domains with modern MLOps practices.

> **� Quick Commands**: See [`docs/HYDRA_OPTUNA_QUICK_SETUP.md`](docs/HYDRA_OPTUNA_QUICK_SETUP.md) for 5 tested working commands you can run immediately.

> **�📚 Complete Documentation**: See [`docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md) for comprehensive guides and tutorials.

## 🚀 Quick Start

> **📚 Installation**: See [Setup Guide](docs/SETUP.md) for detailed environment setup and installation instructions.

```bash
# Run a model
python src/ml_portfolio/training/train.py model=arima dataset_factory=walmart

# Multi-model comparison
python src/ml_portfolio/training/train.py -m model=arima,lstm,random_forest dataset_factory=walmart
```

## 🏗️ Architecture

```
├── src/ml_portfolio/           # Reusable ML components
│   ├── models/                 # ARIMA, LSTM, TCN, Transformers
│   ├── data/                   # Datasets, loaders, transforms
│   ├── training/               # Training engine with MLflow
│   └── utils/                  # MLflow integration
├── projects/                   # Self-contained demos
│   ├── retail_sales_walmart/   # Walmart sales forecasting
│   ├── rideshare_demand_ola/   # OLA demand prediction
│   ├── inventory_forecasting/  # Supply chain optimization
│   └── transportation_tsi/     # TSI economic indicators
```

## ⚡ Key Features

- **🔧 Hydra Configuration**: Structured configs with override support
- **📊 MLflow Tracking**: Automated experiment tracking and model registry
- **🤖 Multiple Models**: Statistical (ARIMA) + ML (RF, LSTM, TCN, Transformers)
- **📈 Time Series Ready**: Proper temporal splits, windowing, lag features
- **🔄 Production Ready**: Type hints, testing, CI/CD, Docker support

## 🎯 Available Models

| Model | Type | Use Case |
|-------|------|----------|
| ARIMA | Statistical | Trend + seasonality |
| Random Forest | ML | Non-linear patterns |
| LSTM | Deep Learning | Long sequences |
| TCN | Deep Learning | Efficient training |
| Transformer | Deep Learning | Attention mechanisms |

## 📋 Project Demos

| Project | Domain | Dataset | Primary Metric |
|---------|--------|---------|----------------|
| **Walmart** | Retail | Sales data | WMAE |
| **OLA** | Rideshare | Demand patterns | MAPE |
| **Inventory** | Supply Chain | Stock levels | RMSE |
| **TSI** | Economics | Transport indicators | RMSE |

## 🛠️ Configuration Examples

```bash
# Single run
python src/ml_portfolio/training/train.py model=lstm dataset_factory=walmart optimizer=adam

# Hyperparameter sweep (grid search)
python src/ml_portfolio/training/train.py -m model=lstm dataset_factory=walmart optimizer=adam,adamw optimizer.lr=0.001,0.01,0.1

# Optuna optimization
python src/ml_portfolio/training/train.py experiment=lstm_sweep

# Model comparison
python src/ml_portfolio/training/train.py experiment=model_comparison model=arima,random_forest,lstm
```

## 🔄 Hyperparameter Optimization

**Comprehensive sweep capabilities** with Optuna integration:

- **📈 Statistical Models**: ARIMA order selection, Prophet seasonality tuning
- **🌳 ML Models**: Random Forest, XGBoost with regularization optimization
- **🧠 Deep Learning**: LSTM, TCN, Transformer architecture search
- **🎯 Multi-Objective**: Accuracy vs efficiency trade-offs
- **⚡ Advanced Features**: Pruning, persistent studies, distributed execution

```bash
# Quick start with pre-configured experiments
python src/ml_portfolio/training/train.py experiment=lstm_sweep          # 60 trials
python src/ml_portfolio/training/train.py experiment=random_forest_sweep # 50 trials
python src/ml_portfolio/training/train.py experiment=walmart_optimization # Dataset-specific

# See full documentation
# docs/HYPERPARAMETER_SWEEPS.md - Comprehensive guide
# docs/SWEEP_COMMANDS.md - Ready-to-use commands
```

## � Configuration Management

- **Hydra Framework**: Structured, modular configuration system
- **Config Composition**: Combine model, dataset, optimizer configs
- **Runtime Overrides**: Modify parameters via command line
- **Grid Search**: Built-in hyperparameter optimization with multirun

```bash
# Configuration examples
python src/ml_portfolio/training/train.py model=lstm optimizer.lr=0.01
python src/ml_portfolio/training/train.py -m 'model.hidden_size=64,128,256'

# See full documentation: docs/HYDRA_CONFIGURATION.md
```

## �📦 MLflow Integration

- **Automatic Tracking**: Parameters, metrics, artifacts
- **Model Registry**: Versioned model storage
- **Experiment Comparison**: Web UI with visualizations
- **Hydra Synergy**: Config-based metadata extraction

```bash
# View experiments
mlflow ui --port 5000
```

## 🎨 Outputs

Each run generates:
- **Metrics**: MAPE, RMSE, MAE, directional accuracy
- **Plots**: Actual vs predicted, residuals, time series
- **Artifacts**: Model files, predictions CSV
- **Logs**: Comprehensive training logs

## 🧹 Cleanup

Remove stored training runs to free up disk space:

```bash
# Preview what will be deleted
python clean_runs.py --dry-run

# Clean all runs (MLflow, Hydra outputs, checkpoints)
python clean_runs.py

# See full documentation: docs/CLEANUP.md
```

## 🔬 Development

> **📚 Setup**: See [Setup Guide](docs/SETUP.md) for development environment setup and requirements.

```bash
# Run tests
pytest

# Format code
black src/ projects/
ruff check src/ projects/

# Pre-commit hooks
pre-commit install
```

---

**Built with**: Python 3.11+ • Hydra • MLflow • PyTorch • scikit-learn • pandas
