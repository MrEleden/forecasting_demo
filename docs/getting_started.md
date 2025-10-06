# Getting Started with ML Forecasting Portfolio

This guide will help you get started with the ML Forecasting Portfolio in 10 minutes.

## Prerequisites

- Python 3.9+ installed
- Git installed
- 4GB RAM minimum
- Windows, Linux, or macOS

## Step 1: Installation (5 minutes)

### Clone Repository

```bash
git clone https://github.com/MrEleden/forecasting_demo.git
cd forecasting_demo
```

### Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# ML models
pip install -r requirements-ml.txt

# Development tools (optional)
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

## Step 2: Download Data (2 minutes)

```bash
# Download all datasets
python download_all_data.py

# Or download specific dataset
cd projects/retail_sales_walmart
python scripts/download_data.py
```

## Step 3: Train Your First Model (3 minutes)

### Simple Training

```bash
python src/ml_portfolio/training/train.py --config-name walmart model=lightgbm
```

This will:
1. Load Walmart sales data
2. Create lag features and date features
3. Train LightGBM model
4. Evaluate on validation set
5. Save model checkpoint

### View Results

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

You'll see:
- Experiment name: `walmart_sales_forecasting`
- Metrics: MAPE, RMSE, MAE
- Parameters: All hyperparameters
- Artifacts: Trained model

## Next Steps

### Run Hyperparameter Optimization

```bash
python src/ml_portfolio/scripts/run_optimization.py --models lightgbm --trials 50
```

### Compare Multiple Models

```bash
python src/ml_portfolio/training/train.py --config-name walmart --multirun model=lightgbm,xgboost,catboost
```

### Use Different Dataset

```bash
python src/ml_portfolio/training/train.py --config-name ola model=lightgbm
```

## Common Issues

### Issue: ModuleNotFoundError

**Solution:** Make sure you installed the package
```bash
pip install -e .
```

### Issue: CUDA not found (for deep learning models)

**Solution:** Use CPU-only version
```bash
pip install -r requirements.txt  # Installs CPU-only PyTorch
```

### Issue: Out of memory

**Solution:** Reduce batch size or use smaller model
```bash
python train.py model.n_estimators=100  # Smaller model
```

## Configuration Overview

All configurations are in `src/ml_portfolio/conf/`:

```
conf/
├── config.yaml              # Base config
├── walmart.yaml             # Walmart dataset config
├── model/                   # Model configs
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   └── ...
└── optuna/                  # Hyperparameter search spaces
    ├── lightgbm.yaml
    └── ...
```

### Override Configurations

```bash
# Change hyperparameters
python train.py model.n_estimators=1000 model.learning_rate=0.1

# Change dataset split
python train.py dataset_factory.train_ratio=0.8

# Disable MLflow
python train.py use_mlflow=false
```

## Project Structure

```
forecasting_demo/
├── src/ml_portfolio/          # Shared library
│   ├── conf/                  # Hydra configs
│   ├── data/                  # Data processing
│   ├── models/                # Model implementations
│   │   ├── statistical/       # LightGBM, XGBoost, etc.
│   │   └── deep_learning/     # LSTM, Transformer
│   ├── training/              # Training engine
│   ├── evaluation/            # Metrics
│   └── api/                   # REST API
├── projects/                  # Datasets
│   ├── retail_sales_walmart/
│   ├── rideshare_demand_ola/
│   └── ...
├── tests/                     # Test suite
├── docs/                      # Documentation
└── mlruns/                    # MLflow experiments (auto-created)
```

## What to Explore Next

1. **[API Reference](api_reference/)** - Detailed API documentation
2. **[Tutorials](tutorials/)** - Step-by-step guides
3. **[Configuration Guide](guides/configuration.md)** - Advanced config usage
4. **[Model Guide](guides/models.md)** - Model selection and tuning
5. **[Deployment Guide](guides/deployment.md)** - Production deployment

## Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: [GitHub Issues](https://github.com/MrEleden/forecasting_demo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MrEleden/forecasting_demo/discussions)

## Quick Reference Commands

```bash
# Training
python src/ml_portfolio/training/train.py --config-name walmart model=lightgbm

# Optimization
python src/ml_portfolio/scripts/run_optimization.py --models lightgbm --trials 50

# Multi-run comparison
python train.py --config-name walmart --multirun model=lightgbm,xgboost

# View experiments
mlflow ui

# Run tests
pytest tests/ -v

# API server
python src/ml_portfolio/api/main.py
```

Congratulations! You're ready to start forecasting! 🎉
