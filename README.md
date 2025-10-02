# 📈 ML Portfolio: Time Series Forecasting

Professional ML portfolio showcasing time series forecasting across multiple domains with modern MLOps practices.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd forecasting_demo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

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

```yaml
# Single run
python src/ml_portfolio/training/train.py \
  model=lstm \
  dataset_factory=walmart \
  optimizer=adam \
  mlflow=production

# Hyperparameter sweep
python src/ml_portfolio/training/train.py -m \
  model=lstm \
  optimizer=adam,adamw \
  optimizer.lr=0.001,0.01,0.1
```

## 📦 MLflow Integration

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

## 🔬 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

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
