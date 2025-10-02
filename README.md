# 📊 ML Portfolio - Forecasting Demo

> **Professional time series forecasting portfolio showcasing multiple domains, architectures, and production-ready MLOps patterns**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-blue)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This repository demonstrates **professional-grade time series forecasting** across four diverse domains:

| Domain | Dataset | Models | Key Challenge |
|--------|---------|--------|---------------|
| **🏪 Retail Sales** | Walmart (6.4K records) | ARIMA, Prophet, LSTM | Store-level demand, holiday effects |
| **🚴 Ride-sharing** | Ola Demand (52.5K records) | TCN, Transformer | Peak hours, weather impact |
| **📦 Inventory** | Multi-SKU (3.1K records) | Hierarchical models | Category patterns, promotions |
| **🚛 Transportation** | TSI Economic (307 records) | Statistical models | Economic cycles, seasonality |

## 🏗️ Architecture

```
ml-portfolio/
├── 📁 src/ml_portfolio/          # Shared forecasting library
│   ├── data/                     # PyTorch datasets, loaders, transforms
│   ├── models/                   # LSTM, TCN, Transformer implementations
│   ├── training/                 # Training loops, callbacks, metrics
│   └── evaluation/               # Backtesting, time series CV
├── 📁 projects/                  # Self-contained domain projects
│   ├── retail_sales_walmart/     # Store sales forecasting
│   ├── rideshare_demand_ola/     # Urban mobility demand
│   ├── inventory_forecasting/    # Supply chain optimization
│   └── transportation_tsi/       # Economic indicator prediction
├── 📁 docs/                      # Documentation and guides
└── 📁 apps/                      # Streamlit dashboards, FastAPI
```

### **Key Design Principles**
- **🔧 Modular**: Reusable components across projects
- **⚙️ Configurable**: Hydra-based object instantiation
- **🎯 Optimized**: Optuna hyperparameter tuning
- **📊 Production-Ready**: Docker, monitoring, deployment
- **🔄 Reproducible**: Experiment tracking, version control

## 🚀 Quick Start

### **1. Setup Environment**
```bash
# Clone repository
git clone https://github.com/MrEleden/forecasting_demo.git
cd forecasting_demo

# Install dependencies (Poetry recommended)
poetry install
poetry shell

# Alternative: pip install
pip install -r requirements/base.txt
```

### **2. Download Datasets**
```bash
# Download all datasets
python download_all_data.py

# Or individual datasets
python download_all_data.py --dataset walmart
python download_all_data.py --dataset ola
python download_all_data.py --dataset inventory
python download_all_data.py --dataset tsi
```

### **3. Explore Projects**
```bash
# Walmart retail sales
cd projects/retail_sales_walmart
jupyter notebook notebooks/01_eda.ipynb

# Ola ride-sharing
cd projects/rideshare_demand_ola
python scripts/generate_data.py

# View all project READMEs for specific instructions
```

## 📊 Datasets & Models

### **🏪 Walmart Retail Sales**
- **Data**: 45 stores, 6.4K weekly records (2010-2012)
- **Features**: Store ID, sales, weather, fuel price, holidays
- **Models**: ARIMA baselines → Prophet → LSTM/TCN → Ensemble
- **Challenge**: Store hierarchy, promotional effects, seasonality

### **🚴 Ola Ride-sharing Demand**
- **Data**: 12 zones, 52.5K hourly records (2022-2023)
- **Features**: Location, weather, time patterns, festivals
- **Models**: Prophet → TCN → Transformer → Multi-zone
- **Challenge**: Peak hour prediction, weather impact, city patterns

### **📦 Inventory Forecasting**
- **Data**: 20 SKUs, 5 categories, 3.1K weekly records (2021-2023)
- **Features**: Product hierarchy, pricing, promotions
- **Models**: Hierarchical reconciliation → XGBoost → Deep learning
- **Challenge**: Cross-category effects, price elasticity, new products

### **🚛 Transportation Services Index**
- **Data**: 307 monthly records (2000-2025), 66 economic indicators
- **Features**: Freight, passenger, air, rail, truck transportation
- **Models**: ARIMA-GARCH → VAR → State space models
- **Challenge**: Economic cycles, policy impacts, long-term trends

## 🔧 Technical Stack

### **Core Technologies**
- **🐍 Python 3.9+**: Modern language features
- **📦 Poetry**: Dependency management and packaging
- **⚙️ Hydra**: Configuration management and experimentation
- **🎯 Optuna**: Hyperparameter optimization with pruning
- **🔥 PyTorch**: Deep learning models (LSTM, TCN, Transformer)

### **Data & ML Libraries**
- **📊 Pandas/NumPy**: Data manipulation and analysis
- **📈 Statsmodels**: Statistical forecasting (ARIMA, GARCH)
- **🔮 Prophet**: Automated time series forecasting
- **🚀 XGBoost/LightGBM**: Gradient boosting models
- **🧪 Scikit-learn**: ML pipelines and evaluation

### **Deployment & Monitoring**
- **🌐 Streamlit**: Interactive dashboards
- **🔌 FastAPI**: Production REST APIs
- **🐳 Docker**: Containerization
- ** MLflow**: Experiment tracking (optional)

## 📈 Model Development Workflow

### **1. Hydra-Configured Training**
```bash
# Single model training
poetry run python projects/retail_sales_walmart/scripts/train.py \
    model=lstm dataset=walmart optimizer=adam

# Multi-run experiments
poetry run python scripts/train.py -m \
    model=lstm,tcn,transformer \
    optimizer=adam,adamw \
    dataset.lookback_window=24,48,168
```

### **2. Hyperparameter Optimization**
```bash
# Optuna study with pruning
poetry run python scripts/optimize.py \
    --config-path conf --config-name config \
    hydra.sweep.dir=outputs/optuna_study
```

### **3. Model Evaluation**
```bash
# Backtesting with time series CV
poetry run python scripts/evaluate.py \
    --study-name walmart_optimization \
    --metrics wmae,mape,directional_accuracy
```

### **4. Dashboard Deployment**
```bash
# Interactive Streamlit app
poetry run streamlit run app/dashboard.py

# Production API
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🐳 Docker Deployment

```bash
# Build container
docker build -t forecasting-demo .

# Run with specific dataset
docker run -p 8501:8501 -e DATASET=walmart forecasting-demo

# Production deployment
docker-compose up -d
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Data Download Guide](docs/DATA_DOWNLOAD_README.md)** | Dataset setup and requirements |
| **[Kaggle Setup](docs/KAGGLE_SETUP.md)** | API configuration for Walmart data |
| **[Dataset Status](docs/DATASET_STATUS.md)** | Current data availability report |
| **[Copilot Instructions](.github/copilot-instructions.md)** | AI development guidelines |

### **Project-Specific Guides**
- **[Walmart README](projects/retail_sales_walmart/README.md)**: Retail forecasting
- **[Ola README](projects/rideshare_demand_ola/README.md)**: Urban mobility
- **[Inventory README](projects/inventory_forecasting/README.md)**: Supply chain
- **[TSI README](projects/transportation_tsi/README.md)**: Economic indicators

## 🎯 Performance Targets

| Dataset | Primary Metric | Target | Production Threshold |
|---------|----------------|--------|---------------------|
| Walmart | WMAE | < 15% | < 20% |
| Ola | MAPE | < 25% | < 30% |
| Inventory | MAPE | < 20% | < 25% |
| TSI | RMSE | < 5.0 | < 8.0 |

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-model`)
3. **Follow** coding standards (black, ruff, type hints)
4. **Add** tests for new functionality
5. **Submit** pull request with clear description

### **Development Standards**
```bash
# Code formatting
poetry run black src/ tests/
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/

# Testing
poetry run pytest tests/ -v
```

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🌟 Showcase Features

- **✅ Production Architecture**: Separates reusable library from domain projects
- **✅ Modern MLOps**: Hydra configs, Optuna optimization, experiment tracking
- **✅ Diverse Domains**: Retail, mobility, supply chain, economic forecasting
- **✅ Multiple Approaches**: Statistical, ML, deep learning, ensemble methods
- **✅ Deployment Ready**: Docker, APIs, dashboards, monitoring
- **✅ Professional Quality**: Testing, documentation, CI/CD, type hints

---

**🚀 Ready to explore professional time series forecasting? Start with the [Quick Start](#-quick-start) guide!**