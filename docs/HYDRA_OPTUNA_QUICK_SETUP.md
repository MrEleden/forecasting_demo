# Quick Training Guide

## 🚀 5 Simple Commands - Ready to Run

This guide provides 5 tested and verified commands for basic model training using default configurations. All commands use the simplified setup without complex experiment configurations.

**✅ Latest Status**: All commands use simple default configurations for quick and reliable training.

### ✅ Command 1: Basic LSTM Training
```bash
python src/ml_portfolio/training/train.py model=lstm dataset_factory=walmart optimizer=adam
```
**Result**: Single LSTM model training with default hyperparameters
**Runtime**: ~2-5 minutes with GPU
**Use Case**: Quick LSTM model training and validation

### ✅ Command 2: Random Forest Training
```bash
python src/ml_portfolio/training/train.py model=random_forest dataset_factory=walmart
```
**Result**: Single Random Forest model with default parameters
**Runtime**: ~2-5 minutes (CPU-based)
**Use Case**: Fast baseline model training

### ✅ Command 3: ARIMA Training
```bash
python src/ml_portfolio/training/train.py model=arima dataset_factory=walmart
```
**Result**: ARIMA model with default seasonal parameters
**Runtime**: ~3-7 minutes
**Use Case**: Statistical baseline model

### ✅ Command 4: Parameter Override Example
```bash
python src/ml_portfolio/training/train.py model=lstm dataset_factory=walmart optimizer=adam model.hidden_size=128 optimizer.lr=0.001
```
**Result**: LSTM with custom hidden size and learning rate
**Runtime**: ~2-5 minutes with GPU
**Use Case**: Quick parameter testing without complex sweeps

### ✅ Command 5: Multi-Model Comparison
```bash
python src/ml_portfolio/training/train.py -m model=lstm,random_forest,arima dataset_factory=walmart
```
**Result**: 3 jobs (3 models) - Simple architecture comparison
**Runtime**: ~5-10 minutes
**Use Case**: Quick model comparison with default parameters

### ✅ Command 5: ARIMA Parameter Optimization
## 📊 Command Results Summary

| Command | Type | Runtime | Best For |
|---------|------|---------|----------|
| #1 LSTM | Single Model | 2-5 min | Quick LSTM validation |
| #2 Random Forest | Single Model | 2-5 min | Fast baseline |
| #3 ARIMA | Single Model | 3-7 min | Statistical modeling |
| #4 Parameter Override | Single Model | 2-5 min | Custom parameters |
| #5 Multi-Model | 3 Models | 5-10 min | Model comparison |

## 🔧 Current System Status

### ✅ Working Features
- **Simple Training**: Single model runs with default configurations
- **MLflow Tracking**: Automatic experiment logging for all jobs
- **CUDA Support**: GPU acceleration for deep learning models (LSTM)
- **Walmart Dataset**: 17-feature dataset with proper preprocessing
- **Config Override**: Command-line parameter modification

### ✅ Available Models
- **LSTM**: Deep learning time series forecasting
- **Random Forest**: Ensemble learning with sklearn
- **ARIMA**: Statistical time series modeling

## 🚀 Quick Start Workflow

### 1. Choose Your Command
- **New to system**: Start with Command #1 (LSTM)
- **Fast baseline**: Try Command #2 (Random Forest)
- **Statistical approach**: Use Command #3 (ARIMA)
- **Custom parameters**: Try Command #4 (Parameter Override)
- **Model comparison**: Run Command #5 (Multi-Model)

### 2. Monitor Training
```bash
# Check MLflow UI for results
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

**Expected Output:**
- Single model training with progress logging
- `Training LSTM on device: cuda` - GPU acceleration (LSTM only)
- Automatic MLflow experiment tracking
- `Created MLflow autologging run` - sklearn autologging active (Random Forest, XGBoost)
- `MLflow experiment: ml_portfolio_forecasting` - Experiment tracking enabled
- Individual job progress logs in terminal

### 3. Analyze Results
- **Best Models**: Sort by MAPE or RMSE in MLflow UI
## 📁 Output Structure

Each command generates organized outputs:
```
outputs/YYYY-MM-DD/HH-MM-SS/
├── .hydra/                # Configuration files
├── model_weights.pt       # Saved model (for LSTM)
└── training.log           # Training logs
```

## 🔍 Troubleshooting

### Command Not Starting
```bash
# Verify Python environment
python --version

# Check if modules are available
python -c "import hydra, mlflow, torch"

# Validate configuration
python src/ml_portfolio/training/train.py --help
```

### Common Solutions
- **Model not found**: Use available models (lstm, random_forest, arima)
- **GPU issues**: LSTM automatically uses CUDA if available
- **Config errors**: Check parameter names match model configuration

## 🎯 Next Steps

1. **Run Command #1** (LSTM) to verify system works
2. **Check MLflow UI** for experiment tracking
3. **Try Command #2** (Random Forest) for fast baseline
4. **Compare models** with Command #5 (Multi-Model)
5. **Experiment with Command #4** for custom parameters

## 📚 Related Documentation

- [`docs/DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) - Full documentation overview
- [`README.md`](../README.md) - Project overview and setup

---

**💡 Pro Tip**: All 5 commands use simple default configurations for reliable, fast training. Start with Command #1, then explore other models based on your specific needs.
