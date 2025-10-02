# Documentation Index

## 📚 Complete Documentation Guide

### 🚀 Getting Started
- [`README.md`](../README.md) - Project overview and quick start
- [`docs/SETUP.md`](SETUP.md) - Development environment setup
- [`docs/HYDRA_OPTUNA_QUICK_SETUP.md`](HYDRA_OPTUNA_QUICK_SETUP.md) - **NEW** 5 working commands for immediate use

### ⚡ Quick Reference

#### Simple Training Commands
1. **LSTM**: `python src/ml_portfolio/training/train.py model=lstm dataset_factory=walmart optimizer=adam`
2. **Random Forest**: `python src/ml_portfolio/training/train.py model=random_forest dataset_factory=walmart`
3. **ARIMA**: `python src/ml_portfolio/training/train.py model=arima dataset_factory=walmart`
4. **Parameter Override**: `python src/ml_portfolio/training/train.py model=lstm dataset_factory=walmart optimizer=adam model.hidden_size=128 optimizer.lr=0.001`
5. **Multi-Model**: `python src/ml_portfolio/training/train.py -m model=lstm,random_forest,arima dataset_factory=walmart`

### 🔧 Configuration System Overview

#### Current Working Features ✅
- **Simple Default Configs**: Reliable model training with minimal setup
- **MLflow Integration**: Automatic experiment tracking and model registry
- **Config Composition**: Modular configs (model + dataset + optimizer)
- **Command-line Overrides**: Runtime parameter modification
- **Complex Parameters**: List/tuple syntax complications in some contexts

### 🎯 Documentation Priorities

1. **Start Here**: `HYDRA_OPTUNA_QUICK_SETUP.md` for immediate working commands
2. **Setup**: `SETUP.md` for environment configuration (if needed)
3. **Reference**: This index for navigation and status

### 🔄 Recent Updates

#### 2025-10-02 - **Simplified to Default Configurations**
- ✅ Removed complex experiment configurations
- ✅ Updated `HYDRA_OPTUNA_QUICK_SETUP.md` with simple commands
- ✅ Focused on single model training with defaults
- ✅ All commands use straightforward parameter configurations

#### Working Status Summary
- **Simple Training**: Single model runs with reliable defaults
- **MLflow Tracking**: Working seamlessly for all models
- **Command Examples**: All simplified and easy to use
- **Documentation**: Streamlined for immediate productivity

### 🚨 Critical Notes

- **Use Simple Commands**: All complex experiments removed for simplicity
- **Start with LSTM**: Basic LSTM training for system validation
- **Try All Models**: LSTM, Random Forest, and ARIMA available
- **Default parameters**: Reliable configurations without complex tuning

### 📊 Quick Command Summary

| Command | Type | Runtime | Purpose |
|---------|------|---------|---------|
| #1 | LSTM | 2-5 min | Neural network training |
| #2 | Random Forest | 2-5 min | Ensemble baseline |
| #3 | ARIMA | 3-7 min | Statistical modeling |
| #4 | Parameter Override | 2-5 min | Custom parameters |
| #5 | Multi-Model | 5-10 min | Model comparison |

---

**💡 Recommendation**: Start with the `HYDRA_OPTUNA_QUICK_SETUP.md` guide - it contains 5 simple commands for reliable model training with default configurations.
