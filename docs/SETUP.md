# Environment Setup Guide

Complete guide for setting up the ML Forecasting Portfolio development environment.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- Virtual environment tool (venv, virtualenv, or conda)

## Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/forecasting_demo.git
cd forecasting_demo
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

The project has three requirement files for different use cases:

#### **Base Installation** (Core Dependencies Only)
```bash
pip install -r requirements.txt
```
Includes: pandas, numpy, scikit-learn, matplotlib, hydra-core

#### **ML Installation** (Machine Learning Models)
```bash
pip install -r requirements-ml.txt
```
Includes: statsmodels, torch, xgboost, lightgbm, optuna, prophet

#### **Development Installation** (Full Setup)
```bash
pip install -r requirements-dev.txt
```
Includes: All ML dependencies + jupyter, black, pytest, pre-commit

## Requirements Files Overview

### `requirements.txt` - Core Dependencies
Essential libraries for basic functionality:
- Data processing: numpy, pandas
- Configuration: hydra-core, omegaconf
- Visualization: matplotlib, seaborn, plotly
- Utilities: tqdm, pyyaml, joblib

**When to use**: Lightweight installations, CI/CD pipelines, production deployments

### `requirements-ml.txt` - Machine Learning
Statistical and deep learning models:
- Statistical: statsmodels, prophet
- ML models: xgboost, lightgbm, catboost
- Deep learning: torch
- Hyperparameter tuning: optuna

**When to use**: Training models, running experiments, full forecasting functionality

### `requirements-dev.txt` - Development Tools
Development and testing tools:
- Notebooks: jupyter, ipykernel
- Code quality: black, ruff, pycodestyle, pre-commit
- Testing: pytest, pytest-cov
- Documentation: sphinx, myst-parser

**When to use**: Development, contributing, testing, documentation

## Step-by-Step Installation

### For Users (Running Models)
```bash
# 1. Clone and create environment
git clone <repo-url>
cd forecasting_demo
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install ML dependencies
pip install -r requirements-ml.txt

# 3. Download datasets
python download_all_data.py

# 4. Verify installation
python -c "import pandas, torch, statsmodels; print('Setup successful!')"
```

### For Developers (Contributing)
```bash
# 1. Clone and create environment
git clone <repo-url>
cd forecasting_demo
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install all dependencies
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Run tests to verify
pytest tests/ -v

# 5. Check code quality
black --check src/ tests/
pycodestyle src/ tests/
```

## Troubleshooting

### Prophet Installation Issues
Prophet can be problematic on some systems. If installation fails:

**Option 1: Skip Prophet**
```bash
# Edit requirements-ml.txt and comment out:
# prophet>=1.1.4

# Or install manually after:
pip install prophet
```

**Option 2: Use conda**
```bash
conda install -c conda-forge prophet
```

### PyTorch Installation Issues
For specific CUDA versions or CPU-only:

**CPU-only (smaller download)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Windows-Specific Issues

**Long path names**
```bash
git config --system core.longpaths true
```

**Visual C++ Build Tools** (required for some packages)
Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Verifying Installation

### Check Core Dependencies
```bash
python -c "import pandas, numpy, hydra; print('Core dependencies OK')"
```

### Check ML Dependencies
```bash
python -c "import torch, statsmodels, xgboost; print('ML dependencies OK')"
```

### Check Development Tools
```bash
black --version
pytest --version
pre-commit --version
```

### Run Sample Training
```bash
# Test training pipeline
python src/ml_portfolio/training/train.py dataset_factory=walmart model=arima
```

## Development Workflow

### Before Starting Work
```bash
# Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate

# Update dependencies if needed
pip install -r requirements-dev.txt --upgrade
```

### Before Committing
Pre-commit hooks will automatically run, but you can test manually:
```bash
# Format code
black src/ tests/

# Check style
pycodestyle src/ tests/

# Run tests
pytest tests/ -v

# Or let pre-commit handle it
pre-commit run --all-files
```

## Environment Management

### Deactivate Virtual Environment
```bash
deactivate
```

### Remove Virtual Environment
```bash
# Windows
rmdir /s .venv

# Linux/Mac
rm -rf .venv
```

### Recreate Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Updating Dependencies

### Update All Packages
```bash
pip install -r requirements-dev.txt --upgrade
```

### Update Specific Package
```bash
pip install --upgrade pandas
```

### Generate New Requirements
```bash
pip freeze > requirements-freeze.txt
```

## Alternative: Using Conda

### Create Conda Environment
```bash
conda create -n forecasting python=3.11
conda activate forecasting
pip install -r requirements-dev.txt
```

### Export Conda Environment
```bash
conda env export > environment.yml
```

## Next Steps

After successful setup:
1. Read [README.md](../README.md) for project overview
2. Download datasets: `python download_all_data.py`
3. Explore notebooks in `projects/*/notebooks/`
4. Review [copilot-instructions.md](../.github/copilot-instructions.md) for development guidelines

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with details:
   - Operating system and version
   - Python version (`python --version`)
   - Full error message
   - Installation command used
