# Requirements.txt Setup Guide

This project uses requirements.txt files for dependency management instead of Poetry for simplicity.

## 📁 Requirements Files

- **`requirements.txt`** - Core dependencies (numpy, pandas, hydra, etc.)
- **`requirements-dev.txt`** - Development tools (streamlit, jupyter, testing)
- **`requirements-ml.txt`** - Machine learning extras (statsmodels, xgboost, optuna)

## 🚀 Setup Instructions

### 1. Create Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Alternative if execution policy blocked
.venv\Scripts\python.exe

# Or use Command Prompt style
.venv\Scripts\activate.bat

# Activate on Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies

#### Core Dependencies Only
```powershell
pip install -r requirements.txt
```

#### Development Environment
```powershell
pip install -r requirements-dev.txt
```

#### Full ML Environment
```powershell
pip install -r requirements-ml.txt
```

#### All Dependencies
```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-ml.txt
```

### 3. Install Project in Development Mode
```powershell
pip install -e .
```

## 📝 Adding New Dependencies

### Add to Core
Edit `requirements.txt` and add the package:
```
new-package>=1.0.0
```

### Add to Development
Edit `requirements-dev.txt` and add the package:
```
new-dev-package>=1.0.0
```

### Pin Exact Versions (Production)
Generate a lock file with exact versions:
```powershell
pip freeze > requirements-lock.txt
```

## 🔄 Updating Dependencies

```powershell
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade package-name

# Check outdated packages
pip list --outdated
```

## 🧪 Testing the Setup

```powershell
# Test core imports (using venv Python directly)
.venv\Scripts\python.exe -c "import numpy as np, pandas as pd, hydra; print('✅ Core dependencies working!')"

# Test development tools
.venv\Scripts\python.exe -c "import streamlit, jupyter; print('✅ Dev dependencies working!')"

# Test ML packages (if installed)
.venv\Scripts\python.exe -c "import sklearn, statsmodels; print('✅ ML dependencies working!')"

# Test optuna compatibility
.venv\Scripts\python.exe -c "import optuna; print(f'✅ Optuna version: {optuna.__version__}')"
```

