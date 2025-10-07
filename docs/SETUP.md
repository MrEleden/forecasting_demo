# Environment Setup Guide

Complete guide for setting up the ML Forecasting Portfolio development environment.

## Prerequisites

- Python 3.9 or higher (Python 3.11 recommended)
- pip (Python package manager)
- Git
- Virtual environment tool (venv, virtualenv, or conda)

## TL;DR - Quick Start

For experienced developers who just want to get started:

```bash
# Clone and setup (macOS/Linux)
git clone <repo-url> && cd forecasting_demo
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

# Verify
pytest tests/ -v --cov
```

**Critical**: Don't forget `pip install -e .` or imports will fail!

## Important Notes

- **Python 3.11 is recommended** for best compatibility with all dependencies
- The package must be installed in **editable mode** (`pip install -e .`) for tests to work
- Known issue: `optuna>=3.0.0` conflicts with `hydra-optuna-sweeper`, version is constrained to `<3.0.0`

## Detailed Setup Guide

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/forecasting_demo.git
cd forecasting_demo
```

### 2. Create Virtual Environment with Python 3.11
```bash
# Using Python 3.11 (recommended)
# On macOS with Homebrew:
/opt/homebrew/bin/python3.11 -m venv .venv

# Or if python3.11 is in PATH:
python3.11 -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

**IMPORTANT**: After installing dependencies, you must install the package in editable mode for imports to work.

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

# CRITICAL: Install package in editable mode (required for tests and imports)
pip install -e .
```
Includes: All ML dependencies + jupyter, black, pytest, pre-commit

**Note**: The `pip install -e .` command installs the `ml_portfolio` package in editable mode, making it importable in tests and scripts.

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

### `requirements-gpu.txt` - GPU Acceleration (OPTIONAL)
PyTorch with CUDA support for GPU-accelerated deep learning:
- PyTorch with CUDA 12.1
- torchvision with CUDA 12.1
- torchaudio with CUDA 12.1

**When to use**: Training LSTM/Transformer models on NVIDIA GPU (2-3x faster)
**Requirements**: NVIDIA GPU with CUDA support (RTX 20/30/40 series, Tesla, etc.)

## Step-by-Step Installation

### For Users (Running Models)
```bash
# 1. Clone and create environment
git clone <repo-url>
cd forecasting_demo
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install ML dependencies
pip install -r requirements-ml.txt

# 3. Install package in editable mode
pip install -e .

# 4. Download datasets
python download_all_data.py

# 5. Verify installation
python -c "import pandas, torch, statsmodels, ml_portfolio; print('Setup successful!')"
```

### For Developers (Contributing)
```bash
# 1. Clone and create environment
git clone <repo-url>
cd forecasting_demo
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install all dependencies
pip install -r requirements-dev.txt

# 3. CRITICAL: Install package in editable mode (required for tests)
pip install -e .

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify
pytest tests/ -v --cov

# 6. Check code quality
black --check src/ tests/
pycodestyle src/ tests/
```

### For GPU Users (Deep Learning Acceleration)
If you have an NVIDIA GPU and want 2-3x faster LSTM/Transformer training:

```bash
# 1. Check GPU availability
nvidia-smi

# 2. Clone and create environment
git clone <repo-url>
cd forecasting_demo
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install ML dependencies (CPU version first)
pip install -r requirements-ml.txt

# 4. Install GPU-accelerated PyTorch
pip install -r requirements-gpu.txt

# 5. Install package in editable mode
pip install -e .

# 6. Verify GPU is detected
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 7. Test GPU training (should show "Training LSTM on device: cuda")
python src/ml_portfolio/training/train.py dataset_factory=walmart model=lstm dataloader=pytorch training.max_epochs=2
```

**Expected output**: Training logs should show `Training LSTM on device: cuda`

## Troubleshooting

### Module Import Errors (`ModuleNotFoundError: No module named 'ml_portfolio'`)

**Symptom**: Tests fail with import errors for `ml_portfolio` module

**Solution**: Install the package in editable mode
```bash
pip install -e .
```

This installs the `src/ml_portfolio` package so it can be imported by tests and scripts. Without this, Python cannot find the module.

### Pytest Marker Errors (`'slow' not found in markers configuration`)

**Symptom**: Pytest reports marker configuration errors

**Solution**: The markers are defined in `pyproject.toml`. If you see this error, check that your `pyproject.toml` includes:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

### Optuna Version Conflict

**Symptom**: pip reports dependency conflict between `optuna>=3.0.0` and `hydra-optuna-sweeper`

**Solution**: This is a known issue. The `requirements.txt` file constrains optuna to `<3.0.0`:
```
optuna>=2.10.0,<3.0.0
hydra-optuna-sweeper>=1.2.0
```

If you need optuna 3.x features, you may need to wait for `hydra-optuna-sweeper` to update compatibility.

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

### PyTorch/GPU Installation Issues

**GPU not detected after installation**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install -r requirements-gpu.txt

# Verify CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**CUDA version mismatch**
The driver CUDA version (from `nvidia-smi`) should be >= PyTorch CUDA version (12.1).
- Driver: 11.x → Use PyTorch CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- Driver: 12.x+ → Use PyTorch CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

**CPU-only (no GPU / smaller download)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Out of memory during GPU training**
Reduce batch size in config:
```bash
python src/ml_portfolio/training/train.py model=lstm dataloader.batch_size=32
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

### Check Package Installation
```bash
python -c "import ml_portfolio; print('ml_portfolio package OK')"
```

### Check Development Tools
```bash
black --version
pytest --version
pre-commit --version
```

### Run Tests
```bash
# Run all tests with coverage
pytest tests/ -v --cov

# Run specific test file
pytest tests/unit/test_metrics.py -v

# Run tests excluding slow tests
pytest tests/ -v -m "not slow"
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
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .  # Don't forget this!
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
pip install -e .
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

## Common Issues Quick Reference

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ml_portfolio'` | Run `pip install -e .` |
| Pytest marker errors | Check `pyproject.toml` has markers configured |
| Optuna dependency conflict | Constraint to `optuna>=2.10.0,<3.0.0` in requirements.txt |
| Tests fail to import | Make sure virtual environment is activated and package is installed |
| Import errors in notebooks | Restart kernel after `pip install -e .` |

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Check the [Common Issues Quick Reference](#common-issues-quick-reference)
3. Search existing GitHub issues
4. Create a new issue with details:
   - Operating system and version
   - Python version (`python --version`)
   - Full error message
   - Installation command used
