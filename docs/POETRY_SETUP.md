# Poetry Setup Guide 🚀

This guide provides step-by-step instructions for setting up Poetry in the ML Portfolio Forecasting Demo project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installing Poetry](#installing-poetry)
- [Project Setup](#project-setup)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

### Required Software
- **Python 3.9+**: Ensure Python is installed and accessible via command line
- **Git**: For version control
- **PowerShell** (Windows) or **Bash** (Linux/macOS): For running commands

### Verify Python Installation
```powershell
python --version
# Should output: Python 3.11.x or similar
```

## Installing Poetry

### Method 1: Using pipx (Recommended)

pipx provides isolated environments for Python applications, making it the cleanest way to install Poetry.

#### Step 1: Install pipx
```powershell
# Install pipx using pip
python -m pip install --user pipx

# Ensure pipx is in PATH
python -m pipx ensurepath
```

#### Step 2: Install Poetry with pipx
```powershell
# Install Poetry
python -m pipx install poetry

# Verify installation (use pipx run if PATH not updated)
python -m pipx run poetry --version
# Should output: Poetry (version 2.2.1)

# Alternative: Use poetry directly if PATH is updated
poetry --version
```

**Note**: After installation, you may need to restart your terminal for direct `poetry` command access. If `poetry` command is not found, use `python -m pipx run poetry` instead.

### Method 2: Official Installer (Alternative)

If pipx is not available, use the official installer:

#### Windows (PowerShell)
```powershell
# Download and install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### Linux/macOS (Bash)
```bash
# Download and install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### Method 3: Using pip (Not Recommended)
```powershell
# Only use if other methods fail
pip install poetry
```

## Project Setup

### 1. Navigate to Project Directory
```powershell
cd c:\Users\mvill\github\forecasting_demo
```

### 2. Verify Poetry Configuration
```powershell
# Check Poetry configuration
poetry config --list

# Optional: Configure Poetry to create virtual environments in project directory
poetry config virtualenvs.in-project true
```

### 3. Install Project Dependencies

#### Install All Dependencies
```powershell
# Install all dependencies including optional extras
poetry install --all-extras
```

#### Install Specific Dependency Groups
```powershell
# Core dependencies only
poetry install

# With development tools
poetry install --with dev

# With training dependencies
poetry install --with train

# With app dependencies for dashboards
poetry install --with app

# Custom combinations
poetry install --with dev,train --extras "statistical deep_learning"
```

### 4. Activate Virtual Environment
```powershell
# Activate Poetry shell
poetry shell

# Alternative: Run commands with Poetry prefix
poetry run python --version
```

## Development Workflow

### Common Commands

#### Dependency Management
```powershell
# Add new dependency
poetry add pandas

# Add development dependency
poetry add --group dev pytest

# Add optional dependency
poetry add --optional statsmodels

# Remove dependency
poetry remove pandas

# Update dependencies
poetry update

# Show installed packages
poetry show
poetry show --tree  # Show dependency tree
```

#### Environment Management
```powershell
# Show environment information
poetry env info

# List available environments
poetry env list

# Remove environment
poetry env remove python

# Use specific Python version
poetry env use python3.11
```

#### Running Code
```powershell
# Run Python scripts
poetry run python scripts/train.py

# Run with Hydra configuration
poetry run python projects/retail_sales_walmart/scripts/train.py model=lstm

# Run tests
poetry run pytest tests/

# Run linting
poetry run ruff check src/
poetry run black src/

# Start Streamlit dashboard
poetry run streamlit run app/dashboard.py
```

#### Building and Publishing
```powershell
# Build distribution packages
poetry build

# Check package
poetry check

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
poetry export --with dev -f requirements.txt --output requirements-dev.txt
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Poetry Command Not Found
```powershell
# Add Poetry to PATH manually (Windows)
$env:PATH += ";$env:APPDATA\Python\Scripts"

# Or restart terminal after installation
```

#### 2. Virtual Environment Issues
```powershell
# Clear Poetry cache
poetry cache clear pypi --all

# Remove and recreate environment
poetry env remove python
poetry install
```

#### 3. Dependency Conflicts
```powershell
# Lock dependencies without updating
poetry lock --no-update

# Install without development dependencies
poetry install --only=main

# Check for outdated packages
poetry show --outdated
```

#### 4. Windows-Specific Path Issues
```powershell
# If experiencing long path issues
poetry config virtualenvs.path "C:\venvs"

# Enable long paths in Windows (run as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### 5. SSL/Certificate Errors
```powershell
# Configure Poetry to use system certificates
poetry config certificates.path "path/to/certificates"

# Or disable SSL verification (not recommended for production)
poetry config repositories.pypi.verify-ssl false
```

## Advanced Configuration

### Custom Poetry Configuration

#### Set Global Configurations
```powershell
# Create virtual environments in project directory
poetry config virtualenvs.in-project true

# Set custom cache directory
poetry config cache-dir "C:\Poetry\Cache"

# Configure PyPI repository
poetry config repositories.private-pypi https://private-pypi.company.com/simple/
```

#### Project-Specific Configuration
Create a `poetry.toml` file in your project root:

```toml
[virtualenvs]
in-project = true
path = ".venv"

[repositories]
private = "https://private-pypi.company.com/simple/"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### Integration with IDEs

#### VS Code Integration
1. Install Python extension
2. Select Poetry virtual environment:
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose the Poetry environment (usually in `.venv/Scripts/python.exe`)

#### PyCharm Integration
1. Go to File → Settings → Project → Python Interpreter
2. Add New Interpreter → Poetry Environment
3. Select existing Poetry environment

### Docker Integration

Example `Dockerfile` with Poetry:

```dockerfile
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY src/ ./src/

# Run application
CMD ["poetry", "run", "python", "-m", "src.ml_portfolio"]
```

## Best Practices

### 1. Version Management
- Always commit `poetry.lock` to version control
- Use semantic versioning for your package
- Pin critical dependencies to specific versions

### 2. Dependency Organization
- Use dependency groups to organize dependencies by purpose
- Keep optional dependencies minimal for faster installs
- Regularly update and review dependencies

### 3. Environment Isolation
- Use `poetry shell` for development
- Never install packages with pip in Poetry environments
- Use `poetry run` for one-off commands

### 4. CI/CD Integration
```yaml
# GitHub Actions example
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install Poetry
      uses: snok/install-poetry@v1
    - name: Install dependencies
      run: poetry install
    - name: Run tests
      run: poetry run pytest
```

## Quick Reference

### Essential Commands
```powershell
# Project setup
poetry install                    # Install dependencies
poetry shell                      # Activate environment

# Development
poetry add <package>               # Add dependency
poetry run <command>               # Run in environment
poetry show                        # List packages

# Building
poetry build                       # Create distributions
poetry check                       # Validate configuration

# Maintenance
poetry update                      # Update dependencies
poetry env remove python          # Reset environment
```

## Additional Resources

- [Official Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry GitHub Repository](https://github.com/python-poetry/poetry)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [Dependency Specification](https://python-poetry.org/docs/dependency-specification/)

---

*This guide is specifically tailored for the ML Portfolio Forecasting Demo project. For project-specific configurations, refer to the `pyproject.toml` file in the repository root.*