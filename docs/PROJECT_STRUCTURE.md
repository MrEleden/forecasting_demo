# 🏗️ Project Structure Guide

Standardized folder structure enforced across all forecasting projects in the ML Portfolio.

## 🎯 Overview

This project enforces a **mandatory 22-folder structure** for consistency, maintainability, and CI/CD automation. Every project must follow this exact structure to pass validation checks.

## 📁 Standard Project Structure

```
projects/<project_name>/
├── README.md                 # Project-specific documentation
├── api/                      # FastAPI endpoints and REST services
├── app/                      # Streamlit dashboard and web interfaces  
├── conf/                     # Hydra configuration management
│   ├── config.yaml          # Main project configuration
│   ├── dataset/             # Dataset-specific configurations
│   ├── hydra/               # Hydra runtime settings
│   ├── model/               # Model configurations (ARIMA, LSTM, etc.)
│   ├── optimizer/           # Optimizer configurations (Adam, SGD, etc.)
│   └── scheduler/           # Learning rate scheduler configurations
├── data/                    # DVC-style data layout
│   ├── external/            # External/third-party reference data
│   ├── interim/             # Intermediate processed data
│   ├── processed/           # Final analysis-ready data
│   └── raw/                 # Original, immutable data
├── models/                  # Trained models and artifacts
│   ├── artifacts/           # Model artifacts (scalers, encoders, etc.)
│   └── checkpoints/         # Model checkpoints during training
├── notebooks/               # Jupyter notebooks for exploration
├── reports/                 # Generated analysis reports
│   └── figures/             # Generated plots and visualizations
├── scripts/                 # Python scripts (training, data download)
└── tests/                   # Unit and integration tests
```

## 🔧 Folder Descriptions

### **📄 README.md**
- **Purpose**: Project-specific documentation
- **Content**: Setup instructions, data sources, model details
- **Required**: Yes
- **Template**: Available in `docs/templates/`

### **🌐 api/**
- **Purpose**: FastAPI endpoints and REST API services
- **Content**: Model serving endpoints, health checks, validation
- **Structure**:
  ```
  api/
  ├── __init__.py
  ├── main.py              # FastAPI application entry point
  ├── endpoints/           # API route definitions
  ├── models/              # Pydantic request/response models
  └── dependencies.py      # Shared dependencies and middleware
  ```

### **📱 app/**
- **Purpose**: Streamlit dashboard and web interfaces
- **Content**: Interactive visualizations, model comparison, data exploration
- **Structure**:
  ```
  app/
  ├── dashboard.py         # Main Streamlit application
  ├── pages/               # Multi-page applications
  ├── components/          # Reusable UI components
  └── utils.py             # Dashboard-specific utilities
  ```

### **⚙️ conf/**
- **Purpose**: Hydra configuration management
- **Content**: YAML configurations for reproducible experiments
- **Required Structure**:
  ```
  conf/
  ├── config.yaml          # Main configuration file
  ├── dataset/
  │   ├── walmart.yaml     # Dataset-specific configs
  │   └── synthetic.yaml
  ├── hydra/
  │   └── default.yaml     # Hydra runtime settings
  ├── model/
  │   ├── arima.yaml       # Statistical models
  │   ├── lstm.yaml        # Deep learning models
  │   └── prophet.yaml
  ├── optimizer/
  │   ├── adam.yaml        # Optimizer configurations
  │   └── sgd.yaml
  └── scheduler/
      ├── step_lr.yaml     # Learning rate schedulers
      └── cosine.yaml
  ```

### **📊 data/**
- **Purpose**: DVC-style data pipeline organization
- **Content**: All project data in stages of processing
- **Required Structure**:
  ```
  data/
  ├── external/            # Reference data, lookups, external APIs
  ├── interim/             # Intermediate processed data
  ├── processed/           # Final analysis-ready datasets
  └── raw/                 # Original, immutable source data
  ```

#### **Data Stage Guidelines**
- **raw/**: Never modify, always immutable source data
- **interim/**: Temporary processing outputs, safe to delete/regenerate
- **processed/**: Final clean data ready for modeling
- **external/**: Reference data from external sources

### **🧠 models/**
- **Purpose**: Trained model storage and artifacts
- **Content**: Model files, preprocessing artifacts, metadata
- **Required Structure**:
  ```
  models/
  ├── artifacts/           # Scalers, encoders, feature transformers
  └── checkpoints/         # Model training checkpoints and final models
  ```

### **📓 notebooks/**
- **Purpose**: Jupyter notebooks for exploration and analysis
- **Content**: EDA, experimentation, prototyping, visualization
- **Naming Convention**: 
  ```
  01-data-exploration.ipynb
  02-feature-engineering.ipynb  
  03-model-comparison.ipynb
  04-results-analysis.ipynb
  ```

### **📈 reports/**
- **Purpose**: Generated analysis outputs and documentation
- **Content**: Automated reports, performance summaries, visualizations
- **Required Structure**:
  ```
  reports/
  ├── figures/             # Generated plots, charts, visualizations
  ├── performance_summary.md
  └── model_comparison.html
  ```

### **🔧 scripts/**
- **Purpose**: Executable Python scripts for automation
- **Content**: Training scripts, data download, preprocessing, evaluation
- **Common Files**:
  ```
  scripts/
  ├── download_data.py     # Data acquisition
  ├── train.py             # Model training with Hydra
  ├── evaluate.py          # Model evaluation
  ├── optimize.py          # Hyperparameter optimization
  └── generate_data.py     # Synthetic data generation (if applicable)
  ```

### **🧪 tests/**
- **Purpose**: Unit and integration tests
- **Content**: Test files for scripts, models, and data processing
- **Structure**:
  ```
  tests/
  ├── __init__.py
  ├── test_data_loading.py
  ├── test_models.py
  └── test_preprocessing.py
  ```

## 🔒 Structure Enforcement

### **CI/CD Validation**
The project structure is **automatically validated** by CI/CD pipelines:

```yaml
# .github/workflows/validate-structure.yml
- name: Validate Project Structure
  run: python ci/scripts/validate_structure.py
```

### **Validation Script**
```bash
# Manual validation
python ci/scripts/validate_structure.py

# Fix missing folders automatically
python ci/scripts/validate_structure.py --fix

# Check specific project
python ci/scripts/validate_structure.py --project retail_sales_walmart
```

### **Required Files**
Each folder **must contain** either:
- **Content files** (Python scripts, YAML configs, data files)
- **`.gitkeep` file** to maintain folder structure in Git

The validation script automatically creates `.gitkeep` files for empty folders.

## 📋 Configuration Standards

### **Hydra Configuration Keys**
All projects must use **consistent configuration naming**:

```yaml
# config.yaml structure
defaults:
  - dataset: walmart           # Dataset configuration
  - model: arima              # Model configuration  
  - optimizer: adam           # Optimizer configuration
  - scheduler: step_lr        # LR scheduler configuration

experiment:
  name: "walmart_arima_baseline"
  tags: ["baseline", "arima"]
  
hydra:
  run:
    dir: outputs/${experiment.name}
```

### **Dataset Configuration**
```yaml
# dataset/walmart.yaml
_target_: ml_portfolio.data.loaders.WalmartDataLoader
file_path: "data/raw/train.csv"
date_column: "Date"
target_column: "Weekly_Sales"
train_size: 0.8
validation_size: 0.1
test_size: 0.1
```

### **Model Configuration**
```yaml
# model/arima.yaml  
_target_: ml_portfolio.models.statistical.ARIMAWrapper
order: [1, 1, 1]
seasonal_order: [0, 0, 0, 0]
trend: null
enforce_stationarity: true
```

## 🚀 Project Initialization

### **Creating New Project**
```bash
# Use project template
python scripts/create_project.py --name new_forecasting_project

# Or copy existing structure
cp -r projects/retail_sales_walmart projects/new_project
# Then clean and customize content
```

### **Template Files**
Templates are available in `docs/templates/`:
- `README_template.md`
- `config_template.yaml`  
- `download_data_template.py`
- `train_template.py`

## 🔄 Best Practices

### **Naming Conventions**
- **Projects**: `lowercase_with_underscores`
- **Scripts**: `descriptive_action.py` (e.g., `train_arima.py`)
- **Configs**: `model_name.yaml` (e.g., `lstm.yaml`)
- **Notebooks**: `##-descriptive-name.ipynb`

### **File Organization**
- **Keep raw data immutable** - never modify files in `data/raw/`
- **Use interim for processing** - temporary files go in `data/interim/`
- **Document configurations** - every YAML should have comments
- **Version models** - use timestamps or version numbers in model names

### **Git Integration**
```gitignore
# Automatically ignored (already in .gitignore)
data/raw/                    # Large source files
data/interim/                # Temporary processing files  
models/checkpoints/          # Large model files
*.pyc                        # Python cache
__pycache__/                 # Python cache directories
.hydra/                      # Hydra output folders
outputs/                     # Experiment outputs
```

## 🔍 Validation Details

### **Validation Checks**
The structure validator performs these checks:

1. **✅ Folder Existence**: All 22 required folders present
2. **✅ File Requirements**: README.md exists and is non-empty  
3. **✅ Configuration Structure**: Hydra configs follow standard layout
4. **✅ Git Integration**: `.gitkeep` files in empty folders
5. **✅ Naming Compliance**: Folder names match requirements

### **Error Messages**
```bash
# Example validation output
❌ Missing folder: projects/new_project/api/
❌ Missing folder: projects/new_project/tests/
✅ All configuration folders present
✅ README.md exists and non-empty
⚠️  Empty folders need .gitkeep files

Summary: 2 errors, 1 warning
```

### **Auto-Fix Capability**
```bash
# Fix missing folders and .gitkeep files
python ci/scripts/validate_structure.py --fix

✅ Created: projects/new_project/api/.gitkeep
✅ Created: projects/new_project/tests/.gitkeep
✅ Structure validation passed
```

## 📊 Structure Benefits

### **Consistency**
- **Predictable Layout**: All developers know where to find components
- **Easier Onboarding**: New team members understand structure immediately  
- **Reduced Errors**: Standard locations prevent misplaced files

### **Automation**
- **CI/CD Integration**: Automated testing and deployment
- **Script Portability**: Scripts work across all projects
- **Configuration Management**: Hydra configs are interchangeable

### **Maintainability**
- **Clear Separation**: Data, models, configs, and code are organized
- **Scalability**: Structure supports project growth
- **Documentation**: Self-documenting through organization

## 🔗 Related Documentation

- **[Data Download Guide](DATA_DOWNLOAD_README.md)**: Data acquisition patterns
- **[Configuration Guide](CONFIGURATION_GUIDE.md)**: Hydra setup details
- **[Development Guide](DEVELOPMENT_GUIDE.md)**: Coding standards
- **[API Reference](API_REFERENCE.md)**: ML Portfolio library usage

## 🛠️ Troubleshooting

### **Common Issues**

#### **Validation Failures**
```bash
# Problem: Missing folders
# Solution: Run with --fix flag
python ci/scripts/validate_structure.py --fix
```

#### **Git Issues with Empty Folders**
```bash
# Problem: Git doesn't track empty folders
# Solution: Add .gitkeep files (done automatically by validator)
```

#### **Configuration Errors**
```bash
# Problem: Hydra can't find configs
# Solution: Check conf/ folder structure matches requirements
```

---

*This structure is enforced by CI/CD and must be followed for all projects.*  
*Last updated: October 2025 | Structure version: 2.0*