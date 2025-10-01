# 🏗️ Standardized Project Structure

This document defines the **mandatory project structure** for all forecasting projects in the ML Portfolio. The structure is enforced by automated validation and ensures consistency across all domains.

## 🎯 Design Principles

1. **🔧 Uniformity**: All projects follow identical folder structure
2. **📦 Predictable Paths**: Portfolio apps and pipelines can rely on consistent paths  
3. **👔 Professional Presentation**: Recruiters see uniform, organized projects
4. **🚀 Future-Ready**: New development fills content without reshaping structure
5. **🤖 CI/CD Enforced**: Automated validation prevents structural drift

## 📁 Mandatory Structure

Every project **MUST** have the following structure:

```
projects/{project_name}/
├── 📄 README.md                    # Project-specific documentation
├── 📁 api/                         # FastAPI endpoints (.gitkeep)
├── 📁 app/                         # Streamlit dashboard (.gitkeep)
├── 📁 conf/                        # Hydra configuration management
│   ├── config.yaml                 # Main configuration file
│   ├── dataset/                    # Dataset configurations (.gitkeep)
│   ├── model/                      # Model configurations (.gitkeep)
│   ├── optimizer/                  # Optimizer configurations (.gitkeep)
│   ├── scheduler/                  # LR scheduler configurations (.gitkeep)
│   └── hydra/                      # Hydra runtime settings (.gitkeep)
├── 📁 data/                        # Data management (DVC-style)
│   ├── external/                   # External/third-party data (.gitkeep)
│   ├── interim/                    # Intermediate processed data (.gitkeep)
│   ├── processed/                  # Final processed data (.gitkeep)
│   └── raw/                        # Raw unprocessed data (actual data files)
├── 📁 models/                      # Trained models and artifacts
│   ├── artifacts/                  # Model artifacts (scalers, encoders) (.gitkeep)
│   └── checkpoints/                # Model checkpoints during training (.gitkeep)
├── 📁 notebooks/                   # Jupyter notebooks for exploration (.gitkeep)
├── 📁 reports/                     # Generated analysis reports
│   └── figures/                    # Generated plots and visualizations (.gitkeep)
├── 📁 scripts/                     # Python scripts
│   └── {download_data|generate_data}.py  # Data acquisition script
└── 📁 tests/                       # Unit and integration tests (.gitkeep)
```

## 🔧 Standardized CLI Commands

All projects support **identical command patterns**:

### **Data Acquisition**
```bash
# From any project directory
python scripts/download_data.py    # For real datasets (Walmart, TSI)
python scripts/generate_data.py    # For synthetic datasets (Ola, Inventory)
```

### **Training (Future)**
```bash
# Hydra-based training with consistent patterns
python scripts/train.py model=lstm dataset={PROJECT} optimizer=adam
python scripts/train.py model=tcn dataset={PROJECT} optimizer=adamw
python scripts/train.py model=transformer dataset={PROJECT} optimizer=adam
```

### **Evaluation (Future)**
```bash
# Consistent evaluation commands
python scripts/evaluate.py --study-name {PROJECT}_optimization
python scripts/evaluate.py --model-path models/checkpoints/best_model.pt
```

## ⚙️ Configuration Standards

Every project has a standardized `conf/config.yaml`:

```yaml
# Main configuration for {project_name}
# This file defines default settings for all experiments

defaults:
  - dataset: {project_dataset_name}
  - model: lstm
  - optimizer: adam
  - scheduler: cosine
  - _self_

# Experiment settings
experiment:
  name: {project_name}_baseline
  tags: [baseline, {dataset_name}]

# Training settings  
trainer:
  max_epochs: 100
  patience: 10
  
# Evaluation settings
evaluation:
  primary_metric: {WMAE|MAPE|RMSE}
  cv_folds: 5
```

## 📊 Project-Specific Configurations

| Project | Dataset Name | Primary Metric | Data Script |
|---------|-------------|----------------|-------------|
| `retail_sales_walmart` | `walmart` | `WMAE` | `download_data.py` |
| `rideshare_demand_ola` | `ola` | `MAPE` | `generate_data.py` |
| `inventory_forecasting` | `inventory` | `MAPE` | `generate_data.py` |
| `transportation_tsi` | `tsi` | `RMSE` | `download_data.py` |

## 🔍 Validation & Enforcement

### **Local Validation**
```bash
# Validate all projects
python validate_structure.py

# Validate specific project
python validate_structure.py --project walmart

# Auto-fix missing structure
python validate_structure.py --fix

# CI mode (exit with error if issues)
python validate_structure.py --ci
```

### **CI/CD Enforcement**
- **GitHub Actions**: Automatic validation on every push/PR
- **Structure Check**: Validates folders and required files
- **Config Validation**: Ensures Hydra configs are properly formatted
- **CLI Consistency**: Verifies command patterns across projects

## 📦 Empty Folder Management

### **Decision Rule: Keep All Folders**
- **Method**: Use `.gitkeep` files in empty folders
- **Rationale**: 
  - Portfolio apps can rely on paths being present
  - Recruiters see complete, professional structure
  - Future development fills content without reshaping
  - No broken import paths or missing directory errors

### **Implementation**
- Empty folders contain `.gitkeep` with comment:
  ```
  # Placeholder to maintain folder structure
  ```
- Git tracks the structure even when folders are empty
- Development can proceed without structural changes

## 🎯 Benefits

### **For Developers**
- **Predictable Paths**: No guessing where files should go
- **Consistent Commands**: Same CLI patterns across all projects
- **Import Reliability**: Paths always exist for imports and file operations

### **For Portfolio Presentation**
- **Professional Appearance**: Uniform, organized project structure
- **Easy Navigation**: Recruiters can quickly understand any project
- **Scalability**: New projects follow the same proven pattern

### **For CI/CD & Automation**
- **Reliable Automation**: Scripts can depend on consistent paths
- **Quality Gates**: Structural validation prevents drift
- **Template-Based**: New projects start with complete structure

## 🔄 Adding New Projects

1. **Create Project Directory**: `projects/new_project_name/`
2. **Run Structure Validation**: `python validate_structure.py --project new_project_name --fix`
3. **Customize Configuration**: Edit `conf/config.yaml` with project-specific settings
4. **Add Data Script**: Create `scripts/download_data.py` or `scripts/generate_data.py`
5. **Update README**: Project-specific documentation
6. **Verify**: `python validate_structure.py --project new_project_name`

## 🚨 Structure Violations

The following will cause **CI failure**:
- Missing required folders
- Missing `conf/config.yaml`
- Missing data acquisition script
- Invalid YAML configuration
- Inconsistent CLI patterns

**Resolution**: Run `python validate_structure.py --fix` to automatically correct issues.

---

**💡 Remember**: This structure is **mandatory** and **enforced**. Any deviation will be caught by CI and must be corrected before merge.