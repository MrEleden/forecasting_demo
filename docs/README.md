# 📚 Documentation Index

Welcome to the ML Portfolio Forecasting Demo documentation. This section contains comprehensive guides, setup instructions, and reference materials.

## 📋 Quick Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| **[Data Download Guide](DATA_DOWNLOAD_README.md)** | Dataset setup for all projects | Developers, Data Scientists |
| **[Kaggle Setup Guide](KAGGLE_SETUP.md)** | Walmart dataset API configuration | New users |
| **[Dataset Status Report](DATASET_STATUS.md)** | Current data availability | Project managers |
| **[Project Structure Guide](PROJECT_STRUCTURE.md)** | Standardized folder structure | Developers, Contributors |

## 🎯 Getting Started

### **New to the Project?**
1. Start with **[Dataset Status Report](DATASET_STATUS.md)** for overview
2. Follow **[Data Download Guide](DATA_DOWNLOAD_README.md)** for setup
3. Use **[Kaggle Setup Guide](KAGGLE_SETUP.md)** if needed for Walmart data

### **Developers & Contributors**
- **[Copilot Instructions](../.github/copilot-instructions.md)**: AI development guidelines
- **[Main README](../README.md)**: Architecture and technical overview
- **Project READMEs**: Domain-specific documentation

## 📊 Project Documentation

### **Domain-Specific Guides**
- **[🏪 Walmart Retail Sales](../projects/retail_sales_walmart/README.md)**
  - Kaggle dataset setup
  - Store hierarchy forecasting
  - Holiday effect modeling

- **[🚴 Ola Ride-sharing](../projects/rideshare_demand_ola/README.md)**
  - Synthetic data generation
  - Peak hour analysis
  - Weather impact modeling

- **[📦 Inventory Forecasting](../projects/inventory_forecasting/README.md)**
  - Multi-SKU demand patterns
  - Hierarchical forecasting
  - Price elasticity analysis

- **[🚛 Transportation TSI](../projects/transportation_tsi/README.md)**
  - Official government data
  - Economic indicator forecasting
  - Seasonal adjustment analysis

## 🔧 Technical Documentation

### **Architecture & Design**
- **[Copilot Instructions](../.github/copilot-instructions.md)**: Complete technical specification
  - Project structure and principles
  - Development commands and patterns
  - Model implementation standards
  - Configuration management

- **[Project Structure Guide](PROJECT_STRUCTURE.md)**: Mandatory folder organization
  - Standardized structure enforcement
  - CLI command consistency
  - CI/CD validation rules
  - Empty folder management

### **Data Management**
- **[Data Download Guide](DATA_DOWNLOAD_README.md)**: Comprehensive data setup
  - All dataset sources and requirements
  - Download automation scripts
  - Validation and troubleshooting

### **API & Credentials**
- **[Kaggle Setup Guide](KAGGLE_SETUP.md)**: External API configuration
  - Account creation and token setup
  - Troubleshooting common issues
  - Alternative download methods

## 📈 Development Workflow

### **1. Environment Setup**
```bash
# Install dependencies
poetry install

# Activate environment
poetry shell
```

### **2. Data Preparation**
```bash
# Download all datasets
python download_all_data.py

# Check status
cat docs/DATASET_STATUS.md
```

### **3. Project Development**
```bash
# Choose domain project
cd projects/{project_name}/

# Read project README
cat README.md

# Start development
jupyter notebook notebooks/01_eda.ipynb
```

## 🎯 Use Cases by Role

### **🔬 Data Scientists**
- **Start**: [Dataset Status](DATASET_STATUS.md) → [Data Download](DATA_DOWNLOAD_README.md)
- **Focus**: Project READMEs for domain-specific patterns
- **Tools**: Jupyter notebooks, model training scripts

### **💻 ML Engineers**
- **Start**: [Copilot Instructions](../.github/copilot-instructions.md)
- **Focus**: Architecture, Hydra configs, production patterns
- **Tools**: Docker, CI/CD, API deployment

### **📊 Business Analysts**
- **Start**: [Main README](../README.md) overview
- **Focus**: Domain applications and business value
- **Tools**: Streamlit dashboards, model results

### **🎓 Students & Researchers**
- **Start**: [Main README](../README.md) for comprehensive overview
- **Focus**: Multiple forecasting approaches and evaluation
- **Tools**: Notebooks, experiment tracking, model comparison

## 🔍 Quick Navigation

### **Need to...**
- **Set up data?** → [Data Download Guide](DATA_DOWNLOAD_README.md)
- **Fix Kaggle API?** → [Kaggle Setup Guide](KAGGLE_SETUP.md)
- **Check data status?** → [Dataset Status Report](DATASET_STATUS.md)
- **Understand architecture?** → [Copilot Instructions](../.github/copilot-instructions.md)
- **Learn about domain?** → Project-specific READMEs
- **Deploy models?** → [Main README](../README.md) Docker section

### **Troubleshooting**
- **Data download fails**: Check [Kaggle Setup](KAGGLE_SETUP.md) and [Data Download](DATA_DOWNLOAD_README.md)
- **Missing dependencies**: Follow [Main README](../README.md) setup
- **Configuration issues**: Review [Copilot Instructions](../.github/copilot-instructions.md)

---

💡 **Tip**: All documentation is interconnected. Start with your role-specific entry point above, then follow the cross-references as needed.