# 📊 Data Download Guide

Comprehensive guide for setting up datasets across all forecasting projects in the ML Portfolio.

## 🎯 Quick Start

```bash
# Download all datasets at once
python download_all_data.py

# Or download individual project data
python projects/retail_sales_walmart/scripts/download_data.py
python projects/transportation_tsi/scripts/download_data.py
```

## 📋 Dataset Overview

| Project | Dataset | Source | Size | Type |
|---------|---------|--------|------|------|
| **Walmart Retail Sales** | Store sales data | Kaggle API | ~450MB | Real historical |
| **Ola Ride-sharing** | Demand patterns | Synthetic | ~50MB | Generated |
| **Inventory Forecasting** | Multi-SKU data | Synthetic | ~100MB | Generated |
| **Transportation TSI** | Economic indicators | BTS API | ~25MB | Government data |

## 🏪 Walmart Retail Sales Data

### **Prerequisites**
- Kaggle account and API key setup
- See **[Kaggle Setup Guide](KAGGLE_SETUP.md)** for detailed instructions

### **Download Process**
```bash
# Method 1: Automatic download
cd projects/retail_sales_walmart
python scripts/download_data.py

# Method 2: Manual Kaggle CLI
kaggle competitions download -c walmart-recruiting-store-sales-forecasting
unzip walmart-recruiting-store-sales-forecasting.zip -d data/raw/
```

### **Data Structure**
```
data/raw/
├── train.csv          # Historical sales data (421,570 rows)
├── test.csv           # Test period data (115,064 rows)  
├── stores.csv         # Store metadata (45 stores)
├── features.csv       # External features (8,190 rows)
└── sampleSubmission.csv
```

### **Data Schema**
- **train.csv**: Store, Dept, Date, Weekly_Sales, IsHoliday
- **stores.csv**: Store, Type, Size  
- **features.csv**: Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday

## 🚴 Ola Ride-sharing Data

### **Generation Process**
```bash
cd projects/rideshare_demand_ola
python scripts/generate_data.py
```

### **Generated Files**
```
data/raw/
├── ride_requests.csv      # Hourly demand by pickup zone
├── weather_data.csv       # Synthetic weather patterns
└── events_calendar.csv    # Special events affecting demand
```

### **Synthetic Data Features**
- **Time Series**: 2 years of hourly data (17,520 observations)
- **Spatial**: 50 pickup zones across city
- **Seasonality**: Weekly, daily, and holiday patterns
- **Weather**: Temperature, precipitation, visibility effects
- **Events**: Festivals, sports, concerts impact

## 📦 Inventory Forecasting Data

### **Generation Process**
```bash
cd projects/inventory_forecasting  
python scripts/generate_data.py
```

### **Generated Files**
```
data/raw/
├── product_hierarchy.csv  # SKU, category, brand mapping
├── sales_history.csv      # Historical sales by SKU
├── price_data.csv         # Pricing history and promotions
└── external_factors.csv   # Economic and seasonal factors
```

### **Multi-Level Hierarchy**
- **Categories**: 10 product categories
- **Brands**: 25 brands across categories
- **SKUs**: 500 individual products
- **Locations**: 20 distribution centers
- **Time Span**: 3 years of daily data

## 🚛 Transportation TSI Data

### **Prerequisites**
- Internet connection for BTS API access
- No authentication required (public data)

### **Download Process**
```bash
cd projects/transportation_tsi
python scripts/download_data.py
```

### **Data Source**
- **Bureau of Transportation Statistics (BTS)**
- **Transportation Services Index (TSI)**
- **URL**: https://www.bts.gov/tsi
- **Update Frequency**: Monthly
- **Historical Range**: 2000-present

### **Data Components**
```
data/raw/
├── tsi_total.csv         # Overall transportation index
├── tsi_freight.csv       # Freight transportation
├── tsi_passenger.csv     # Passenger transportation
└── tsi_pipeline.csv      # Pipeline transportation
```

## 🔄 Automated Data Pipeline

### **Master Download Script**
```bash
# Download all project data
python download_all_data.py

# Options available:
python download_all_data.py --projects walmart,tsi  # Specific projects
python download_all_data.py --skip-synthetic       # Skip generated data
python download_all_data.py --force-refresh        # Re-download existing
```

### **Validation Checks**
- ✅ File existence and sizes
- ✅ Data schema validation  
- ✅ Date range completeness
- ✅ Missing value analysis
- ✅ Outlier detection

## 📁 Data Directory Structure

Each project follows the **DVC-style data layout**:

```
projects/<project_name>/data/
├── external/          # External reference data
├── interim/           # Intermediate processed data  
├── processed/         # Final analysis-ready data
└── raw/              # Original, immutable data
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Kaggle API Problems**
```bash
# Error: "Kaggle API key not found"
Solution: Follow Kaggle Setup Guide for API configuration

# Error: "Competition data not accessible"  
Solution: Accept competition rules on Kaggle website first
```

#### **Synthetic Data Generation**
```bash
# Error: "Memory error during generation"
Solution: Reduce data size in config files

# Error: "Random seed inconsistency"
Solution: Set numpy random seed before generation
```

#### **Network Issues**
```bash
# Error: "Connection timeout"
Solution: Check internet connection, retry with --timeout 300

# Error: "SSL certificate verify failed"
Solution: Update certificates or use --no-ssl-verify flag
```

### **Data Quality Checks**

```bash
# Validate downloaded data
python -c "
from ml_portfolio.utils.io import validate_data_structure
validate_data_structure('projects/retail_sales_walmart/data')
"

# Check data completeness
python scripts/data_quality_check.py --project all
```

## 📊 Data Size and Storage

### **Disk Space Requirements**
- **Minimum**: 1GB free space
- **Recommended**: 2GB for all projects + processing
- **Cloud Storage**: Consider for large-scale experiments

### **Data Refresh Schedule**
- **Walmart**: Static competition data (no updates)
- **TSI**: Monthly updates from BTS
- **Synthetic**: Regenerate as needed for experiments
- **External**: Quarterly refresh recommended

## 🔗 Related Documentation

- **[Kaggle Setup Guide](KAGGLE_SETUP.md)**: API authentication
- **[Dataset Status Report](DATASET_STATUS.md)**: Current availability
- **[Project Structure Guide](PROJECT_STRUCTURE.md)**: Folder organization
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Common fixes

## 📞 Support

- **Data Issues**: Check DATASET_STATUS.md first
- **API Problems**: Review Kaggle/BTS documentation
- **Generation Failures**: Verify Python environment and dependencies
- **Storage Issues**: Monitor disk space and clean interim files

---

*Last updated: October 2025 | Next review: January 2026*