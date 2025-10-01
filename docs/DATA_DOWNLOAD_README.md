# Data Download Guide

## Overview
This guide explains how to download all datasets for the ML Portfolio Forecasting Projects.

## Quick Start

### Download All Datasets
```bash
python download_all_data.py
```

### Download Specific Dataset
```bash
python download_all_data.py --dataset walmart
python download_all_data.py --dataset ola
python download_all_data.py --dataset inventory
python download_all_data.py --dataset tsi
```

## Dataset Details

### 1. Walmart Retail Sales (`dataset=walmart`)
- **Source**: [yasserh/walmart-dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset) (Kaggle)
- **Files**: Walmart.csv (main sales dataset)
- **Requirements**: Kaggle API credentials
- **Setup**:
  ```bash
  pip install kaggle
  # Place kaggle.json in ~/.kaggle/ (see Kaggle documentation)
  python projects/retail_sales_walmart/scripts/download_data.py
  ```

### 2. Ola Bike Ride-Sharing (`dataset=ola`)
- **Source**: Synthetic data (realistic patterns)
- **Files**: ola_ride_requests.csv (3.3 MB)
- **Features**: 
  - 12 pickup zones across major Indian cities
  - Hourly ride request patterns
  - Weather data (temperature, humidity, rain)
  - Seasonal and daily patterns
- **Status**: ✅ Available (generated successfully)

### 3. Inventory Forecasting (`dataset=inventory`)
- **Source**: Synthetic data (realistic patterns)
- **Files**: inventory_demand.csv (0.2 MB)
- **Features**:
  - 5 product categories (Electronics, Clothing, Home, Sports, Books)
  - 20 SKUs total
  - Weekly demand patterns
  - Promotional effects
  - Seasonal variations
- **Status**: ✅ Available (generated successfully)

### 4. U.S. Transportation Services Index (`dataset=tsi`)
- **Source**: U.S. Bureau of Transportation Statistics Data Portal
- **URL**: https://data.bts.gov/Research-and-Statistics/Transportation-Services-Index-and-Seasonally-Adjus/bw6n-ddqk/about_data
- **Files**: TSI data (CSV format from BTS portal)
- **Features**:
  - Monthly economic indicators
  - Seasonally adjusted data
  - Freight and passenger indices
  - Economic cycle patterns
- **Status**: ✅ Official BTS data portal available

## Data Structure

Each project follows the DVC-style data layout:
```
projects/{project_name}/data/
├── raw/           # Original downloaded data
├── external/      # External reference data (calendars, etc.)
├── interim/       # Cleaned and merged data
└── processed/     # Final datasets for modeling
```

## Manual Download Instructions

### Walmart Data (if Kaggle API fails)
1. Visit: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
2. Accept competition rules
3. Download all CSV files
4. Place in: `projects/retail_sales_walmart/data/raw/`

### TSI Data (from official BTS portal)
1. Visit: https://data.bts.gov/Research-and-Statistics/Transportation-Services-Index-and-Seasonally-Adjus/bw6n-ddqk/about_data
2. Click "Export" and choose CSV format
3. Download the complete dataset
4. Place CSV files in: `projects/transportation_tsi/data/raw/`

## Data Validation

After downloading, validate your data:
```bash
# Check data files exist
ls projects/*/data/raw/

# Validate file contents
python -c "
import pandas as pd
from pathlib import Path

for project in ['retail_sales_walmart', 'rideshare_demand_ola', 'inventory_forecasting', 'transportation_tsi']:
    data_dir = Path(f'projects/{project}/data/raw')
    if data_dir.exists():
        files = list(data_dir.glob('*.csv'))
        print(f'{project}: {len(files)} CSV files')
        for f in files:
            df = pd.read_csv(f)
            print(f'  {f.name}: {len(df)} rows, {len(df.columns)} columns')
    else:
        print(f'{project}: No data directory')
"
```

## Troubleshooting

### Kaggle API Issues
- Install: `pip install kaggle`
- Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials
- Accept competition rules on Kaggle website

### Permission Errors
- Run as administrator on Windows
- Check directory permissions
- Ensure Python has write access to project folders

### Network Issues
- Check internet connection
- Try manual download for official sources
- Use VPN if accessing international sources

### Sample Data Quality
- Sample datasets are designed for development/testing
- Replace with real data for production models
- Check data patterns match your use case requirements

## Next Steps After Download

1. **Explore Data**: Run EDA notebooks in each project
   ```bash
   cd projects/retail_sales_walmart
   jupyter notebook notebooks/01_eda.ipynb
   ```

2. **Data Preprocessing**: Transform raw data to processed formats
   ```bash
   python scripts/build_matrix.py
   ```

3. **Model Training**: Start with baseline models
   ```bash
   python scripts/train_arima.py
   ```

## File Sizes (Approximate)
- Walmart: ~15-20 MB (actual Kaggle data)
- Ola: ~3.3 MB (synthetic data)
- Inventory: ~0.2 MB (synthetic data)
- TSI: ~0.1 MB (official or synthetic data)

Total: ~20-25 MB for all datasets