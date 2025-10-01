# Walmart Retail Sales - Data Download Guide

## Dataset Overview
- **Source**: [yasserh/walmart-dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset) (Kaggle)
- **Size**: ~0.35 MB
- **Format**: CSV file (Walmart.csv)
- **Records**: 6,435 rows × 8 columns
- **Time Period**: 2010-2012 (weekly data)
- **Stores**: 45 unique stores

## Dataset Features
- `Store`: Store identifier (1-45)
- `Date`: Week ending date
- `Weekly_Sales`: Target variable ($209K - $3.8M)
- `Holiday_Flag`: Holiday indicator (0/1)
- `Temperature`: Weekly temperature
- `Fuel_Price`: Fuel price per gallon
- `CPI`: Consumer Price Index
- `Unemployment`: Unemployment rate

## Setup Requirements

### Option 1: Kaggle API (Automated)
1. **Create Kaggle Account**: [kaggle.com](https://www.kaggle.com)
2. **Get API Token**: 
   - Go to Account → API → Create New Token
   - Download `kaggle.json` file
3. **Place API Token**:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
4. **Install Kaggle API**: `pip install kaggle`

### Option 2: Manual Download
1. Go to: https://www.kaggle.com/datasets/yasserh/walmart-dataset
2. Click "Download" button
3. Extract `Walmart.csv` to: `data/raw/`

## Download Commands

```bash
# Setup Kaggle API (one-time)
pip install kaggle
# Place kaggle.json in ~/.kaggle/ (see docs/KAGGLE_SETUP.md)

# Download Walmart data
python scripts/download_data.py
```

## Expected Output
```
📊 Dataset Shape: (6435, 8)
📁 File Size: 0.35 MB
🗂️ Columns: ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
✅ No missing values found!
🎄 Holiday Analysis:
  Regular weeks: 5985
  Holiday weeks: 450
```

## Troubleshooting

### Error: "Could not find kaggle.json"
- Ensure API token is in correct directory
- Check file permissions (Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`)

### Error: "Dataset not found" 
- Verify dataset name: `yasserh/walmart-dataset`
- Check internet connection
- Try manual download option

## Next Steps
1. **EDA**: Run `notebooks/01_eda.ipynb`
2. **Baseline Models**: ARIMA, Prophet forecasting
3. **Deep Learning**: LSTM, TCN, Transformer models
4. **Hyperparameter Tuning**: Optuna optimization