# Kaggle API Setup Guide

## Quick Setup for Walmart Dataset

To download the Walmart dataset from `yasserh/walmart-dataset`, you need to set up Kaggle API credentials:

### 1. Create Kaggle Account
- Go to [kaggle.com](https://www.kaggle.com) and create an account (free)

### 2. Get API Token
1. Go to your Kaggle Account page: https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json` file

### 3. Place API Token
**Windows:**
```powershell
# Create directory
New-Item -Path "$env:USERPROFILE\.kaggle" -ItemType Directory -Force

# Move the downloaded kaggle.json file to:
# C:\Users\{your-username}\.kaggle\kaggle.json
```

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Test Setup
```bash
# This should work without errors
python projects/retail_sales_walmart/scripts/download_data.py
```

### 5. Expected Download
- **Dataset**: yasserh/walmart-dataset
- **URL**: https://www.kaggle.com/datasets/yasserh/walmart-dataset
- **Size**: ~1-5 MB
- **Files**: Walmart.csv (main sales dataset)

### Alternative: Manual Download
If API setup fails, you can manually download:
1. Go to: https://www.kaggle.com/datasets/yasserh/walmart-dataset
2. Click "Download" button
3. Extract to: `projects/retail_sales_walmart/data/raw/`

## Troubleshooting

### Error: "Could not find kaggle.json"
- Make sure `kaggle.json` is in the correct directory
- Windows: `C:\Users\{username}\.kaggle\kaggle.json`
- Linux/Mac: `~/.kaggle/kaggle.json`

### Error: "API credentials invalid"
- Re-download `kaggle.json` from your account page
- Make sure the file is not corrupted

### Error: "Dataset not found"
- The dataset URL is: `yasserh/walmart-dataset`
- Make sure you're using the correct dataset name

## Next Steps
Once Kaggle API is set up:
```bash
# Download all datasets
python download_all_data.py

# Or just Walmart
python download_all_data.py --dataset walmart
```