# Transportation Services Index (TSI) - Data Download Guide

## Dataset Overview

- **Source**: [U.S. Bureau of Transportation Statistics (BTS)](https://data.bts.gov/Research-and-Statistics/Transportation-Services-Index-and-Seasonally-Adjus/bw6n-ddqk/about_data)
- **Size**: ~0.19 MB
- **Format**: CSV file (tsi_official.csv)
- **Records**: 307 rows × 66 columns
- **Time Period**: 2000-2025 (monthly data)
- **Frequency**: Monthly economic indicators

## Dataset Description

The Transportation Services Index (TSI) is a seasonally adjusted economic indicator that tracks freight and passenger traffic in the United States. It includes data across multiple transportation modes:

### Key Indicators

- **TSI Total**: Overall transportation services index
- **TSI Freight**: Freight transportation component
- **TSI Passenger**: Passenger transportation component
- **Air Transportation**: RPM, ASM, load factors
- **Rail Transportation**: Freight carloads, intermodal, passenger miles
- **Truck Transportation**: Vehicle miles traveled
- **Transit**: Public transportation metrics
- **Pipeline**: Petroleum and natural gas transport
- **Waterborne**: Freight transportation

### Economic Context Features

- **Industrial Production**: Manufacturing index
- **Inventory-to-Sales Ratio**: Economic indicator
- **Unemployment**: Economic context data

## Data Files

- **tsi_official.csv**: Official BTS data (307 records, 66 columns)
- **tsi_sample.csv**: Simplified sample data (180 records, 7 columns)

## Download Commands

```bash
# Download TSI data
python scripts/download_data.py
```

## Expected Output

```
📊 307 records with 66 columns
📅 Date range: 2000-01 to 2025-07
🔍 Key columns: tsi_total, tsi_freight, tsi_passenger, obs_date
✅ Official BTS data successfully downloaded
```

## Data Quality

- ✅ Official government data source
- ✅ Regular monthly updates
- ✅ Seasonally adjusted indicators
- ✅ Multiple transportation modes covered

## Use Cases

- **Economic Forecasting**: Predict transportation demand
- **Business Intelligence**: Transportation sector analysis
- **Policy Analysis**: Infrastructure investment planning
- **Academic Research**: Transportation economics studies

## Troubleshooting

### Network Issues

- Government APIs can be slow: increase timeout
- Try manual download if automated fails
- Use sample data for development/testing

### Data Validation

```bash
# Check downloaded files
ls data/raw/
# Should show: tsi_official.csv, tsi_sample.csv
```

## Next Steps

1. **EDA**: Explore seasonal patterns and trends
1. **Economic Analysis**: Correlate with business cycles
1. **Time Series Models**: ARIMA, Prophet, LSTM forecasting
1. **Multi-variate Analysis**: Freight vs passenger patterns
