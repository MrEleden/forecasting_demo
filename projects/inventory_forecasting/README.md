# Inventory Forecasting - Data Generation Guide

## Dataset Overview

- **Source**: Generated synthetic data (realistic patterns)
- **Size**: ~0.17 MB
- **Format**: CSV file (inventory_demand.csv)
- **Records**: 3,140 rows × 7 columns
- **Time Period**: 2021-2023 (weekly data)
- **Coverage**: 20 SKUs across 5 product categories

## Dataset Features

- `date`: Week ending date (weekly frequency)
- `sku_id`: Product identifier (SKU_001 to SKU_020)
- `category`: Product category (5 categories)
- `demand`: Target variable (weekly demand units)
- `price`: Product price ($)
- `promotion`: Promotion indicator (0/1)
- `week_of_year`: Week number (1-52)

## Product Categories

### Electronics (4 SKUs)

- SKU_001, SKU_002, SKU_003, SKU_004
- Higher price range ($200-800)
- Seasonal patterns (Q4 holiday boost)

### Clothing (4 SKUs)

- SKU_005, SKU_006, SKU_007, SKU_008
- Medium price range ($50-200)
- Seasonal fashion cycles

### Home & Garden (4 SKUs)

- SKU_009, SKU_010, SKU_011, SKU_012
- Medium price range ($100-400)
- Spring/summer seasonality

### Sports & Outdoors (4 SKUs)

- SKU_013, SKU_014, SKU_015, SKU_016
- Medium price range ($75-300)
- Summer activity peaks

### Books & Media (4 SKUs)

- SKU_017, SKU_018, SKU_019, SKU_020
- Lower price range ($10-50)
- Steady demand with back-to-school peaks

## Data Patterns

### Seasonal Trends

- **Q4 Holiday Boost**: 50% increase in electronics demand
- **Spring Surge**: Home & garden products peak
- **Summer Peak**: Sports & outdoors equipment
- **Back-to-school**: Books & media increase

### Weekly Patterns

- **Steady Demand**: Most categories follow weekly patterns
- **Promotional Impact**: 30% demand boost during promotions
- **Price Elasticity**: Demand inversely related to price

### Category-specific Patterns

- **Electronics**: High variance, holiday-driven
- **Clothing**: Fashion cycles, seasonal changes
- **Home & Garden**: Weather-dependent patterns
- **Sports**: Activity season alignment
- **Books**: Academic calendar influence

## Generation Commands

```bash
# Generate inventory demand data
python scripts/generate_data.py
```

## Expected Output

```
📦 Creating sample inventory forecasting data...
✅ Created: inventory_demand.csv (0.2 MB)
📊 3140 records, 5 categories
📅 Date range: 2021-01-01 00:00:00 to 2023-12-29 00:00:00
```

## Data Validation

```bash
# Check generated file
ls data/raw/inventory_demand.csv
# Should show: inventory_demand.csv (~170KB)

# Quick analysis
python -c "
import pandas as pd
df = pd.read_csv('data/raw/inventory_demand.csv')
print(f'Shape: {df.shape}')
print(f'SKUs: {df.sku_id.nunique()}')
print(f'Categories: {df.category.nunique()}')
print(f'Date range: {df.date.min()} to {df.date.max()}')
"
```

## Use Cases

- **Demand Forecasting**: Predict weekly product demand
- **Inventory Optimization**: Stock level planning
- **Promotion Planning**: Promotional impact analysis
- **Category Management**: Cross-category demand patterns

## Model Targets

- **SKU-level**: Individual product forecasting
- **Category-level**: Aggregate demand prediction
- **Price Sensitivity**: Demand elasticity modeling
- **Promotional Lift**: Promotion impact quantification

## Forecasting Challenges

- **Multiple Seasonalities**: Annual, quarterly, weekly patterns
- **Cross-SKU Effects**: Category cannibalization
- **External Factors**: Promotions, price changes
- **New Product Introduction**: Limited historical data

## Next Steps

1. **EDA**: Explore demand patterns by category and SKU
1. **Seasonality Analysis**: Decompose time series components
1. **Price Elasticity**: Analyze price-demand relationships
1. **Forecasting Models**: Hierarchical forecasting, Prophet, LSTM
