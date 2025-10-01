# Ola Ride-sharing Demand - Data Generation Guide

## Dataset Overview
- **Source**: Generated synthetic data (realistic patterns)
- **Size**: ~3.32 MB
- **Format**: CSV file (ola_ride_requests.csv)
- **Records**: 52,500 rows × 10 columns
- **Time Period**: 2022-2023 (hourly data)
- **Coverage**: 12 pickup zones across major Indian cities

## Dataset Features
- `datetime`: Timestamp (hourly frequency)
- `pickup_zone`: Location identifier (12 zones)
- `ride_requests`: Target variable (ride demand count)
- `temperature`: Weather temperature (°C)
- `humidity`: Weather humidity (%)
- `is_raining`: Rain indicator (0/1)
- `hour_of_day`: Hour (0-23)
- `day_of_week`: Weekday (0-6)
- `is_weekend`: Weekend indicator
- `month`: Month (1-12)

## Pickup Zones
### Mumbai (3 zones)
- Mumbai_Central
- Mumbai_Andheri
- Mumbai_Bandra

### Delhi (3 zones)
- Delhi_CP (Connaught Place)
- Delhi_Gurgaon
- Delhi_Noida

### Bangalore (3 zones)
- Bangalore_Koramangala
- Bangalore_Whitefield
- Bangalore_MG_Road

### Other Cities (3 zones)
- Chennai_T_Nagar
- Pune_Koregaon_Park
- Hyderabad_Hitech_City

## Data Patterns
### Daily Patterns
- **Morning Peak**: 8-10 AM (1.5x demand)
- **Evening Peak**: 6-8 PM (1.8x demand)
- **Night Hours**: Reduced demand

### Weekly Patterns
- **Weekdays**: Higher commuter demand
- **Weekends**: 30% reduced demand, different timing patterns

### Seasonal Patterns
- **Monsoon Effect**: 40% demand reduction during rain
- **Festival Boost**: 40% increase during major festivals (Diwali)
- **Temperature Impact**: Demand varies with extreme temperatures

### City-specific Patterns
- **Mumbai**: Highest base demand (25 rides/hour)
- **Delhi**: Medium demand (22 rides/hour)
- **Bangalore**: Tech hub patterns (20 rides/hour)
- **Others**: Lower base demand (15 rides/hour)

## Generation Commands

```bash
# Generate Ola ride-sharing data
python scripts/generate_data.py
```

## Expected Output
```
🚴 Creating sample Ola bike ride-sharing data...
✅ Created: ola_ride_requests.csv (3.3 MB)
📊 52500 records, 12 zones
📅 Date range: 2022-01-01 00:00:00 to 2023-12-31 00:00:00
```

## Data Validation
```bash
# Check generated file
ls data/raw/ola_ride_requests.csv
# Should show: ola_ride_requests.csv (~3.3MB)

# Quick analysis
python -c "
import pandas as pd
df = pd.read_csv('data/raw/ola_ride_requests.csv')
print(f'Shape: {df.shape}')
print(f'Zones: {df.pickup_zone.nunique()}')
print(f'Date range: {df.datetime.min()} to {df.datetime.max()}')
"
```

## Use Cases
- **Demand Forecasting**: Predict hourly ride requests
- **Resource Planning**: Optimize driver allocation
- **Dynamic Pricing**: Demand-based pricing models
- **Weather Impact**: Weather-aware demand prediction

## Model Targets
- **Short-term**: Next 24-48 hours demand
- **Medium-term**: Weekly demand patterns
- **Long-term**: Monthly/seasonal trends
- **Multi-zone**: Cross-location demand modeling

## Next Steps
1. **EDA**: Explore temporal patterns and seasonality
2. **Weather Analysis**: Rain impact on demand
3. **Peak Detection**: Rush hour pattern analysis
4. **Forecasting Models**: ARIMA, Prophet, LSTM for demand prediction