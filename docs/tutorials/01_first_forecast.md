# Tutorial 1: Your First Forecast

Learn to train a forecasting model from scratch in 15 minutes.

## What You'll Build

A sales forecasting model for Walmart stores that predicts weekly sales based on:
- Historical sales data
- Store characteristics
- Temperature and fuel prices
- Holiday indicators

## Step 1: Understand the Data

```python
import pandas as pd

# Load Walmart data
data = pd.read_csv('projects/retail_sales_walmart/data/raw/Walmart.csv')

print(data.head())
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
```

**Output:**
```
       Store  Date  Weekly_Sales  Holiday_Flag  Temperature  Fuel_Price
0          1  ...      24924.50             0         42.31        2.572
...

Shape: (6435, 8)
Columns: ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday',
          'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
```

## Step 2: Feature Engineering

The pipeline automatically creates:
- **Lag features**: Previous week's sales (lag_1, lag_2, ...)
- **Rolling statistics**: Moving averages (rolling_mean_4, rolling_mean_8, ...)
- **Date features**: Month, day of week, quarter
- **Cyclical encoding**: Sin/cos encoding for periodic features

```python
from ml_portfolio.data.preprocessing import StaticTimeSeriesPreprocessingPipeline

pipeline = StaticTimeSeriesPreprocessingPipeline(
    date_column='Date',
    target_column='Weekly_Sales',
    group_columns=['Store'],
    lag_features=[1, 2, 4, 8, 13, 26, 52],  # Weekly lags
    rolling_windows=[4, 8, 13, 26],         # Moving averages
    date_features=True,
    cyclical_features=['month', 'dayofweek', 'quarter']
)

# Transform data
data_transformed = pipeline.fit_transform(data)
print(f"Original features: {len(data.columns)}")
print(f"Engineered features: {len(data_transformed.columns)}")
```

## Step 3: Split Data (Time Series Split)

**Critical:** Never shuffle time series data!

```python
from ml_portfolio.data.dataset_factory import DatasetFactory

factory = DatasetFactory(
    data_path="projects/retail_sales_walmart/data/raw/Walmart.csv",
    target_column='Weekly_Sales',
    timestamp_column='Date',
    train_ratio=0.7,   # 70% for training
    val_ratio=0.15,    # 15% for validation
    test_ratio=0.15,   # 15% for testing
    static_feature_engineer=pipeline
)

train, val, test = factory.get_datasets()

print(f"Train: {len(train)} samples")
print(f"Val: {len(val)} samples")
print(f"Test: {len(test)} samples")
```

**Timeline:**
```
|------------ Train (70%) ------------|--Val (15%)--|--Test (15%)--|
2010-01-01              2011-08-01    2012-03-01   2012-10-01
```

## Step 4: Train a Model

```python
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster

# Initialize model
model = LightGBMForecaster(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

# Train
print("Training model...")
model.fit(
    train.drop(columns=['Weekly_Sales', 'Date']),
    train['Weekly_Sales']
)
print("Training complete!")
```

## Step 5: Evaluate Performance

```python
from ml_portfolio.evaluation.metrics import MAPEMetric, RMSEMetric, MAEMetric

# Make predictions
val_pred = model.predict(val.drop(columns=['Weekly_Sales', 'Date']))

# Calculate metrics
mape = MAPEMetric()(val['Weekly_Sales'], val_pred)
rmse = RMSEMetric()(val['Weekly_Sales'], val_pred)
mae = MAEMetric()(val['Weekly_Sales'], val_pred)

print(f"Validation Metrics:")
print(f"  MAPE: {mape:.2f}%")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAE:  ${mae:,.2f}")
```

**Expected Output:**
```
Validation Metrics:
  MAPE: 32.19%
  RMSE: $86,414.68
  MAE:  $54,540.47
```

## Step 6: Visualize Predictions

```python
import matplotlib.pyplot as plt

# Get predictions for validation set
val_dates = val['Date']
val_actual = val['Weekly_Sales']
val_pred = model.predict(val.drop(columns=['Weekly_Sales', 'Date']))

# Plot
plt.figure(figsize=(15, 6))
plt.plot(val_dates, val_actual, label='Actual', alpha=0.7)
plt.plot(val_dates, val_pred, label='Predicted', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Weekly Sales ($)')
plt.title('Walmart Sales Forecast: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=300)
plt.show()
```

## Step 7: Analyze Feature Importance

```python
import pandas as pd

# Get feature importance
importance = model.get_feature_importance()

# Create DataFrame
feature_importance = pd.DataFrame({
    'feature': train.drop(columns=['Weekly_Sales', 'Date']).columns,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 8))
plt.barh(
    feature_importance.head(20)['feature'],
    feature_importance.head(20)['importance']
)
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

print("Top 10 Features:")
print(feature_importance.head(10))
```

**Typical Top Features:**
```
              feature  importance
0               lag_1      0.234
1               lag_2      0.156
2        rolling_mean_4    0.112
3               lag_4      0.089
4              Store_1    0.067
...
```

## Step 8: Test Set Evaluation

```python
# Final evaluation on test set (unseen data)
test_pred = model.predict(test.drop(columns=['Weekly_Sales', 'Date']))

test_mape = MAPEMetric()(test['Weekly_Sales'], test_pred)
test_rmse = RMSEMetric()(test['Weekly_Sales'], test_pred)
test_mae = MAEMetric()(test['Weekly_Sales'], test_pred)

print(f"Test Set Metrics:")
print(f"  MAPE: {test_mape:.2f}%")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE:  ${test_mae:,.2f}")
```

## Step 9: Save the Model

```python
import joblib

# Save model
joblib.dump(model, 'walmart_forecaster_v1.pkl')

# Save preprocessing pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

print("Model saved successfully!")
```

## Step 10: Make Predictions on New Data

```python
# Load model
model_loaded = joblib.load('walmart_forecaster_v1.pkl')
pipeline_loaded = joblib.load('preprocessing_pipeline.pkl')

# New data (example)
new_data = pd.DataFrame({
    'Store': [1],
    'Date': ['2025-10-06'],
    'Temperature': [75.5],
    'Fuel_Price': [3.2],
    'Holiday_Flag': [0]
})

# Transform features
new_data_transformed = pipeline_loaded.transform(new_data)

# Predict
prediction = model_loaded.predict(
    new_data_transformed.drop(columns=['Weekly_Sales', 'Date'])
)

print(f"Predicted Weekly Sales: ${prediction[0]:,.2f}")
```

## Common Mistakes to Avoid

### ❌ Mistake 1: Shuffling Time Series Data

```python
# WRONG! This causes data leakage
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, shuffle=True)  # DON'T DO THIS
```

**Why:** Future data leaks into training set.

**✅ Correct Way:**
```python
# Use temporal split
train = data[data['Date'] < '2012-01-01']
test = data[data['Date'] >= '2012-01-01']
```

### ❌ Mistake 2: Using Future Information in Features

```python
# WRONG! This uses future data
data['next_week_sales'] = data['Weekly_Sales'].shift(-1)  # DON'T DO THIS
```

**Why:** In production, you won't know next week's sales.

**✅ Correct Way:**
```python
# Use past data only
data['last_week_sales'] = data['Weekly_Sales'].shift(1)
```

### ❌ Mistake 3: Ignoring Seasonality

```python
# WRONG! Missing important patterns
model.fit(data[['Temperature', 'Fuel_Price']], target)  # Too few features
```

**✅ Correct Way:**
```python
# Include temporal features
data['month'] = data['Date'].dt.month
data['dayofweek'] = data['Date'].dt.dayofweek
data['is_holiday'] = data['Holiday_Flag']
```

## Next Steps

1. **Tutorial 2**: Hyperparameter Tuning with Optuna
2. **Tutorial 3**: Comparing Multiple Models
3. **Tutorial 4**: Probabilistic Forecasting
4. **Tutorial 5**: Production Deployment

## Complete Code

The full code is available at: `examples/tutorial_01_first_forecast.py`

```bash
# Run complete tutorial
python examples/tutorial_01_first_forecast.py
```

## Summary

You've learned to:
- ✅ Load and understand time series data
- ✅ Create lag and rolling features
- ✅ Split data temporally (no shuffling!)
- ✅ Train a LightGBM forecaster
- ✅ Evaluate with MAPE, RMSE, MAE
- ✅ Visualize predictions
- ✅ Analyze feature importance
- ✅ Save and load models
- ✅ Avoid common pitfalls

**Next:** [Tutorial 2 - Hyperparameter Optimization](02_hyperparameter_tuning.md)
