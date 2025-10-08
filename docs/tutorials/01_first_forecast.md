---
title: Tutorial 1 – Your First Forecast
description: Step-by-step walkthrough for training and evaluating a LightGBM sales forecaster using the shared ML portfolio components.
---

# Tutorial 1: Your First Forecast

Learn how to build a weekly sales forecaster for Walmart stores by combining the reusable components in this repository. You will:

- Explore the raw dataset.
- Engineer deterministic time-series features.
- Split data without temporal leakage.
- Train and evaluate a `LightGBMForecaster`.
- Visualise predictions, inspect feature importance, and persist the artefacts.

> **Prerequisites**
>
> - Virtual environment activated (`.\.venv\Scripts\Activate` on Windows PowerShell, `source .venv/bin/activate` on macOS/Linux).
> - Dependencies installed from `requirements-ml.txt` and the package installed in editable mode (`pip install -e .`).
> - Walmart dataset downloaded via `python src/ml_portfolio/scripts/download_all_data.py --dataset walmart` (creates `projects/retail_sales_walmart/data/raw/Walmart.csv`).

______________________________________________________________________

## Step 1: Inspect the raw data

```python
import pandas as pd

raw_path = "projects/retail_sales_walmart/data/raw/Walmart.csv"
data = pd.read_csv(raw_path)

print(data.head())
print(f"Shape: {data.shape}")
print("Columns:", data.columns.tolist())
```

**Typical output**

```
   Store Dept        Date  Weekly_Sales  IsHoliday  Temperature  Fuel_Price   CPI  Unemployment
0      1    1  05-02-2010      24924.50          0        42.31       2.572  211.096358        8.106
1      1    1  12-02-2010      46039.49          1        38.51       2.548  211.242170        8.106
...

Shape: (6435, 8)
Columns: ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
```

______________________________________________________________________

## Step 2: Engineer static features

Use the built-in `StaticTimeSeriesPreprocessingPipeline` to add lagged sales, rolling statistics, and calendar flags *before* you split the data.

```python
from ml_portfolio.data.preprocessing import StaticTimeSeriesPreprocessingPipeline

pipeline = StaticTimeSeriesPreprocessingPipeline(
    date_column="Date",
    target_column="Weekly_Sales",
    group_columns=["Store", "Dept"],
    lag_features=[1, 2, 4, 8, 52],
    rolling_windows=[4, 8, 52],
    date_features=True,
    cyclical_features=["week"],
)

engineered = pipeline.engineer_features(data)
print("Original columns:", len(data.columns))
print("Engineered columns:", len(engineered.columns))
```

______________________________________________________________________

## Step 3: Split deterministically with `DatasetFactory`

`DatasetFactory.create_datasets()` returns `TimeSeriesDataset` objects for train/validation/test splits, ensuring the temporal order is preserved.

```python
from ml_portfolio.data.dataset_factory import DatasetFactory

factory = DatasetFactory(
    data_path=raw_path,
    target_column="Weekly_Sales",
    timestamp_column="Date",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    static_feature_engineer=pipeline,
)

train_ds, val_ds, test_ds = factory.create_datasets()

print(len(train_ds), "train samples")
print(len(val_ds), "validation samples")
print(len(test_ds), "test samples")
```

Convert each dataset into pandas structures for LightGBM. The helper below preserves feature names and timestamps when available.

```python
import pandas as pd

def dataset_to_frame(dataset, target_name="Weekly_Sales", timestamp_name="Date"):
    X, y = dataset.get_data()
    feature_names = dataset.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

    features = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name=target_name).reset_index(drop=True)
    df = pd.concat([features, target], axis=1)

    if dataset.timestamps is not None:
        timestamps = pd.to_datetime(pd.Series(dataset.timestamps, name=timestamp_name))
        df.insert(0, timestamp_name, timestamps.reset_index(drop=True))

    return df

train_df = dataset_to_frame(train_ds)
val_df = dataset_to_frame(val_ds)
test_df = dataset_to_frame(test_ds)

feature_cols = [col for col in train_df.columns if col not in {"Weekly_Sales", "Date"}]
```

> **Tip:** When you are ready for a final model, concat `train_df` and `val_df` before fitting to use more history.

______________________________________________________________________

## Step 4: Train a LightGBM forecaster

```python
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster

model = LightGBMForecaster(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

print("Training model...")
model.fit(train_df[feature_cols], train_df["Weekly_Sales"])
print("Training complete!")
```

______________________________________________________________________

## Step 5: Evaluate on the validation split

```python
from ml_portfolio.evaluation.metrics import MAPEMetric, RMSEMetric, MAEMetric

val_pred = model.predict(val_df[feature_cols])

mape = MAPEMetric()(val_df["Weekly_Sales"], val_pred)
rmse = RMSEMetric()(val_df["Weekly_Sales"], val_pred)
mae = MAEMetric()(val_df["Weekly_Sales"], val_pred)

print("Validation metrics:")
print(f"  MAPE: {mape:.2f}%")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAE:  ${mae:,.2f}")
```

Expect validation MAPE in the low 30s as a baseline; tuning and additional features can push it lower.

______________________________________________________________________

## Step 6: Visualise predictions vs. actuals

```python
import matplotlib.pyplot as plt

val_plot = val_df.copy()
val_plot["prediction"] = val_pred

plt.figure(figsize=(14, 6))
plt.plot(val_plot["Date"], val_plot["Weekly_Sales"], label="Actual", alpha=0.7)
plt.plot(val_plot["Date"], val_plot["prediction"], label="Predicted", alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Weekly Sales ($)")
plt.title("Walmart sales forecast – validation window")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_comparison.png", dpi=300)
plt.show()
```

______________________________________________________________________

## Step 7: Analyse feature importance

```python
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(20)["feature"], feature_importance.head(20)["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 20 engineered features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("Top 10 features:")
print(feature_importance.head(10))
```

Lag features (especially `Weekly_Sales_lag_1_*`) typically dominate, signalling weekly persistence in demand.

______________________________________________________________________

## Step 8: Evaluate on the test split

```python
test_pred = model.predict(test_df[feature_cols])

test_mape = MAPEMetric()(test_df["Weekly_Sales"], test_pred)
test_rmse = RMSEMetric()(test_df["Weekly_Sales"], test_pred)
test_mae = MAEMetric()(test_df["Weekly_Sales"], test_pred)

print("Test metrics:")
print(f"  MAPE: {test_mape:.2f}%")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE:  ${test_mae:,.2f}")
```

The test window confirms whether validation performance generalises to unseen weeks.

______________________________________________________________________

## Step 9: Persist the model and pipeline

```python
import joblib

joblib.dump(model, "models/walmart_forecaster_lightgbm.pkl")
joblib.dump(pipeline, "models/walmart_static_pipeline.pkl")

print("Model and pipeline saved to ./models")
```

______________________________________________________________________

## Step 10: Score new data

```python
model_loaded = joblib.load("models/walmart_forecaster_lightgbm.pkl")
pipeline_loaded = joblib.load("models/walmart_static_pipeline.pkl")

new_data = pd.DataFrame({
    "Store": [1],
    "Dept": [1],
    "Date": ["06-10-2025"],
    "Weekly_Sales": [0.0],  # placeholder (removed after feature engineering)
    "IsHoliday": [0],
    "Temperature": [75.5],
    "Fuel_Price": [3.2],
    "CPI": [212.1],
    "Unemployment": [7.8],
})

new_features = pipeline_loaded.engineer_features(new_data)
prediction = model_loaded.predict(
    new_features.drop(columns=["Weekly_Sales", "Date"], errors="ignore")
)

print(f"Predicted weekly sales: ${prediction[0]:,.2f}")
```

______________________________________________________________________

## Common pitfalls

- **Shuffling time-series data**: Use temporal splits (`DatasetFactory`) instead of `train_test_split(..., shuffle=True)`.
- **Peeking at the future**: Do not create features that rely on forward-looking values (e.g., `shift(-1)`). Stick to lags and rolling stats that shift before aggregation.
- **Ignoring seasonality**: Include date-derived features (month, week, holiday flags) for demand patterns.
- **Skipping editable installs**: Always run `pip install -e .` so `ml_portfolio` imports resolve inside notebooks and scripts.

______________________________________________________________________

## Next steps

- Optimise hyperparameters with Optuna (`docs/BENCHMARK.md` and `src/ml_portfolio/scripts/run_optimization.py`).
- Compare multiple models using the benchmark suite (`docs/BENCHMARK.md`).
- Surface the results in the Streamlit dashboard (`docs/DASHBOARD.md`).

______________________________________________________________________

## Summary

You have now:

- Engineered deterministic time-series features.
- Split data safely with `DatasetFactory`.
- Trained, evaluated, and visualised a LightGBM forecaster.
- Saved artefacts for reuse in downstream applications.
- Prepared a template for future experiments using the shared library.

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
1. **Tutorial 3**: Comparing Multiple Models
1. **Tutorial 4**: Probabilistic Forecasting
1. **Tutorial 5**: Production Deployment

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
