# Feature Engineering Configuration Guide

## Overview

The `feature_engineering` config group provides a unified way to configure both **static** and **statistical** preprocessing pipelines.

## Architecture

Each feature engineering config contains two components:

### 1. `static`: StaticTimeSeriesPreprocessingPipeline
- **When**: Applied BEFORE temporal splitting (in DatasetFactory)
- **Purpose**: Create deterministic, backward-looking features
- **Safe**: No data leakage (all features use past data only)
- **Features**:
  - Date features (year, month, day, dayofweek, weekend flags, etc.)
  - Lag features (t-1, t-7, t-14, etc.)
  - Rolling window statistics (mean, std with proper shifting)
  - Cyclical encoding (sine/cosine for periodic features)

### 2. `statistical`: StatisticalPreprocessingPipeline
- **When**: Applied AFTER temporal splitting (in train.py Phase 4)
- **Purpose**: Statistical transformations (scaling, normalization)
- **Follows Burkov's Principle**: Fit on train, transform all splits
- **Transformers**:
  - StandardScaler (mean=0, std=1)
  - RobustScaler (median-based, good for outliers)
  - MinMaxScaler (scale to [0, 1])
  - Any sklearn-compatible transformer

## Available Configs

### `full.yaml`
- **Static**: All features (lags 1,7,14,28 + rolling 7,14,30 + date + cyclical)
- **Statistical**: StandardScaler for features and target
- **Use case**: Complete feature engineering for most models

### `static_only.yaml`
- **Static**: All features
- **Statistical**: None
- **Use case**: Tree-based models that don't need scaling

### `statistical_only.yaml`
- **Static**: None
- **Statistical**: StandardScaler for features and target
- **Use case**: When using pre-extracted features

### `minimal.yaml`
- **Static**: Only date features + cyclical encoding (no lags/rolling)
- **Statistical**: StandardScaler for features only
- **Use case**: Lightweight feature set

### `robust.yaml`
- **Static**: Standard feature set (lags 1,7,14 + rolling 7,14)
- **Statistical**: RobustScaler
- **Use case**: Datasets with outliers

### `minmax.yaml`
- **Static**: Standard feature set
- **Statistical**: MinMaxScaler (scales to [0, 1])
- **Use case**: Neural networks (LSTM, Transformer)

### `advanced.yaml`
- **Static**: Full feature set
- **Statistical**: Multiple transformers (RobustScaler → StandardScaler)
- **Use case**: Advanced preprocessing pipelines

### `none.yaml`
- **Static**: None
- **Statistical**: None
- **Use case**: Raw features only (Random Forest, XGBoost)

## Usage Examples

### Basic Usage
```bash
# Use full feature engineering
python -m ml_portfolio.training.train feature_engineering=full

# Use only static features
python -m ml_portfolio.training.train feature_engineering=static_only

# Use only statistical preprocessing
python -m ml_portfolio.training.train feature_engineering=statistical_only

# No preprocessing
python -m ml_portfolio.training.train feature_engineering=none
```

### Override Specific Parameters
```bash
# Change lag features
python -m ml_portfolio.training.train \
  feature_engineering=full \
  feature_engineering.static.lag_features=[1,7]

# Change scaler parameters
python -m ml_portfolio.training.train \
  feature_engineering=full \
  feature_engineering.statistical.steps.0.1.with_mean=false

# Add group columns for grouped time series
python -m ml_portfolio.training.train \
  feature_engineering=full \
  feature_engineering.static.group_columns=[store_id,item_id]
```

### Model-Specific Recommendations
```bash
# ARIMA (statistical model, no preprocessing needed)
python -m ml_portfolio.training.train model=arima feature_engineering=none

# Random Forest (tree-based, only static features)
python -m ml_portfolio.training.train model=random_forest feature_engineering=static_only

# LSTM (neural network, needs scaling)
python -m ml_portfolio.training.train model=lstm feature_engineering=minmax

# Linear Model (needs both features and scaling)
python -m ml_portfolio.training.train model=linear feature_engineering=full
```

## Config Structure

### Example: `full.yaml`
```yaml
# Static feature engineering (before split)
static:
  _target_: ml_portfolio.data.preprocessing.StaticTimeSeriesPreprocessingPipeline
  date_column: date
  target_column: target
  lag_features: [1, 7, 14, 28]
  rolling_windows: [7, 14, 30]
  date_features: true
  cyclical_features: [month, dayofweek, quarter]

# Statistical preprocessing (after split, fit on train)
statistical:
  _target_: ml_portfolio.data.preprocessing.StatisticalPreprocessingPipeline
  steps:
    - - feature_scaler
      - _target_: sklearn.preprocessing.StandardScaler
        with_mean: true
        with_std: true
    - - target_scaler
      - _target_: sklearn.preprocessing.StandardScaler
        with_mean: true
        with_std: true
```

## Custom Transformers

You can add any sklearn-compatible transformer:

```yaml
statistical:
  _target_: ml_portfolio.data.preprocessing.StatisticalPreprocessingPipeline
  steps:
    # Power transformation
    - - power_transform
      - _target_: sklearn.preprocessing.PowerTransformer
        method: yeo-johnson

    # Standard scaling
    - - feature_scaler
      - _target_: sklearn.preprocessing.StandardScaler
        with_mean: true
        with_std: true

    # Quantile transformation for target
    - - target_quantile
      - _target_: sklearn.preprocessing.QuantileTransformer
        output_distribution: normal
```

## Naming Convention for Steps

- **`target_*`**: Applied to target variable (y)
  - Example: `target_scaler`, `target_quantile`
- **Other names**: Applied to features (X)
  - Example: `feature_scaler`, `robust_scaler`, `power_transform`

## Integration with DatasetFactory

The `static` pipeline should be integrated into `DatasetFactory.create_datasets()`:

```python
def create_datasets(self, feature_engineer=None):
    # Load data
    df = self.load_data()

    # Apply static feature engineering BEFORE splitting
    if feature_engineer:
        df = feature_engineer.engineer_features(df)

    # Split temporally
    train, val, test = self.temporal_split(df)

    return train, val, test
```

## Integration with train.py

The `statistical` pipeline is applied in train.py Phase 4:

```python
# Phase 4: Statistical preprocessing
if cfg.feature_engineering.statistical:
    pipeline = instantiate(cfg.feature_engineering.statistical)

    # Fit ONLY on training data
    pipeline.fit(train_dataset)

    # Transform all splits
    train_dataset = pipeline.transform(train_dataset)
    val_dataset = pipeline.transform(val_dataset)
    # test_dataset transformed later (Phase 9)
```

## Best Practices

1. **Static features**: Use for time series patterns (trends, seasonality)
2. **Statistical preprocessing**: Use for model compatibility (scaling for neural networks)
3. **Tree-based models**: Often work better with `static_only` (no scaling)
4. **Neural networks**: Always use scaling (`minmax` or `full`)
5. **Group columns**: Set when forecasting multiple time series (stores, items, etc.)
6. **Lag/rolling windows**: Adjust based on your forecast horizon and data frequency

## Troubleshooting

### Issue: "Column 'date' not found"
**Solution**: Update `date_column` to match your dataset
```bash
python train.py feature_engineering=full feature_engineering.static.date_column=timestamp
```

### Issue: "Target column not found"
**Solution**: Update `target_column` to match your dataset
```bash
python train.py feature_engineering=full feature_engineering.static.target_column=sales
```

### Issue: Too many features created
**Solution**: Reduce lag features and rolling windows
```bash
python train.py feature_engineering=minimal
# or customize
python train.py feature_engineering=full feature_engineering.static.lag_features=[1,7]
```

### Issue: Model not learning (neural network)
**Solution**: Ensure you're using scaling
```bash
python train.py model=lstm feature_engineering=minmax
```

---

**Created**: October 3, 2025
**Part of**: ML Portfolio Forecasting Demo
