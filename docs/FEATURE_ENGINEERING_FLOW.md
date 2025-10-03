# Feature Engineering Flow - Implementation Complete

## Architecture Overview

The feature engineering system is now fully integrated with a clean separation between **static** (pre-split) and **statistical** (post-split) preprocessing.

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: DATA LOADING + STATIC FEATURE ENGINEERING                  │
│ (DatasetFactory.create_datasets())                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Load raw data (CSV/Parquet)                                      │
│ 2. Apply StaticTimeSeriesPreprocessingPipeline (BEFORE split)       │
│    - Date features (year, month, dayofweek, etc.)                   │
│    - Lag features (t-1, t-7, t-14, etc.)                            │
│    - Rolling window statistics (mean, std with shifting)            │
│    - Cyclical encoding (sin/cos for periodic features)              │
│ 3. Temporal split (train/val/test)                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: STATISTICAL PREPROCESSING                                   │
│ (train.py applies StatisticalPreprocessingPipeline)                 │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Fit on TRAIN ONLY (Burkov's principle)                           │
│    pipeline.fit(train_dataset)                                       │
│ 2. Transform train and validation                                    │
│    - StandardScaler, RobustScaler, MinMaxScaler, etc.               │
│ 3. Test set REMAINS UNTOUCHED                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 4-7: TRAINING                                                  │
│ (Create loaders, train model, validate)                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 8: TEST EVALUATION                                             │
│ (NOW we transform test set with fitted pipeline)                    │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Transform test set with fitted statistical pipeline              │
│ 2. Evaluate final performance                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration Structure

### Unified Config: `feature_engineering`

Each feature_engineering config contains two pipelines:

```yaml
# Static preprocessing (BEFORE split)
static:
  _target_: ml_portfolio.data.preprocessing.StaticTimeSeriesPreprocessingPipeline
  date_column: date
  target_column: target
  lag_features: [1, 7, 14, 28]
  rolling_windows: [7, 14, 30]
  date_features: true
  cyclical_features: [month, dayofweek, quarter]

# Statistical preprocessing (AFTER split)
statistical:
  _target_: ml_portfolio.data.preprocessing.StatisticalPreprocessingPipeline
  steps:
    - - feature_scaler
      - _target_: sklearn.preprocessing.StandardScaler
        with_mean: true
        with_std: true
```

## Integration Points

### 1. DatasetFactory Integration

**File**: `src/ml_portfolio/data/dataset_factory.py`

**Changes**:
- Added `static_feature_engineer` parameter to `__init__()`
- Apply static features BEFORE splitting in `create_datasets()`
- Features are deterministic and backward-looking (safe, no leakage)

```python
def create_datasets(self):
    df = self._load_raw_data()

    # Apply static features BEFORE split (safe)
    if self.static_feature_engineer is not None:
        df = self.static_feature_engineer.engineer_features(df)

    # Split temporally
    train, val, test = self._temporal_split(df)
    return train, val, test
```

### 2. Config Integration

**File**: `src/ml_portfolio/conf/dataset_factory/default.yaml`

**Changes**:
- Added `static_feature_engineer: ${feature_engineering.static}`
- This passes the static pipeline from feature_engineering config to factory

```yaml
_target_: ml_portfolio.data.dataset_factory.DatasetFactory
# ... other params ...
static_feature_engineer: ${feature_engineering.static}
```

### 3. Training Pipeline Integration

**File**: `src/ml_portfolio/training/train.py`

**Changes**:
- **Phase 2**: Static features applied automatically via DatasetFactory
- **Phase 3**: Statistical preprocessing uses `cfg.feature_engineering.statistical`
- **Phase 8**: Test set transformed with fitted statistical pipeline

```python
# Phase 3: Statistical preprocessing
if cfg.feature_engineering and cfg.feature_engineering.get("statistical"):
    statistical_pipeline = instantiate(cfg.feature_engineering.statistical)
    statistical_pipeline.fit(train_dataset)  # Fit on train only
    train_dataset = statistical_pipeline.transform(train_dataset)
    val_dataset = statistical_pipeline.transform(val_dataset)
    # Test NOT transformed yet

# Phase 8: Test evaluation
if statistical_pipeline is not None:
    test_dataset = statistical_pipeline.transform(test_dataset)
```

## Usage Examples

### Full Feature Engineering
```bash
python -m ml_portfolio.training.train feature_engineering=full
```
- Static: All features (lags, rolling, date, cyclical)
- Statistical: StandardScaler for features and target

### Static Only (Tree-Based Models)
```bash
python -m ml_portfolio.training.train \
  model=random_forest \
  feature_engineering=static_only
```
- Static: All features
- Statistical: None (tree models don't need scaling)

### Minimal + MinMax (Neural Networks)
```bash
python -m ml_portfolio.training.train \
  model=lstm \
  feature_engineering=minmax
```
- Static: Date + cyclical features only
- Statistical: MinMaxScaler (good for neural nets)

### No Preprocessing (Raw Features)
```bash
python -m ml_portfolio.training.train feature_engineering=none
```
- Static: None
- Statistical: None

### Custom Overrides
```bash
# Change lag features
python -m ml_portfolio.training.train \
  feature_engineering=full \
  feature_engineering.static.lag_features=[1,7]

# Add group columns for grouped forecasting
python -m ml_portfolio.training.train \
  feature_engineering=full \
  feature_engineering.static.group_columns=[store_id,item_id]

# Change scaler
python -m ml_portfolio.training.train \
  feature_engineering=statistical_only \
  feature_engineering.statistical.steps.0.1._target_=sklearn.preprocessing.RobustScaler
```

## Key Principles

### ✅ What's Safe (Static Features)
- Date features (deterministic)
- Lag features (backward-looking only)
- Rolling windows (with proper shifting to avoid leakage)
- Cyclical encoding (deterministic)

Applied **BEFORE** splitting - no data leakage!

### ⚠️ What's Not Safe (Statistical Preprocessing)
- StandardScaler (needs mean/std from data)
- RobustScaler (needs median/IQR from data)
- MinMaxScaler (needs min/max from data)
- Any transformation that uses statistics from data

Applied **AFTER** splitting - **fit on train only**!

## Testing the Integration

```bash
# Test with full feature engineering
python -m ml_portfolio.training.train \
  feature_engineering=full \
  dataset_factory=default \
  model=random_forest

# Should see in logs:
# Phase 2: Loading data and applying static feature engineering...
# Phase 3: Applying statistical preprocessing...
# Phase 8: Applied statistical preprocessing to test set
```

## Benefits

1. **Clean Separation**: Static vs statistical preprocessing clearly separated
2. **No Data Leakage**: Static features applied before split, statistical after
3. **Unified Config**: Single `feature_engineering` config for both pipelines
4. **Flexible**: Can use either, both, or neither
5. **Model-Specific**: Easy to choose appropriate preprocessing per model type
6. **Burkov's Principles**: Fits on train, transforms all splits correctly

## Next Steps

1. ✅ **COMPLETE**: DatasetFactory applies static features
2. ✅ **COMPLETE**: train.py uses unified config
3. ✅ **COMPLETE**: Config interpolation works (`${feature_engineering.static}`)
4. ⏳ **TODO**: Test end-to-end with real data
5. ⏳ **TODO**: Implement concrete model classes
6. ⏳ **TODO**: Create metrics.py functions

---

**Status**: Feature engineering flow fully implemented and integrated ✅
**Date**: October 3, 2025
