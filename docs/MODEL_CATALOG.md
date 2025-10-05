# Model Catalog - Implemented Models

## Overview
This document lists all implemented forecasting models in the ML portfolio, organized by type and performance tier.

## Implemented Models

### Gradient Boosting Models (Tier 1 - Competition Winners)

#### 1. LightGBM
**File**: `src/ml_portfolio/models/statistical/lightgbm.py`
**Config**: `src/ml_portfolio/conf/model/lightgbm.yaml`
**Status**: ✅ Implemented

**Best For**: Fast training, large datasets, feature-rich data
**Key Features**:
- Lightning-fast training speed
- Memory efficient
- Handles missing values automatically
- Leaf-wise tree growth

**Usage**:
```bash
python src/ml_portfolio/training/train.py --config-name walmart model=lightgbm
```

**Dependencies**: `pip install lightgbm`

---

#### 2. XGBoost
**File**: `src/ml_portfolio/models/statistical/xgboost.py`
**Config**: `src/ml_portfolio/conf/model/xgboost.yaml`
**Status**: ✅ Implemented

**Best For**: Robust predictions, structured data
**Key Features**:
- Strong regularization (L1/L2)
- Stable and reliable
- Excellent feature importance
- Level-wise tree growth

**Usage**:
```bash
python src/ml_portfolio/training/train.py --config-name walmart model=xgboost
```

**Dependencies**: `pip install xgboost`

---

#### 3. CatBoost
**File**: `src/ml_portfolio/models/statistical/catboost.py`
**Config**: `src/ml_portfolio/conf/model/catboost.yaml`
**Status**: ✅ Implemented

**Best For**: Categorical features, minimal tuning
**Key Features**:
- Native categorical feature support
- Ordered boosting algorithm
- Less hyperparameter tuning needed
- Robust to overfitting

**Usage**:
```bash
python src/ml_portfolio/training/train.py --config-name walmart model=catboost
```

**Dependencies**: `pip install catboost`

---

#### 4. Random Forest
**File**: Uses sklearn directly
**Config**: `src/ml_portfolio/conf/model/random_forest.yaml`
**Status**: ✅ Implemented

**Best For**: Baseline tree ensemble
**Key Features**:
- Simple and robust
- No hyperparameter tuning required
- Good feature importance
- Parallel training

**Usage**:
```bash
python src/ml_portfolio/training/train.py --config-name walmart model=random_forest
```

**Dependencies**: Built-in with sklearn

---

### Statistical Models (Tier 3)

#### 5. Prophet
**File**: `src/ml_portfolio/models/statistical/prophet.py`
**Config**: `src/ml_portfolio/conf/model/prophet.yaml`
**Status**: ✅ Implemented

**Best For**: Seasonality, holidays, interpretability
**Key Features**:
- Automatic seasonality detection
- Holiday effects modeling
- Missing data handling
- Uncertainty intervals
- Interpretable components

**Usage**:
```bash
python src/ml_portfolio/training/train.py --config-name walmart model=prophet
```

**Dependencies**: `pip install prophet`

---

#### 6. ARIMA
**File**: `src/ml_portfolio/models/statistical/arima.py` (needs implementation)
**Config**: `src/ml_portfolio/conf/model/arima.yaml`
**Status**: ⏳ Config exists, needs implementation

**Best For**: Stationary time series, baseline
**Key Features**:
- Classic statistical method
- Interpretable parameters
- Fast training
- Univariate forecasting

**Dependencies**: `pip install statsmodels`

---

### Deep Learning Models (Tier 2)

#### 7. LSTM
**File**: `src/ml_portfolio/models/pytorch/lstm.py`
**Config**: `src/ml_portfolio/conf/model/lstm.yaml`
**Status**: ✅ Implemented

**Best For**: Sequential patterns, long-term dependencies
**Key Features**:
- Handles sequences naturally
- Long-term memory
- Standard DL architecture
- GPU acceleration

**Usage**:
```bash
python src/ml_portfolio/training/train.py --config-name walmart model=lstm
```

**Dependencies**: `pip install torch`

---

## Model Comparison Matrix

| Model | Training Speed | Accuracy | GPU Required | Interpretability | Categorical Support |
|-------|---------------|----------|--------------|------------------|---------------------|
| **LightGBM** | ⚡⚡⚡ Very Fast | ⭐⭐⭐⭐ Excellent | ❌ No | ⭐⭐⭐ Good | ⭐⭐ Moderate |
| **XGBoost** | ⚡⚡ Fast | ⭐⭐⭐⭐ Excellent | ❌ No | ⭐⭐⭐ Good | ⭐⭐ Moderate |
| **CatBoost** | ⚡ Moderate | ⭐⭐⭐⭐ Excellent | ❌ No | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Random Forest** | ⚡⚡ Fast | ⭐⭐⭐ Good | ❌ No | ⭐⭐⭐⭐ Excellent | ⭐ Poor |
| **Prophet** | ⚡⚡ Fast | ⭐⭐⭐ Good | ❌ No | ⭐⭐⭐⭐⭐ Excellent | ❌ None |
| **ARIMA** | ⚡⚡⚡ Very Fast | ⭐⭐ Fair | ❌ No | ⭐⭐⭐⭐⭐ Excellent | ❌ None |
| **LSTM** | ⚡ Moderate | ⭐⭐⭐ Good | ✅ Recommended | ⭐ Poor | ⭐⭐ Moderate |

## Quick Start Commands

### Single Model Training
```bash
# Best overall performance (gradient boosting)
python src/ml_portfolio/training/train.py --config-name walmart model=lightgbm

# Interpretable with seasonality
python src/ml_portfolio/training/train.py --config-name walmart model=prophet

# Deep learning approach
python src/ml_portfolio/training/train.py --config-name walmart model=lstm
```

### Model Comparison
```bash
# Compare all gradient boosting models
python scripts/benchmark_models.py --config walmart --models lightgbm xgboost catboost random_forest

# Compare statistical vs ML
python scripts/benchmark_models.py --config walmart --models prophet arima lightgbm

# Full benchmark
python scripts/benchmark_models.py --config walmart --all-models
```

## Installation Guide

### Minimal Setup (Statistical + Tree Models)
```bash
pip install lightgbm xgboost catboost prophet statsmodels scikit-learn pandas numpy
```

### Full Setup (Including Deep Learning)
```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# All other packages
pip install lightgbm xgboost catboost prophet statsmodels scikit-learn pandas numpy
```

### Using requirements file
```bash
pip install -r requirements-ml.txt  # ML models only
pip install -r requirements.txt     # All dependencies
```

## Models to Implement (Future)

### Priority 1 (High Impact)
- [ ] **Ensemble Models**: Weighted averaging and stacking
- [ ] **N-BEATS**: Neural basis expansion for interpretable forecasting
- [ ] **Temporal Fusion Transformer (TFT)**: Attention-based multi-horizon forecasting

### Priority 2 (Specialized)
- [ ] **DeepAR**: Probabilistic forecasting with RNNs
- [ ] **TCN**: Temporal Convolutional Networks
- [ ] **Transformer**: Pure attention-based forecasting

### Priority 3 (Experimental)
- [ ] **Chronos**: Amazon's pretrained time series model (LLM-based)
- [ ] **TimeGPT**: Time series foundation model
- [ ] **AutoML**: Auto-sklearn or FLAML integration

## Performance Benchmarks (Walmart Dataset)

Based on initial testing (will be updated with benchmark results):

| Model | WMAE (Expected) | Training Time | Inference Time |
|-------|-----------------|---------------|----------------|
| LightGBM | 1800-2200 | 2-5 min | <1s |
| XGBoost | 1900-2300 | 5-10 min | <1s |
| CatBoost | 1850-2250 | 10-20 min | <1s |
| Random Forest | 2500-2900 | 3-7 min | <1s |
| Prophet | 2400-2800 | 2-5 min | <1s |
| ARIMA | 2800-3200 | <1 min | <1s |
| LSTM | 2300-2700 | 10-30 min | <1s |

**Note**: Actual performance depends on hyperparameter tuning and feature engineering.

## Contributing

To add a new model:

1. **Create model wrapper**: `src/ml_portfolio/models/{category}/{model_name}.py`
   - Inherit from `StatisticalForecaster` or `PyTorchForecaster`
   - Implement `fit()`, `predict()`, `score()` methods
   - Add `is_fitted` flag

2. **Create config**: `src/ml_portfolio/conf/model/{model_name}.yaml`
   - Add `@package _global_` at top
   - Specify defaults for dataloader and engine
   - Define model parameters under `model:` key

3. **Test integration**:
   ```bash
   python src/ml_portfolio/training/train.py --config-name walmart model={model_name}
   ```

4. **Update this catalog**: Add model to appropriate section

## References

- LightGBM: https://lightgbm.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- CatBoost: https://catboost.ai/
- Prophet: https://facebook.github.io/prophet/
- PyTorch: https://pytorch.org/

---

**Last Updated**: October 3, 2025
**Models Implemented**: 6/6 in Phase 1
**Models Planned**: 9 additional models
