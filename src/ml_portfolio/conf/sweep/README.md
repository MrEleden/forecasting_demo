# Search Space Configurations Reference

## Overview

This directory contains Optuna hyperparameter search space definitions for all available models. These files are automatically loaded when `use_optuna=true` in the Walmart configuration.

## Available Search Spaces

### Gradient Boosting Models

#### 1. LightGBM (`lightgbm_search_space.yaml`)

**Best for**: Fast training, large datasets, interpretability

**Key Parameters**:

- `n_estimators`: 100-1000 (number of boosting iterations)
- `learning_rate`: 0.001-0.3 (log scale)
- `max_depth`: 3-12 (tree depth)
- `num_leaves`: 20-150 (LightGBM-specific)
- `reg_alpha`, `reg_lambda`: L1/L2 regularization

**Trials**: 50 (default)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true --multirun
```

#### 2. XGBoost (`xgboost_search_space.yaml`)

**Best for**: Robust performance, handling missing data, feature importance

**Key Parameters**:

- `n_estimators`: 100-1000
- `learning_rate`: 0.001-0.3 (log scale)
- `max_depth`: 3-12
- `min_child_weight`: 1-10
- `gamma`: 0-10 (minimum loss reduction)
- `alpha`, `lambda`: L1/L2 regularization

**Trials**: 50 (default)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=xgboost use_optuna=true --multirun
```

#### 3. CatBoost (`catboost_search_space.yaml`)

**Best for**: Categorical features, out-of-the-box performance

**Key Parameters**:

- `iterations`: 100-1000 (equivalent to n_estimators)
- `learning_rate`: 0.001-0.3 (log scale)
- `depth`: 4-10 (tree depth)
- `l2_leaf_reg`: 1-10 (L2 regularization)
- `border_count`: 32-255 (for categorical features)

**Trials**: 50 (default)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=catboost use_optuna=true --multirun
```

### Tree-Based Models

#### 4. Random Forest (`random_forest_search_space.yaml`)

**Best for**: Baseline, feature importance, reduced overfitting

**Key Parameters**:

- `n_estimators`: 50-500 (number of trees)
- `max_depth`: 5-30
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10
- `max_features`: sqrt, log2, 0.5, 0.7, 1.0
- `bootstrap`: true/false

**Trials**: 50 (default)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=random_forest use_optuna=true --multirun
```

### Statistical Models

#### 5. Prophet (`prophet_search_space.yaml`)

**Best for**: Strong seasonality, holidays, interpretable components

**Key Parameters**:

- `changepoint_prior_scale`: 0.001-0.5 (trend flexibility)
- `seasonality_prior_scale`: 0.01-10 (seasonality flexibility)
- `seasonality_mode`: additive/multiplicative
- `n_changepoints`: 10-50
- `weekly_seasonality`, `yearly_seasonality`: auto/true/false

**Trials**: 40 (fewer parameters)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=prophet use_optuna=true --multirun
```

#### 6. ARIMA (`arima_search_space.yaml`)

**Best for**: Univariate time series, trend and seasonality

**Key Parameters**:

- `order.0` (p): 0-5 (AR order)
- `order.1` (d): 0-2 (differencing order)
- `order.2` (q): 0-5 (MA order)
- `seasonal_order.0-3` (P, D, Q, m): seasonal components
- `trend`: null/c/t/ct (constant/trend)

**Trials**: 100 (large discrete space)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=arima use_optuna=true --multirun
```

### Deep Learning Models

#### 7. LSTM (`lstm_search_space.yaml`)

**Best for**: Long sequences, complex patterns, non-linear relationships

**Key Parameters**:

- `hidden_size`: 32-256 (LSTM units)
- `num_layers`: 1-4 (stacked LSTM layers)
- `dropout`: 0-0.5
- `learning_rate`: 0.0001-0.01 (log scale)
- `batch_size`: 16/32/64/128
- `seq_len`: 10-100 (sequence length)
- `bidirectional`: true/false

**Trials**: 60 (more complex)

**Usage**:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lstm use_optuna=true --multirun
```

## Multi-Model Optimization

### Compare All Models

```bash
# Optimize all 7 models (50+ trials each)
python src/ml_portfolio/training/train.py --config-name walmart \
    use_optuna=true \
    --multirun model=lightgbm,xgboost,catboost,random_forest,prophet,arima,lstm
```

### Compare Gradient Boosting Models

```bash
# Best performers for tabular data
python src/ml_portfolio/training/train.py --config-name walmart \
    use_optuna=true \
    --multirun model=lightgbm,xgboost,catboost
```

### Compare Statistical Models

```bash
# Traditional time series methods
python src/ml_portfolio/training/train.py --config-name walmart \
    use_optuna=true \
    --multirun model=prophet,arima
```

### Compare ML Models

```bash
# Machine learning approaches
python src/ml_portfolio/training/train.py --config-name walmart \
    use_optuna=true \
    --multirun model=lightgbm,xgboost,random_forest,lstm
```

## Customization

### Override Trial Count

```bash
# More thorough optimization
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true \
    hydra.sweeper.n_trials=100 --multirun
```

### Override Specific Parameter Range

```bash
# Narrow search space for faster results
python src/ml_portfolio/training/train.py --config-name walmart \
    model=xgboost use_optuna=true \
    'hydra.sweeper.params.+model.n_estimators.low=200' \
    'hydra.sweeper.params.+model.n_estimators.high=500' \
    --multirun
```

### Use Persistent Storage

```bash
# Save study for analysis and resumability
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true \
    hydra.sweeper.storage=sqlite:///walmart_lightgbm.db \
    --multirun
```

## Performance Comparison

| Model         | Speed  | Accuracy   | Memory   | Interpretability | Best Use Case                     |
| ------------- | ------ | ---------- | -------- | ---------------- | --------------------------------- |
| LightGBM      | ⚡⚡⚡ | ⭐⭐⭐⭐   | 💾💾     | 📊📊📊           | Large datasets, fast iteration    |
| XGBoost       | ⚡⚡   | ⭐⭐⭐⭐⭐ | 💾💾💾   | 📊📊📊           | Best accuracy, feature importance |
| CatBoost      | ⚡     | ⭐⭐⭐⭐   | 💾💾💾   | 📊📊📊           | Categorical features              |
| Random Forest | ⚡⚡   | ⭐⭐⭐     | 💾💾💾   | 📊📊📊📊         | Baseline, feature selection       |
| Prophet       | ⚡⚡⚡ | ⭐⭐⭐     | 💾       | 📊📊📊📊📊       | Strong seasonality, holidays      |
| ARIMA         | ⚡⚡   | ⭐⭐       | 💾       | 📊📊📊📊         | Univariate, traditional           |
| LSTM          | ⚡     | ⭐⭐⭐⭐   | 💾💾💾💾 | 📊               | Long sequences, non-linear        |

Legend:

- Speed: ⚡ (1-5, more is faster)
- Accuracy: ⭐ (1-5, more is better)
- Memory: 💾 (1-5, more means higher usage)
- Interpretability: 📊 (1-5, more is more interpretable)

## Tips for Optimization

### 1. Start Small

```bash
# Quick test with 10 trials
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true \
    hydra.sweeper.n_trials=10 --multirun
```

### 2. Use Pruning

The default MedianPruner stops bad trials early. Adjust for more aggressive pruning:

```bash
python src/ml_portfolio/training/train.py --config-name walmart \
    model=xgboost use_optuna=true \
    hydra.sweeper.pruner.n_startup_trials=3 \
    hydra.sweeper.pruner.n_warmup_steps=3 --multirun
```

### 3. Parallel Optimization (Use with Caution)

```bash
# Use all CPU cores (requires more RAM)
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true \
    hydra.sweeper.n_jobs=-1 --multirun
```

### 4. Resume from Storage

```bash
# First run
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true \
    hydra.sweeper.storage=sqlite:///optuna.db \
    hydra.sweeper.study_name=walmart_lgb \
    hydra.sweeper.n_trials=50 --multirun

# Resume with more trials
python src/ml_portfolio/training/train.py --config-name walmart \
    model=lightgbm use_optuna=true \
    hydra.sweeper.storage=sqlite:///optuna.db \
    hydra.sweeper.study_name=walmart_lgb \
    hydra.sweeper.n_trials=50 --multirun
```

## Visualization

After optimization, visualize results:

```bash
python src/ml_portfolio/scripts/visualize_optuna.py \
    --study-name walmart_lightgbm_optimization \
    --storage sqlite:///optuna.db \
    --output-dir results/optuna_viz
```

## References

- Search space files: `src/ml_portfolio/conf/sweep/`
- Model configs: `src/ml_portfolio/conf/model/`
- Main config: `src/ml_portfolio/conf/walmart.yaml`
- Usage guide: `docs/WALMART_CONFIG_GUIDE.md`
