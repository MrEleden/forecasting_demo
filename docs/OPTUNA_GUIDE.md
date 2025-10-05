# Optuna Integration - Quick Start Guide

## Overview
Optuna hyperparameter optimization is now integrated into the training pipeline. The system automatically:
- Returns validation MAPE for optimization
- Reports intermediate values for pruning
- Logs all trials to MLflow
- Handles failed trials gracefully
- Cleans up GPU memory between trials

## Installation

```bash
pip install optuna hydra-optuna-sweeper
```

## Usage

### Quick Test (10 trials)
```bash
python src/ml_portfolio/training/train.py \
    --config-name config_optuna_test \
    --multirun
```

### Full Optimization (100 trials)
```bash
python src/ml_portfolio/training/train.py \
    --config-name walmart_sweep \
    --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=100
```

### Custom Search Space
```bash
python src/ml_portfolio/training/train.py \
    --config-name walmart \
    --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.params.model="choice(lightgbm, xgboost)" \
    'hydra.sweeper.params.+model.n_estimators="range(50, 500, 50)"' \
    'hydra.sweeper.params.+model.max_depth="range(3, 10)"' \
    'hydra.sweeper.params.+model.learning_rate="interval(0.01, 0.3)"'
```

### Distributed Optimization (PostgreSQL backend)
```bash
# Terminal 1
python src/ml_portfolio/training/train.py \
    --config-name walmart \
    --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.storage="postgresql://user:pass@localhost/optuna" \
    hydra.sweeper.study_name="walmart_distributed"

# Terminal 2 (same database)
python src/ml_portfolio/training/train.py \
    --config-name walmart \
    --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.storage="postgresql://user:pass@localhost/optuna" \
    hydra.sweeper.study_name="walmart_distributed"
```

## What Gets Optimized

The system always uses **validation set metrics** for Optuna optimization:
- **Train set**: Used for model fitting
- **Validation set**: Used for hyperparameter selection (Optuna target)
- **Test set**: Never touched until final evaluation

This prevents overfitting to the test set and follows ML best practices.

### Configurable Primary Metric
You can choose which metric to optimize via `primary_metric` config:
- `primary_metric: MAPE` (default) - Mean Absolute Percentage Error
- `primary_metric: RMSE` - Root Mean Squared Error
- `primary_metric: MAE` - Mean Absolute Error
- `minimize: true/false` - Whether to minimize or maximize the metric

**Example**:
```yaml
# Optimize for RMSE instead of MAPE
primary_metric: RMSE
minimize: true
```

**Note**: The metric always comes from the **validation set**, ensuring unbiased hyperparameter selection.

## Features

### 1. Early Stopping (Pruning)
Bad trials are stopped early to save computation:
- Uses MedianPruner by default
- Reports val_MAPE after each epoch
- Stops trials that perform worse than median

### 2. MLflow Integration
All trials are logged to MLflow with:
- Trial number in run name
- `optuna_trial` tag
- All hyperparameters
- Train/val/test metrics
- `pruned` tag if stopped early

### 3. Error Handling
- Failed trials return `float('inf')`
- Optuna skips failed trials
- Study continues even if some trials crash

### 4. Memory Management
- GPU cache cleared after each trial
- Garbage collection between trials
- Prevents OOM in long sweeps

## Viewing Results

### MLflow UI
```bash
mlflow ui
# Navigate to http://localhost:5000
# Filter by experiment name to see all trials
```

### Optuna Dashboard (Optional)
```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
```

### Get Best Model via MLflow
```bash
python src/ml_portfolio/scripts/get_best_model_mlflow.py walmart_model_comparison
```

## Configuration

### Basic Optuna Config
```yaml
hydra:
  sweeper:
    direction: minimize  # Minimize val_MAPE
    n_trials: 100
    n_jobs: 1  # Parallel trials

    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42

    pruner:
      _target_: optuna.pruners.MedianPruner
      n_startup_trials: 5
```

### Search Space Syntax
- `choice(a, b, c)`: Categorical choice
- `range(low, high)`: Integer range
- `range(low, high, step)`: Integer with step
- `interval(low, high)`: Float interval
- `tag(log)`: Log scale for floats

## Troubleshooting

### Issue: "No module named 'optuna'"
```bash
pip install optuna hydra-optuna-sweeper
```

### Issue: Trials failing silently
Check MLflow UI for error tags or run without multirun to see errors:
```bash
python src/ml_portfolio/training/train.py model=lightgbm
```

### Issue: Out of memory
Reduce batch size or number of parallel jobs:
```bash
hydra.sweeper.n_jobs=1
```

### Issue: Optimization not improving
- Increase n_trials
- Widen search space
- Check if val_MAPE is being calculated correctly

## Example Workflow

1. **Quick test** (10 trials): Verify everything works
2. **Medium sweep** (50 trials): Find good region
3. **Fine-tuning** (100+ trials): Optimize within region
4. **Get best model**: Use MLflow script
5. **Final evaluation**: Test on holdout set

## Next Steps

- Add more models to search space
- Implement multi-objective optimization (MAPE + speed)
- Add conditional search spaces
- Create visualization scripts
- Setup distributed optimization with PostgreSQL
