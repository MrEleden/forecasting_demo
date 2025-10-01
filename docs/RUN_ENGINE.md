# Running Training with engine.py

## Quick Start

Since you're in the root directory, you can now run training directly with `engine.py`:

```powershell
# Basic run with default configuration
python src/ml_portfolio/training/engine.py

# With parameter overrides
python src/ml_portfolio/training/engine.py dataset.batch_size=128 trainer.max_epochs=50

# Use different model
python src/ml_portfolio/training/engine.py model=xgboost

# Quick test with fewer epochs
python src/ml_portfolio/training/engine.py trainer.max_epochs=2 trainer.verbose=true
```

## What Happens

When you run `python src/ml_portfolio/training/engine.py`, the script will:

1. **Load Hydra Config** from `projects/retail_sales_walmart/conf/config.yaml`
2. **Create 3 Datasets** via Hydra instantiation:
   - `train_dataset = hydra.utils.instantiate(cfg.dataset, mode='train')`
   - `val_dataset = hydra.utils.instantiate(cfg.dataset, mode='val')`
   - `test_dataset = hydra.utils.instantiate(cfg.dataset, mode='test')`
3. **Load Data** by calling `dataset.load()` on each
4. **Instantiate Model** from config (e.g., RandomForestRegressor)
5. **Create TrainingEngine** with model, metrics, and config
6. **Train** by calling `engine.fit(train_dataset, val_dataset)`
   - Engine creates DataLoaders
   - Engine handles epoch loops
   - Engine computes validation metrics
   - Engine applies early stopping
7. **Evaluate** on test dataset
8. **Return** primary metric value

## Hydra Configuration

The script uses config from:
```
projects/retail_sales_walmart/conf/
├── config.yaml              # Main config (defaults, metrics, trainer)
├── dataset/
│   └── walmart.yaml         # Dataset config with _target_
└── model/
    └── random_forest.yaml   # Model config with _target_
```

## Example Configurations

### Dataset Config (`conf/dataset/walmart.yaml`)
```yaml
_target_: projects.retail_sales_walmart.data.walmart_dataset.WalmartDataset
data_path: projects/retail_sales_walmart/data/raw/Walmart.csv
target_column: Weekly_Sales
lookback_window: 52
forecast_horizon: 12
batch_size: 64
train_ratio: 0.7
validation_ratio: 0.15
test_ratio: 0.15
```

### Model Config (`conf/model/random_forest.yaml`)
```yaml
_target_: sklearn.ensemble.RandomForestRegressor
n_estimators: 100
max_depth: 20
random_state: 42
```

### Main Config (`conf/config.yaml`)
```yaml
defaults:
  - dataset: walmart
  - model: random_forest
  - _self_

seed: 42
primary_metric: mape

metrics:
  mape:
    _target_: sklearn.metrics.mean_absolute_percentage_error
  mae:
    _target_: sklearn.metrics.mean_absolute_error

trainer:
  max_epochs: 100
  patience: 10
  early_stopping: true
  verbose: true
```

## Common Commands

```powershell
# Run with default settings
python src/ml_portfolio/training/engine.py

# Override batch size
python src/ml_portfolio/training/engine.py dataset.batch_size=32

# Change model
python src/ml_portfolio/training/engine.py model=xgboost

# Adjust training parameters
python src/ml_portfolio/training/engine.py trainer.max_epochs=50 trainer.patience=5

# Multi-run sweep
python src/ml_portfolio/training/engine.py -m model=random_forest,xgboost dataset.batch_size=32,64

# Disable early stopping
python src/ml_portfolio/training/engine.py trainer.early_stopping=false

# Quick test (2 epochs)
python src/ml_portfolio/training/engine.py trainer.max_epochs=2
```

## Output

You'll see output like:
```
============================================================
Training with TrainingEngine
============================================================
Configuration:
...
Creating datasets: projects.retail_sales_walmart.data.walmart_dataset.WalmartDataset
Train dataset: 700 samples
Val dataset: 150 samples
Test dataset: 150 samples
...
TrainingEngine initialized in single-shot mode
Starting training...
Training completed in 2.34s
...
Test Results:
  test_mae: 12345.6789
  test_mape: 0.1234
  test_rmse: 15678.9012
============================================================
Training complete! Primary metric (mape): 0.1234
============================================================
```

## Hydra Output Directory

Hydra automatically creates output directories:
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml        # Full resolved config
        │   ├── hydra.yaml         # Hydra settings
        │   └── overrides.yaml     # Your CLI overrides
        └── main.log               # Training logs
```

## Troubleshooting

### If you get import errors:
```powershell
# Make sure you're in the root directory
cd c:\Users\mvill\github\forecasting_demo

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### If config not found:
The script looks for config at:
`projects/retail_sales_walmart/conf/config.yaml`

Make sure this file exists and has the correct structure.

### If dataset not found:
Check that `WalmartDataset` class exists at:
`projects/retail_sales_walmart/data/walmart_dataset.py`

## Summary

**Command to run from root:**
```powershell
python src/ml_portfolio/training/engine.py
```

The engine.py script now:
✅ Uses Hydra for configuration
✅ Loads datasets via Hydra instantiation
✅ Creates train/val/test datasets automatically
✅ Handles all training orchestration
✅ Computes validation metrics
✅ Applies early stopping
✅ Evaluates on test set

Everything is referenced from the Hydra config file! 🚀
