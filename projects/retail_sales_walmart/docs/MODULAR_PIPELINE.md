# Walmart Forecasting - Modular Training Pipeline

## Overview
This document describes the complete modular training pipeline for the Walmart sales forecasting project. The pipeline follows a **framework-agnostic** design where all components use standard interfaces and are instantiated via Hydra configuration.

## Architecture

### Key Principles
1. **No Type Detection**: No hardcoded model/dataset type checking
2. **Standard Interfaces**: All objects implement `.fit()`, `.predict()`, `.load()`, `.get_splits()`
3. **Hydra Instantiation**: All components created via `hydra.utils.instantiate()` with `_target_`
4. **PyTorch Compatible**: Dataset classes work with PyTorch DataLoader via `__len__`/`__getitem__`
5. **Inheritance Pattern**: Project-specific classes inherit from shared library base classes

## Component Structure

```
Shared Library (src/ml_portfolio/)
├── training/engine.py           → TrainingEngine (unified training loop)
├── data/datasets.py             → TimeSeriesDataset (base class)
├── data/loaders.py              → get_train_val_test_splits() utility
└── models/metrics.py            → Evaluation metrics

Project-Specific (projects/retail_sales_walmart/)
├── data/walmart_dataset.py      → WalmartDataset (extends TimeSeriesDataset)
├── conf/
│   ├── config.yaml              → Main Hydra config (defaults + trainer settings)
│   ├── dataset/walmart.yaml     → Dataset config (_target_ = WalmartDataset)
│   └── model/random_forest.yaml → Model config (_target_ = RandomForestRegressor)
└── scripts/train.py             → Training orchestration script
```

## Pipeline Flow

### 1. Configuration (Hydra)
```yaml
# conf/config.yaml
defaults:
  - dataset: walmart       # Loads conf/dataset/walmart.yaml
  - model: random_forest   # Loads conf/model/random_forest.yaml

# conf/dataset/walmart.yaml
_target_: projects.retail_sales_walmart.data.walmart_dataset.WalmartDataset
data_path: projects/retail_sales_walmart/data/raw/Walmart.csv
lookback_window: 52
forecast_horizon: 12

# conf/model/random_forest.yaml
_target_: sklearn.ensemble.RandomForestRegressor
n_estimators: 100
max_depth: 20
```

### 2. Dataset Loading (TimeSeriesDataset Pattern)
```python
# Instantiate via Hydra
dataset = hydra.utils.instantiate(cfg.dataset)

# Load and preprocess data
dataset.load()  # Calls load_data() -> preprocess_data() -> _extract_features_targets()
```

**WalmartDataset Implementation:**
- Inherits from `TimeSeriesDataset`
- Overrides `load_data()`: Loads CSV, handles date parsing, validates columns
- Overrides `preprocess_data()`: Adds temporal features, store/dept encoding, holidays, economic indicators
- Automatic synthetic data generation if file not found

### 3. Data Splitting (Temporal)
```python
# Get train/val/test splits respecting temporal order
X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(
    dataset, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15
)
```

### 4. Model Instantiation (Hydra)
```python
# Works with ANY model implementing .fit()/.predict()
model = hydra.utils.instantiate(cfg.model)
```

### 5. Training (TrainingEngine)
```python
# Create engine with model and metrics
engine = TrainingEngine(model=model, metrics=metrics, config=trainer_config)

# Train model (automatically detects if iterative or single-shot)
results = engine.fit(X_train, y_train, X_val, y_val)
```

**TrainingEngine Features:**
- No model type detection - works with sklearn, PyTorch, statistical models
- Automatic detection of iterative vs single-shot training
- Early stopping, checkpointing, metric computation
- Returns training history and best model state

### 6. Evaluation
```python
# Predict on test set
y_test_pred = engine.predict(X_test)

# Compute all metrics
test_metrics = engine.compute_metrics(y_test, y_test_pred, "test_")
```

## Running the Pipeline

### Basic Training
```bash
# Use default configuration
python projects/retail_sales_walmart/scripts/train.py

# Override specific parameters
python projects/retail_sales_walmart/scripts/train.py \
    dataset.lookback_window=104 \
    model.n_estimators=200
```

### Hydra Multi-Run Sweeps
```bash
# Test multiple models
python projects/retail_sales_walmart/scripts/train.py -m \
    model=random_forest,xgboost,lightgbm

# Hyperparameter grid search
python projects/retail_sales_walmart/scripts/train.py -m \
    model.n_estimators=100,200,500 \
    model.max_depth=10,20,30
```

### Optuna Optimization
```python
# In optimize.py script
def objective(trial):
    # Override config with trial suggestions
    overrides = [
        f"model.n_estimators={trial.suggest_int('n_estimators', 50, 500)}",
        f"model.max_depth={trial.suggest_int('max_depth', 5, 30)}",
    ]
    
    # Run training with overrides
    result = hydra.compose(config_name="config", overrides=overrides)
    return train(result)
```

## Adding New Models

### Step 1: Create Model Config
```yaml
# conf/model/lstm.yaml
_target_: ml_portfolio.models.deep_learning.lstm.LSTMForecaster
hidden_size: 128
num_layers: 2
dropout: 0.2
learning_rate: 0.001
```

### Step 2: Run Training
```bash
python scripts/train.py model=lstm
```

**No code changes needed!** The pipeline automatically:
- Instantiates the LSTM model
- Detects it has iterative training (has `partial_fit` or returns history)
- Applies early stopping and checkpointing
- Computes all metrics

## Adding New Datasets

### Step 1: Create Dataset Class
```python
# projects/new_project/data/custom_dataset.py
from ml_portfolio.data.datasets import TimeSeriesDataset

class CustomDataset(TimeSeriesDataset):
    def load_data(self):
        # Load from database, API, etc.
        self.raw_data = pd.read_sql(...)
        
    def preprocess_data(self):
        # Domain-specific feature engineering
        self.data = add_custom_features(self.raw_data)
```

### Step 2: Create Dataset Config
```yaml
# conf/dataset/custom.yaml
_target_: projects.new_project.data.custom_dataset.CustomDataset
connection_string: postgresql://...
lookback_window: 24
forecast_horizon: 6
```

### Step 3: Run Training
```bash
python scripts/train.py dataset=custom model=prophet
```

## Benefits of This Architecture

### 1. Framework Agnostic
- Works with sklearn, PyTorch, TensorFlow, statistical models
- No conditional logic based on model type
- Standard interfaces: `.fit()`, `.predict()`, `.load()`

### 2. Configuration-Driven
- All hyperparameters in YAML files
- Easy to version control experiments
- Hydra enables powerful sweeps and overrides

### 3. Modular and Reusable
- Shared components in `src/ml_portfolio/`
- Project-specific extensions in `projects/*/`
- Inheritance pattern promotes code reuse

### 4. PyTorch Compatible
- `TimeSeriesDataset` implements `__len__`/`__getitem__`
- Works with PyTorch DataLoader for batch training
- Supports both tabular and sequential models

### 5. Production Ready
- Single training script for all models
- Automatic checkpointing and early stopping
- Comprehensive metrics and logging
- Easy to integrate with MLflow, Optuna, etc.

## File Summary

### Created Files
- `src/ml_portfolio/data/datasets.py` - TimeSeriesDataset base class with PyTorch compatibility
- `src/ml_portfolio/data/loaders.py` - Added get_train_val_test_splits() utility
- `projects/retail_sales_walmart/data/walmart_dataset.py` - WalmartDataset with full preprocessing
- `projects/retail_sales_walmart/data/__init__.py` - Module exports
- `projects/retail_sales_walmart/conf/config.yaml` - Main Hydra configuration
- `projects/retail_sales_walmart/conf/dataset/walmart.yaml` - Dataset configuration
- `projects/retail_sales_walmart/conf/model/random_forest.yaml` - Model configuration

### Updated Files
- `projects/retail_sales_walmart/scripts/train.py` - Added dataset.load() call

## Next Steps

1. **Test End-to-End**: Run training script to verify complete pipeline
2. **Add More Models**: Create configs for LSTM, Prophet, ARIMA, etc.
3. **Implement Optuna**: Create optimize.py for hyperparameter tuning
4. **Add Visualization**: Enhance dashboard to use model registry
5. **Documentation**: Add docstrings and usage examples
