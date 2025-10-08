---
title: API Reference
description: Reference documentation for reusable components in the ML forecasting portfolio.
---

# API Reference

This guide documents the public surface of the reusable library shipped in `src/ml_portfolio`. It focuses on the classes and functions you will import directly from Python code or Hydra configs.

## At a glance

- **Models** live under `ml_portfolio.models.*` with statistical wrappers and PyTorch forecasters sharing a common interface.
- **Data utilities** provide deterministic feature engineering (`StaticTimeSeriesPreprocessingPipeline`), `TimeSeriesDataset` containers, and simple loaders.
- **Evaluation** exposes metric functions and Hydra-friendly wrappers.
- **Training** supplies engines that orchestrate statistical or PyTorch training loops.

______________________________________________________________________

## Models (`ml_portfolio.models`)

### Statistical forecasters (`ml_portfolio.models.statistical`)

All statistical models inherit from `StatisticalForecaster`. They expect NumPy/Pandas inputs with shape `(n_samples, n_features)` and expose `fit`, `predict`, `get_params`, and `set_params` for sklearn interoperability.

#### `LightGBMForecaster`

```python
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster

model = LightGBMForecaster(
  n_estimators=500,
  learning_rate=0.05,
  max_depth=8,
  num_leaves=31,
  subsample=0.8,
  colsample_bytree=0.8,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Key parameters (defaults in brackets):**

- `n_estimators` (`500`): boosting rounds.
- `learning_rate` (`0.05`): shrinkage rate.
- `max_depth` (`8`): tree depth (`-1` for unlimited).
- `num_leaves` (`31`): maximum leaves per tree.
- Regularisation: `min_child_samples=20`, `min_child_weight=0.001`, `reg_alpha=0.1`, `reg_lambda=0.1`.
- Sampling: `subsample=0.8`, `colsample_bytree=0.8`, `subsample_freq=1`.
- `n_jobs=-1` enables multi-core training.

#### `CatBoostForecaster`

```python
from ml_portfolio.models.statistical.catboost import CatBoostForecaster

model = CatBoostForecaster(
  iterations=500,
  learning_rate=0.05,
  depth=8,
  loss_function="RMSE",
)
model.fit(X_train, y_train)
```

- Handles categorical features natively; pass `cat_features` during `fit` if needed.
- Regularisation knobs: `l2_leaf_reg=3.0`, `bagging_temperature=1.0`, `random_strength=1.0`.
- Early stopping: set `early_stopping_rounds` (default `50`) when providing `eval_set`.

#### `RandomForestForecaster`

```python
from ml_portfolio.models.statistical.random_forest import RandomForestForecaster

model = RandomForestForecaster(n_estimators=300, max_depth=12)
model.fit(X_train, y_train)
importances = model.get_feature_importance()
```

- Ensembles `n_estimators=200` decision trees by default.
- `max_features=1.0` selects all features; adjust for feature subsampling.
- `get_feature_importance()` returns the fitted model’s importance vector.

#### `XGBoostForecaster`

```python
from ml_portfolio.models.statistical.xgboost import XGBoostForecaster

model = XGBoostForecaster(
  n_estimators=500,
  learning_rate=0.05,
  max_depth=8,
  gamma=0.0,
)
model.fit(X_train, y_train)
```

- Extends gradient boosting with sparsity awareness. Defaults include `min_child_weight=1.0`, `subsample=0.8`, `colsample_bytree=0.8`.
- Supports `early_stopping_rounds` when an evaluation set is provided.

> **Optional dependencies**: LightGBM, CatBoost, and XGBoost need their respective packages installed (`requirements-ml.txt` or `requirements-models.txt`). Importing without installation raises `ImportError` with install guidance.

### PyTorch forecasters (`ml_portfolio.models.pytorch`)

PyTorch models mix in `PyTorchForecaster` for training utilities and expect inputs shaped `(batch, time, features)`. All support `.fit(dataloader_or_array, target)` and `.predict(data)`.

#### `LSTMForecaster`

```python
from ml_portfolio.models.pytorch.lstm import LSTMForecaster

model = LSTMForecaster(
  input_size=num_features,
  hidden_size=128,
  num_layers=2,
  dropout=0.2,
  bidirectional=False,
)
model.fit(train_loader, epochs=20)
forecast = model.predict(test_array)
```

- Accepts numpy arrays, pandas frames, or PyTorch tensors. Conversion happens internally.
- Set `device="cuda"` (or leave `"auto"`) to leverage GPU acceleration.

#### `TCNForecaster`

```python
from ml_portfolio.models.pytorch.tcn import TCNForecaster

model = TCNForecaster(
  input_size=num_features,
  num_channels=[64, 64, 128],
  kernel_size=3,
  dropout=0.2,
)
```

- Uses dilated causal convolutions for long-range dependencies.
- Exposes `num_channels` (list of channel widths per layer) in addition to `kernel_size` and `dropout`.

#### `TransformerForecaster`

```python
from ml_portfolio.models.pytorch.transformer import TransformerForecaster

model = TransformerForecaster(
  input_size=num_features,
  d_model=128,
  nhead=4,
  num_encoder_layers=2,
  num_decoder_layers=2,
)
```

- Implements an encoder-decoder transformer with positional encodings.
- Supports teacher forcing during training and scheduled sampling.

______________________________________________________________________

## Data utilities (`ml_portfolio.data`)

### `TimeSeriesDataset`

Immutable container for time-series arrays.

```python
from ml_portfolio.data.datasets import TimeSeriesDataset

dataset = TimeSeriesDataset(
  X=x_array,
  y=y_array,
  timestamps=timestamps,
  feature_names=feature_names,
  metadata={"split": "train"},
)

len(dataset)         # -> n_samples
dataset[0]           # -> (x_0, y_0)
dataset.get_data()   # -> (X, y)
dataset.timestamps   # -> numpy array or None
```

### `DatasetFactory`

Responsible for deterministic splitting and optional static feature engineering.

```python
from ml_portfolio.data.dataset_factory import DatasetFactory
from ml_portfolio.data.preprocessing import StaticTimeSeriesPreprocessingPipeline

feature_engineer = StaticTimeSeriesPreprocessingPipeline(
  date_column="Date",
  target_column="Weekly_Sales",
  group_columns=["Store", "Dept"],
  lag_features=[1, 2, 4, 8, 52],
  rolling_windows=[4, 8, 52],
  cyclical_features=["month", "week"],
)

factory = DatasetFactory(
  data_path="projects/retail_sales_walmart/data/raw/Walmart.csv",
  target_column="Weekly_Sales",
  timestamp_column="Date",
  static_feature_engineer=feature_engineer,
)

train_ds, val_ds, test_ds = factory.create_datasets()
```

`create_datasets()` returns `TimeSeriesDataset` objects (train/validation/test) with metadata describing the split. Feature names are derived from numeric columns if none are provided.

### `StaticTimeSeriesPreprocessingPipeline`

Adds safe, deterministic features **before** train/val/test splitting.

- Lag features (`target_column_lag_k`), rolling statistics, date parts, and optional cyclical encodings.
- Call `engineer_features(df)` manually, or pass the pipeline to `DatasetFactory(static_feature_engineer=...)` for automatic application.

### Data loaders (`ml_portfolio.data.loaders`)

- `SimpleDataLoader(dataset, shuffle=False)`: yields the full dataset in a single batch — ideal for sklearn-style estimators.
- `PyTorchDataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)`: thin wrapper around `torch.utils.data.DataLoader`; requires PyTorch to be installed.

______________________________________________________________________

## Evaluation metrics (`ml_portfolio.evaluation.metrics`)

Metric functions accept NumPy arrays (or array-likes) of identical shape. Hydra-compatible classes expose the same logic when instantiating from configs.

- `mae(y_true, y_pred)` / `MAEMetric`
- `rmse(y_true, y_pred)` / `RMSEMetric`
- `mape(y_true, y_pred, epsilon=1e-8)` / `MAPEMetric`
- `smape(y_true, y_pred, epsilon=1e-8)` / `SMAPEMetric`
- `directional_accuracy(y_true, y_pred)` / `DirectionalAccuracyMetric`
- `mase(y_true, y_pred, y_train, seasonal_period=1)` / `MASEMetric`

For example, the Mean Absolute Percentage Error is computed as:

$$
ext{MAPE}(y, \\hat{y}) = \\frac{100}{n} \\sum\_{i=1}^{n} \\left|\\frac{y_i - \\hat{y}\_i}{\\max(|y_i|, \\varepsilon)}\\right|
$$

`MetricCollection` groups multiple metrics and provides `compute`, `compute_and_store`, and `get_primary_metric` helpers.

______________________________________________________________________

## Benchmarking utilities (`ml_portfolio.evaluation.benchmark`)

Benchmarking tools compare multiple estimators on consistent datasets and metrics.

- `BenchmarkResult`: dataclass capturing `model_name`, `dataset_name`, `mape`, `rmse`, `mae`, timing stats, sample counts, and optional parameter metadata. Use `to_dict()` to serialize a single run.
- `ModelBenchmark(output_dir="results/benchmarks")`: manages experiment runs, result storage, and visualization helpers. The `output_dir` is created automatically if missing.

### Core workflow

```python
from ml_portfolio.evaluation.benchmark import ModelBenchmark
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster
from ml_portfolio.models.statistical.catboost import CatBoostForecaster

benchmark = ModelBenchmark(output_dir="outputs/walmart/benchmarks")

results = benchmark.run_multiple_models(
  models={
    "lightgbm": LightGBMForecaster(),
    "catboost": CatBoostForecaster(),
  },
  X_train=X_train,
  y_train=y_train,
  X_test=X_test,
  y_test=y_test,
  dataset_name="walmart_weekly_sales",
)

ranking = benchmark.get_ranking(metric="mape")
benchmark.save_results()
benchmark.plot_comparison(metric="mape")
```

### Utilities

- `run_benchmark(...)`: Benchmark a single model; returns `BenchmarkResult` with metrics and timing. Internally prints timings and metrics for quick feedback.
- `run_multiple_models(models, ...)`: Loops over a dict of name → estimator, aggregating successful runs.
- `get_results_dataframe()` / `get_summary_statistics()`: Return pandas DataFrames for downstream analysis, including grouped statistics by model.
- `get_ranking(metric="mape")`: Produces an ordered DataFrame of average metric scores.
- `save_results(filename="benchmark_results.json")`: Writes JSON with metadata to `output_dir`.
- `plot_comparison(metric)` / `plot_training_time_vs_accuracy(metric)`: Generate `matplotlib`/`seaborn` figures to visualise accuracy–speed trade-offs (set `save_fig=False` to skip disk writes).
- `generate_report()`: Creates a comprehensive text summary and saves it alongside plots; returns the report string for display in notebooks or logs.
- `load_benchmark_results(path)`: Convenience loader that recreates `BenchmarkResult` objects from a JSON export.

> Optional plotting features require Matplotlib and Seaborn (installed via `requirements-ml.txt`).

______________________________________________________________________

## Training engines (`ml_portfolio.training.engine`)

Two main engines coordinate training loops:

- `StatisticalEngine`: wraps a `StatisticalForecaster`, typically running a single `.fit()` call, metric evaluation, and optional checkpointing.
- `PyTorchEngine`: manages epochs, optimizers, callbacks, gradient clipping, and early stopping for PyTorch-based models.

Common constructor arguments:

```python
from ml_portfolio.training.engine import StatisticalEngine
from ml_portfolio.data.loaders import SimpleDataLoader

train_loader = SimpleDataLoader(train_dataset)
val_loader = SimpleDataLoader(val_dataset)

engine = StatisticalEngine(
  model=LightGBMForecaster(),
  train_loader=train_loader,
  val_loader=val_loader,
  metrics={"mape": MAPEMetric()},
  checkpoint_dir="checkpoints/lightgbm",
  verbose=True,
)

history = engine.train()
metrics = engine.evaluate(val_loader)
```

The engines share a `BaseEngine` interface with:

- `train()` → orchestrates the training loop and returns summary metrics.
- `evaluate(loader)` → computes metrics on a loader.
- `test()` → convenience wrapper around `evaluate` for the configured test set.

When `mlflow_tracker` is supplied, training and validation metrics are logged automatically.

______________________________________________________________________

## REST API skeleton (`projects/*/api`)

Project folders can expose FastAPI apps that consume trained models. Endpoints featured in the docs include:

- `POST /predict` — returns point forecasts and optional confidence intervals.
- `GET /models` — lists registered models and metrics.
- `GET /health` — simple service heartbeat.

Implementations live inside each project’s `api/` folder and are designed to load models through the shared registry (`ml_portfolio.models.registry`).

______________________________________________________________________

## Working with examples

- `hidden_size` (int): LSTM hidden size. Default: 128
- `num_layers` (int): Number of LSTM layers. Default: 2
- `dropout` (float): Dropout rate. Default: 0.2
- `bidirectional` (bool): Use bidirectional LSTM. Default: False
- See `examples/notebooks/phase2_phase3.ipynb` for an end-to-end walkthrough covering feature engineering, model training, benchmarking, and MLflow logging with the APIs above.

**Methods:**

- `fit(X, y, epochs=100)`: Train the model
- `predict(X)`: Make predictions
- `predict_quantiles(X, quantiles=[0.1, 0.5, 0.9])`: Probabilistic forecasts

### ml_portfolio.data

Data loading and preprocessing utilities.

#### DatasetFactory

```python
from ml_portfolio.data.dataset_factory import DatasetFactory

factory = DatasetFactory(
    data_path="data/walmart.csv",
    target_column="Weekly_Sales",
    timestamp_column="Date",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

train, val, test = factory.get_datasets()
```

**Parameters:**

- `data_path` (str): Path to CSV file
- `target_column` (str): Name of target column
- `timestamp_column` (str): Name of timestamp column
- `feature_columns` (List\[str\], optional): List of feature columns
- `train_ratio` (float): Training set ratio. Default: 0.7
- `val_ratio` (float): Validation set ratio. Default: 0.15
- `test_ratio` (float): Test set ratio. Default: 0.15
- `static_feature_engineer` (callable, optional): Feature engineering pipeline

**Methods:**

- `get_datasets()`: Returns (train, val, test) DataFrames
- `get_feature_names()`: Returns list of feature names

#### StaticTimeSeriesPreprocessingPipeline

```python
from ml_portfolio.data.preprocessing import StaticTimeSeriesPreprocessingPipeline

pipeline = StaticTimeSeriesPreprocessingPipeline(
    date_column='Date',
    target_column='Weekly_Sales',
    group_columns=['Store'],
    lag_features=[1, 2, 4, 8, 13, 26, 52],
    rolling_windows=[4, 8, 13, 26],
    date_features=True,
    cyclical_features=['month', 'dayofweek', 'quarter']
)

data_transformed = pipeline.fit_transform(data)
```

**Parameters:**

- `date_column` (str): Date column name
- `target_column` (str): Target column name
- `group_columns` (List\[str\]): Columns for grouping (e.g., Store)
- `lag_features` (List\[int\]): Lag periods to create
- `rolling_windows` (List\[int\]): Rolling window sizes
- `date_features` (bool): Extract date features (year, month, day, etc.)
- `cyclical_features` (List\[str\]): Features to encode cyclically (sin/cos)

**Methods:**

- `fit_transform(df)`: Fit and transform data
- `transform(df)`: Transform data using fitted pipeline

### ml_portfolio.evaluation

Evaluation metrics for forecasting.

#### MAPEMetric

Mean Absolute Percentage Error.

```python
from ml_portfolio.evaluation.metrics import MAPEMetric

metric = MAPEMetric(epsilon=1e-8)
mape = metric(y_true, y_pred)
```

**Formula:** $\\text{MAPE} = \\frac{100}{n} \\sum\_{i=1}^{n} \\left|\\frac{y_i - \\hat{y}\_i}{y_i + \\epsilon}\\right|$

#### RMSEMetric

Root Mean Squared Error.

```python
from ml_portfolio.evaluation.metrics import RMSEMetric

metric = RMSEMetric()
rmse = metric(y_true, y_pred)
```

**Formula:** $\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum\_{i=1}^{n} (y_i - \\hat{y}\_i)^2}$

#### MAEMetric

Mean Absolute Error.

```python
from ml_portfolio.evaluation.metrics import MAEMetric

metric = MAEMetric()
mae = metric(y_true, y_pred)
```

**Formula:** $\\text{MAE} = \\frac{1}{n} \\sum\_{i=1}^{n} |y_i - \\hat{y}\_i|$

### ml_portfolio.training

Training engine and utilities.

#### StatisticalEngine

Training engine for statistical models.

```python
from ml_portfolio.training.engine import StatisticalEngine

engine = StatisticalEngine(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    metrics=metrics,
    checkpoint_dir="checkpoints/"
)

engine.train()
metrics = engine.evaluate(test_loader)
```

**Parameters:**

- `model`: Model instance
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `test_loader` (optional): Test data loader
- `metrics`: List of metric instances
- `checkpoint_dir`: Directory for checkpoints
- `verbose`: Print training progress

**Methods:**

- `train()`: Train the model
- `evaluate(loader)`: Evaluate on data loader
- `save_checkpoint(path)`: Save model checkpoint
- `load_checkpoint(path)`: Load model checkpoint

## Configuration System

### Hydra Configuration

All configurations use Hydra's compositional config system.

#### Base Configuration

```yaml
# config.yaml
defaults:
  - dataset_factory: default
  - feature_engineering: default
  - model: lightgbm
  - metrics: default
  - dataloader: simple
  - engine: statistical

experiment_name: forecasting_experiment
seed: 42
use_mlflow: true
```

#### Model Configuration

```yaml
# model/lightgbm.yaml
model:
  _target_: ml_portfolio.models.statistical.lightgbm.LightGBMForecaster
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 8
  num_leaves: 31
```

#### Optuna Configuration

```yaml
# optuna/lightgbm.yaml
study_name: lightgbm_optimization
direction: minimize
n_trials: 50

params:
  model.n_estimators: range(100, 1000, 50)
  model.learning_rate: interval(0.001, 0.3)
  model.max_depth: range(3, 12)
```

## REST API

### Endpoints

#### POST /predict

Make a forecast.

**Request:**

```json
{
  "store_id": 1,
  "horizon": 7,
  "features": {
    "temperature": 75.5,
    "fuel_price": 3.2,
    "holiday": false
  }
}
```

**Response:**

```json
{
  "predictions": [12345.67, 13456.78, ...],
  "confidence_intervals": {
    "lower": [11000, 12000, ...],
    "upper": [13500, 14500, ...]
  },
  "model_name": "lightgbm_v1",
  "timestamp": "2025-10-06T12:00:00Z"
}
```

#### GET /models

List available models.

**Response:**

```json
{
  "models": [
    {
      "name": "lightgbm_v1",
      "version": "1",
      "stage": "Production",
      "metrics": {
        "MAPE": 32.19,
        "RMSE": 86414.68
      }
    }
  ]
}
```

#### GET /health

Health check.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Examples

### Basic Training

```python
from ml_portfolio.data.dataset_factory import DatasetFactory
from ml_portfolio.models.statistical import LightGBMForecaster
from ml_portfolio.evaluation.metrics import MAPEMetric

# Load data
factory = DatasetFactory(
    data_path="data/walmart.csv",
    target_column="Weekly_Sales"
)
train, val, test = factory.get_datasets()

# Train model
model = LightGBMForecaster(n_estimators=500)
model.fit(train['X'], train['y'])

# Evaluate
predictions = model.predict(val['X'])
metric = MAPEMetric()
mape = metric(val['y'], predictions)
print(f"Validation MAPE: {mape:.2f}%")
```

### Hyperparameter Optimization

```python
import optuna
from ml_portfolio.models.statistical import LightGBMForecaster

def objective(trial):
    model = LightGBMForecaster(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 12)
    )
    model.fit(train['X'], train['y'])
    predictions = model.predict(val['X'])
    return metric(val['y'], predictions)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Probabilistic Forecasting

```python
from ml_portfolio.models.deep_learning import LSTMForecaster

model = LSTMForecaster(input_size=42, hidden_size=128)
model.fit(train['X'], train['y'])

# Get quantile predictions
quantiles = model.predict_quantiles(
    test['X'],
    quantiles=[0.1, 0.5, 0.9]
)

print(f"10th percentile: {quantiles['0.1']}")
print(f"Median: {quantiles['0.5']}")
print(f"90th percentile: {quantiles['0.9']}")
```
