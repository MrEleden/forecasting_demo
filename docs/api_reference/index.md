# API Reference

Complete API documentation for ML Forecasting Portfolio.

## Core Modules

### ml_portfolio.models

Model implementations for time series forecasting.

#### Statistical Models

##### LightGBMForecaster

```python
from ml_portfolio.models.statistical.lightgbm import LightGBMForecaster

model = LightGBMForecaster(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31
)
```

**Parameters:**
- `n_estimators` (int): Number of boosting rounds. Default: 100
- `learning_rate` (float): Learning rate. Default: 0.1
- `max_depth` (int): Maximum tree depth. Default: -1 (no limit)
- `num_leaves` (int): Maximum number of leaves. Default: 31
- `min_child_samples` (int): Minimum samples per leaf. Default: 20
- `reg_alpha` (float): L1 regularization. Default: 0.0
- `reg_lambda` (float): L2 regularization. Default: 0.0
- `subsample` (float): Row sampling ratio. Default: 1.0
- `colsample_bytree` (float): Column sampling ratio. Default: 1.0

**Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `get_feature_importance()`: Get feature importances

##### XGBoostForecaster

```python
from ml_portfolio.models.statistical.xgboost import XGBoostForecaster

model = XGBoostForecaster(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8
)
```

**Parameters:**
- Similar to LightGBM with XGBoost-specific additions
- `gamma` (float): Minimum loss reduction. Default: 0.0
- `min_child_weight` (float): Minimum sum of instance weight. Default: 1.0

##### CatBoostForecaster

```python
from ml_portfolio.models.statistical.catboost import CatBoostForecaster

model = CatBoostForecaster(
    iterations=500,
    learning_rate=0.05,
    depth=8
)
```

**Parameters:**
- `iterations` (int): Number of boosting iterations
- `depth` (int): Tree depth. Default: 6
- `l2_leaf_reg` (float): L2 regularization. Default: 3.0
- `bagging_temperature` (float): Bagging temperature. Default: 1.0

#### Deep Learning Models

##### LSTMForecaster

```python
from ml_portfolio.models.deep_learning.lstm import LSTMForecaster

model = LSTMForecaster(
    input_size=42,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)
```

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_size` (int): LSTM hidden size. Default: 128
- `num_layers` (int): Number of LSTM layers. Default: 2
- `dropout` (float): Dropout rate. Default: 0.2
- `bidirectional` (bool): Use bidirectional LSTM. Default: False

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
- `feature_columns` (List[str], optional): List of feature columns
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
- `group_columns` (List[str]): Columns for grouping (e.g., Store)
- `lag_features` (List[int]): Lag periods to create
- `rolling_windows` (List[int]): Rolling window sizes
- `date_features` (bool): Extract date features (year, month, day, etc.)
- `cyclical_features` (List[str]): Features to encode cyclically (sin/cos)

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

**Formula:** $\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i + \epsilon}\right|$

#### RMSEMetric

Root Mean Squared Error.

```python
from ml_portfolio.evaluation.metrics import RMSEMetric

metric = RMSEMetric()
rmse = metric(y_true, y_pred)
```

**Formula:** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

#### MAEMetric

Mean Absolute Error.

```python
from ml_portfolio.evaluation.metrics import MAEMetric

metric = MAEMetric()
mae = metric(y_true, y_pred)
```

**Formula:** $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

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
