# MLflow Integration - Dashboard & API

## Overview

The forecasting portfolio has **3 main components** that work with MLflow:

1. **Benchmark Script** - Queries MLflow for model comparison
2. **FastAPI** - Serves models from MLflow Model Registry
3. **Streamlit Dashboard** - Visualizes benchmark results (JSON-based, not MLflow direct)

---

## 1. Benchmark Script (MLflow Query)

**File**: `src/ml_portfolio/scripts/run_benchmark.py`

### What It Does:

Retrieves experiment results from MLflow tracking server instead of training models from scratch.

### MLflow Operations:

```python
# Connect to MLflow
mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

# Search experiments
experiments = client.search_experiments()

# Get all runs from experiments
runs = client.search_runs(experiment_ids=[exp_id])

# Extract metrics
for run in runs:
    metrics = run.data.metrics  # test_MAPEMetric, test_RMSEMetric, test_MAEMetric
    params = run.data.params    # model params
    tags = run.data.tags        # model_type, dataset
```

### Key Features:

- **Fast**: No model training, just queries
- **Flexible Filtering**: By experiment, dataset, or model names
- **Metric Support**: Handles multiple naming conventions
  - `test_mape`, `mape`, `test_MAPEMetric`, `MAPEMetric`
  - `test_rmse`, `rmse`, `test_RMSEMetric`, `RMSEMetric`
  - `test_mae`, `mae`, `test_MAEMetric`, `MAEMetric`
- **Smart Model Detection**: Extracts model names from tags/params

### Usage:

```bash
# Query all experiments
python src/ml_portfolio/scripts/run_benchmark.py

# Specific experiment
python src/ml_portfolio/scripts/run_benchmark.py --experiment-name "walmart_sales_forecasting"

# Filter by dataset
python src/ml_portfolio/scripts/run_benchmark.py --dataset walmart

# Filter by models
python src/ml_portfolio/scripts/run_benchmark.py --models lightgbm,catboost,xgboost
```

### Output:

```
================================================================================
BENCHMARK RESULTS FROM MLFLOW
================================================================================

  model_name dataset_name    mape        rmse        mae training_time
    CatBoost      walmart 20.9452 151975.6868 90920.9954        4.9563
    LightGBM      walmart  9.8851  89070.4494 57975.3302        0.3893
RandomForest      walmart  6.0664  90927.6590 46235.3087        0.6178
     XGBoost      walmart  8.1161  92547.7578 55530.8477        0.2042

--------------------------------------------------------------------------------
RANKINGS (by MAPE)
--------------------------------------------------------------------------------

 rank   model_name      mape
    1 RandomForest  6.066410
    2      XGBoost  8.116089
    3     LightGBM  9.885105
    4     CatBoost 20.945188
```

### Generated Files:

- `results/benchmarks/mlflow_benchmark_results.json`
- `results/benchmarks/mlflow_benchmark_results.csv`
- `results/benchmarks/mlflow_benchmark_mape.png`
- `results/benchmarks/mlflow_benchmark_rmse.png`
- `results/benchmarks/mlflow_benchmark_report.txt`

---

## 2. FastAPI (MLflow Model Registry)

**File**: `src/ml_portfolio/api/main.py`

### What It Does:

Production-ready API that **loads and serves models from MLflow Model Registry**.

### MLflow Operations:

#### Load Model for Prediction:
```python
def load_model(model_name: str, model_version: str = "latest"):
    """Load model from MLflow or disk."""
    # Load from MLflow Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
```

#### List Available Models:
```python
@app.get("/models")
async def list_models():
    """List all available models."""
    client = mlflow.tracking.MlflowClient()

    for model in client.search_registered_models():
        for version in client.get_latest_versions(model.name):
            run = client.get_run(version.run_id)
            # Return model info with metrics
```

#### Get Model Metadata:
```python
@app.get("/models/{model_name}")
async def get_model_info(model_name: str, version: str = "latest"):
    """Get information about a specific model."""
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(model_name)[0]
    run = client.get_run(model_version.run_id)

    # Return metrics, version, stage, creation date
    return ModelInfo(
        name=model_name,
        version=model_version.version,
        stage=model_version.current_stage,
        metrics=run.data.metrics,
        created_at=...,
    )
```

#### Promote Model to Production:
```python
@app.post("/models/{model_name}/promote")
async def promote_model(model_name: str, version: str, stage: str = "Production"):
    """Promote a model version to a stage."""
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage  # Staging, Production, Archived
    )
```

### API Endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Root info |
| `/health` | GET | Health check, models loaded count |
| `/predict` | POST | Make forecast with MLflow model |
| `/models` | GET | List all registered models |
| `/models/{name}` | GET | Get model info (metrics, version, stage) |
| `/models/{name}/promote` | POST | Promote model to Production/Staging |

### Usage Examples:

```bash
# Health check
curl http://localhost:8000/health

# List all models
curl http://localhost:8000/models

# Get model info
curl http://localhost:8000/models/lightgbm

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "horizon": 7,
    "model_name": "lightgbm",
    "model_version": "latest",
    "include_intervals": true,
    "confidence_level": 0.9
  }'

# Promote model to production
curl -X POST http://localhost:8000/models/lightgbm/promote \
  -d '{"version": "1", "stage": "Production"}'
```

### Response Example:

```json
{
  "predictions": [125000.5, 130000.2, 128500.7, ...],
  "timestamps": ["2025-10-07", "2025-10-08", "2025-10-09", ...],
  "confidence_intervals": {
    "lower": [120000.0, 125000.0, 123000.0, ...],
    "upper": [130000.0, 135000.0, 134000.0, ...]
  },
  "model_name": "lightgbm",
  "model_version": "2",
  "metrics": {
    "test_mape": 6.5,
    "test_rmse": 85000.0,
    "test_mae": 48000.0
  },
  "timestamp": "2025-10-06T15:30:00Z"
}
```

---

## 3. Streamlit Dashboard (JSON-based)

**File**: `src/ml_portfolio/dashboard/app.py`

### What It Does:

Interactive web dashboard for visualizing model comparisons and predictions.

### Current Implementation:

**Note**: Currently loads from JSON files, **NOT directly from MLflow**. It uses the benchmark results generated by the benchmark script.

```python
@st.cache_data
def load_benchmark_data(filepath: str = "results/benchmarks/benchmark_results.json"):
    """Load benchmark results from file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data["results"])
```

### Dashboard Pages:

#### 1. Overview Page
- **Total models**: Count of available models
- **Datasets**: Multi-domain stats
- **Test coverage**: Current metrics
- **Best model**: From benchmark results (lowest MAPE)
- **Best MAPE**: Best performance metric
- **Avg training time**: Average across models
- **Total benchmark runs**: Number of experiments

#### 2. Model Comparison Page
- **Filters**: By dataset, models, primary metric
- **Results table**: Sortable, highlight best performers
- **Bar charts**: MAPE/RMSE/MAE comparison
- **Scatter plots**: Training time vs accuracy tradeoff
- **Box plots**: Metric distribution across runs
- **Rankings**: Sorted by selected metric

#### 3. Predictions Explorer Page
- **File upload**: CSV with actual vs predicted values
- **Auto-detection**: Date, actual, predicted columns
- **Metrics calculation**: MAPE, RMSE, MAE
- **Time series plot**: Actual vs predicted overlay
- **Residual analysis**:
  - Residuals over time
  - Residual distribution histogram

#### 4. Benchmark Results Page
- **Summary statistics**: Mean, std, min, max by model
- **Detailed results**: Full dataset with all metrics
- **Sorting**: By any metric
- **CSV download**: Export results button

### Usage:

```bash
# Start dashboard
streamlit run src/ml_portfolio/dashboard/app.py

# Access at http://localhost:8501
```

### Data Flow:

```
Training → MLflow Tracking → Benchmark Script → JSON Files → Streamlit Dashboard
   ↓           ↓                    ↓                ↓              ↓
 Models    Experiments         Query MLflow       Save         Visualize
            Metrics             Get Results      Results      Comparisons
```

---

## Complete Workflow

### 1. Train Models (Logs to MLflow)
```bash
# Train different models
python src/ml_portfolio/training/train.py model=lightgbm dataset=walmart
python src/ml_portfolio/training/train.py model=catboost dataset=walmart
python src/ml_portfolio/training/train.py model=xgboost dataset=walmart
```

**MLflow Logs**:
- Metrics: `test_MAPEMetric`, `test_RMSEMetric`, `test_MAEMetric`, `training_time`
- Params: Model hyperparameters
- Tags: `model_type`, `dataset`
- Artifacts: Model files, plots

### 2. Query MLflow with Benchmark Script
```bash
python src/ml_portfolio/scripts/run_benchmark.py --experiment-name "walmart_sales_forecasting"
```

**Generates**:
- `results/benchmarks/mlflow_benchmark_results.json`
- `results/benchmarks/mlflow_benchmark_results.csv`
- Comparison plots

### 3. Visualize in Dashboard
```bash
streamlit run src/ml_portfolio/dashboard/app.py
```

**Shows**:
- Model rankings (RandomForest: 6.07%, XGBoost: 8.12%, LightGBM: 9.89%, CatBoost: 20.95%)
- Performance comparisons
- Training time tradeoffs

### 4. Serve Best Model via API
```bash
# Start API server
uvicorn src.ml_portfolio.api.main:app --reload

# Get best model info
curl http://localhost:8000/models/randomforest

# Make predictions
curl -X POST http://localhost:8000/predict -d '{"model_name": "randomforest", ...}'
```

**API Uses MLflow For**:
- Loading models from registry
- Getting model metadata
- Retrieving performance metrics
- Model versioning and staging

---

## Key Differences

| Component | MLflow Integration | Purpose | Data Source |
|-----------|-------------------|---------|-------------|
| **Benchmark Script** | ✅ Direct Query | Compare models | MLflow Tracking |
| **FastAPI** | ✅ Model Registry | Serve predictions | MLflow Models |
| **Streamlit Dashboard** | ❌ JSON Files | Visualize results | Benchmark JSON |

---

## Enhancement Opportunity

The **Streamlit Dashboard could be enhanced** to query MLflow directly:

```python
# Proposed enhancement
@st.cache_data
def load_mlflow_data(experiment_name: str):
    """Load data directly from MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id])

    results = []
    for run in runs:
        results.append({
            'model_name': run.data.tags.get('model_type'),
            'mape': run.data.metrics.get('test_MAPEMetric'),
            'rmse': run.data.metrics.get('test_RMSEMetric'),
            # ... more fields
        })
    return pd.DataFrame(results)
```

This would eliminate the need for intermediate JSON files.

---

## Summary

### Current State:

1. **Benchmark Script** ✅ Queries MLflow → Generates JSON/CSV/Plots
2. **FastAPI** ✅ Loads models from MLflow Registry → Serves predictions
3. **Dashboard** ⚠️ Reads JSON files → Visualizes comparisons

### Best Result Flow:

```
MLflow Experiment: walmart_sales_forecasting
  ↓
Benchmark Script queries MLflow
  ↓
Best Model: RandomForest (6.07% MAPE)
  ↓
Option 1: View in Dashboard (JSON-based)
Option 2: Serve via API (MLflow Model Registry)
  ↓
API loads RandomForest from MLflow → /predict endpoint
  ↓
Production predictions with model metadata
```

### Commands to See Best Results:

```bash
# 1. Query MLflow for best results
python src/ml_portfolio/scripts/run_benchmark.py --experiment-name "walmart_sales_forecasting"

# 2. View in dashboard
streamlit run src/ml_portfolio/dashboard/app.py
# Navigate to "Model Comparison" page

# 3. Serve best model via API
uvicorn src.ml_portfolio.api.main:app --reload
curl http://localhost:8000/models/randomforest

# 4. Make predictions with best model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "randomforest", "store_id": 1, "horizon": 7}'
```

---

**The FastAPI is the most MLflow-integrated component**, directly loading models from the registry and serving them with full metadata!
