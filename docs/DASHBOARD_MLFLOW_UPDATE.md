# Streamlit Dashboard - MLflow Integration Update

## Changes Made

Updated the Streamlit dashboard to **query MLflow directly** instead of relying on intermediate JSON files.

---

## New Features

### 1. Live MLflow Integration ✅

**Before**: Dashboard loaded data from `results/benchmarks/benchmark_results.json`
**After**: Dashboard queries MLflow tracking server in real-time

```python
@st.cache_data
def load_mlflow_runs(experiment_name=None, dataset_filter=None):
    """Load model runs directly from MLflow tracking server."""
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    # Query experiments and extract metrics
```

### 2. Flexible Data Source Selection 📊

Users can now choose between:
- **MLflow (Live)** - Real-time query of MLflow tracking server
- **JSON (Cache)** - Load from saved benchmark results

Toggle in sidebar:
```
Data Source
○ MLflow (Live)     ← Default
○ JSON (Cache)
```

### 3. Experiment Filtering 🔍

When using MLflow (Live), filter by specific experiments:
- All Experiments
- walmart_sales_forecasting
- rideshare_demand_ola
- transportation_tsi
- (Any custom experiments)

### 4. Automatic Metric Detection 🎯

Supports multiple metric naming conventions:
- Standard: `test_mape`, `test_rmse`, `test_mae`
- Alternative: `mape`, `rmse`, `mae`
- Hydra format: `test_MAPEMetric`, `test_RMSEMetric`, `test_MAEMetric`

### 5. Smart Model Name Extraction 🤖

Extracts model names from:
- Tags: `model_type`, `model_name`
- Params: `model`, `model._target_`
- Fallback: "Unknown"

---

## Updated Functions

### Core Functions:

```python
# NEW: Query MLflow directly
load_mlflow_runs(experiment_name=None, dataset_filter=None)

# NEW: Get list of experiments
get_mlflow_experiments()

# UPDATED: Flexible data loading
load_benchmark_data(data_source="MLflow (Live)", experiment_name=None, dataset_filter=None)

# EXISTING: Fallback JSON loading
load_benchmark_data_from_json(filepath="results/benchmarks/mlflow_benchmark_results.json")
```

---

## Dashboard Pages Updated

### 1. Overview Page
- Shows real-time stats from MLflow
- Best model identification
- Total benchmark runs count
- Data source indicator

### 2. Model Comparison Page
- Live model performance comparison
- Interactive filtering by dataset/models
- Multiple metrics (MAPE, RMSE, MAE)
- Training time vs accuracy tradeoff

### 3. Predictions Explorer Page
- No changes (file upload based)

### 4. Benchmark Results Page
- Live benchmark data from MLflow
- Summary statistics
- Sortable detailed results
- CSV download

---

## Usage

### Start Dashboard:

```bash
streamlit run src/ml_portfolio/dashboard/app.py
```

### Access:
Open browser to **http://localhost:8501**

### Select Data Source:

**Option 1: MLflow (Live)** - Recommended
- Queries MLflow tracking server in real-time
- Always up-to-date with latest experiments
- Filter by specific experiments

**Option 2: JSON (Cache)**
- Loads from saved benchmark results
- Faster load times (cached data)
- Useful when MLflow is not accessible

---

## Screenshots Flow

### Sidebar Configuration:

```
Navigation
○ Overview
○ Model Comparison
○ Predictions Explorer
○ Benchmark Results

─────────────────────
Data Source
● MLflow (Live)
○ JSON (Cache)

─────────────────────
MLflow Filters
Experiment: [All Experiments ▼]
  - All Experiments
  - walmart_sales_forecasting
  - rideshare_demand_ola
```

### Overview Page with MLflow Data:

```
Quick Stats
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Best Model      │ Best MAPE       │ Avg Training    │ Total Benchmark │
│ (MAPE)          │                 │ Time            │ Runs            │
│ RandomForest    │ 6.0664          │ 1.57s           │ 4               │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

📊 Data loaded from: MLflow (Live) (Experiment: walmart_sales_forecasting)
```

### Model Comparison with Live Data:

```
Results Table
┌──────────────┬──────────────┬─────────┬────────────┬───────────┬──────────────┐
│ model_name   │ dataset_name │ mape    │ rmse       │ mae       │ training_time│
├──────────────┼──────────────┼─────────┼────────────┼───────────┼──────────────┤
│ RandomForest │ walmart      │ 6.0664  │ 90927.6590 │ 46235.3087│ 0.6178       │
│ XGBoost      │ walmart      │ 8.1161  │ 92547.7578 │ 55530.8477│ 0.2042       │
│ LightGBM     │ walmart      │ 9.8851  │ 89070.4494 │ 57975.3302│ 0.3893       │
│ CatBoost     │ walmart      │ 20.9452 │ 151975.6868│ 90920.9954│ 4.9563       │
└──────────────┴──────────────┴─────────┴────────────┴───────────┴──────────────┘
```

---

## Data Flow

### New Architecture:

```
Training → MLflow Tracking
              ↓
         (Real-time)
              ↓
    Streamlit Dashboard ← Direct Query
              ↓
         Visualize
```

### Old Architecture:

```
Training → MLflow → Benchmark Script → JSON Files → Streamlit Dashboard
```

---

## Benefits

### ✅ Real-Time Updates
- No need to re-run benchmark script
- See latest experiments immediately
- Auto-refresh with new training runs

### ✅ Experiment Exploration
- Filter by specific experiments
- Compare across multiple experiments
- Drill down into specific datasets

### ✅ Always Current
- No stale data from cached JSON files
- Direct access to MLflow metrics
- Latest model performance

### ✅ Flexible Workflow
- Use MLflow (Live) for exploration
- Use JSON (Cache) for presentations/reports
- Switch between sources seamlessly

---

## Backward Compatibility

The dashboard still supports JSON-based loading for:
- Offline presentations
- Cached benchmark results
- Systems without MLflow access

Simply select "JSON (Cache)" from the data source radio button.

---

## Technical Details

### Dependencies Added:
- `mlflow` (already in requirements.txt)

### Files Modified:
- `src/ml_portfolio/dashboard/app.py` (~636 lines)

### Key Changes:
1. Added `import mlflow`
2. Created `load_mlflow_runs()` function
3. Created `get_mlflow_experiments()` helper
4. Updated `load_benchmark_data()` to accept data source
5. Added sidebar data source selection
6. Updated all page logic to use new loading function

---

## Example Queries

### Query All Experiments:
```python
df = load_mlflow_runs()
# Returns all runs from all experiments
```

### Query Specific Experiment:
```python
df = load_mlflow_runs(experiment_name="walmart_sales_forecasting")
# Returns only walmart experiment runs
```

### Query with Dataset Filter:
```python
df = load_mlflow_runs(dataset_filter="walmart")
# Returns only walmart dataset runs
```

---

## Testing

### Verify MLflow Connection:
```bash
python -c "import mlflow; client = mlflow.tracking.MlflowClient(tracking_uri='file:./mlruns'); print(len(client.search_experiments()), 'experiments found')"
```

### Start Dashboard:
```bash
streamlit run src/ml_portfolio/dashboard/app.py
```

### Check in Browser:
1. Navigate to http://localhost:8501
2. Select "MLflow (Live)" from sidebar
3. Choose experiment from dropdown
4. View live data in all pages

---

## Future Enhancements

Potential improvements:
- Add real-time auto-refresh option
- Show run timestamps and durations
- Add run comparison (diff view)
- Export to PDF/PowerPoint
- Add model registry integration
- Show experiment lineage/hierarchy

---

## Summary

The Streamlit dashboard now provides **direct MLflow integration**, allowing users to:
- 📊 Query experiments in real-time
- 🔍 Filter by experiment name
- 🎯 View latest model performance
- 📈 Compare models interactively
- 💾 Fall back to cached JSON when needed

**No more intermediate JSON files needed!** The dashboard is now a true MLflow visualization layer.
