# Benchmark Suite Guide

## Overview

The benchmark suite provides a comprehensive framework for comparing multiple forecasting models on standardized metrics and datasets.

## Quick Start

```bash
# Run benchmark with default settings
python scripts/run_benchmark.py

# Run on specific dataset
python scripts/run_benchmark.py --dataset walmart

# Test specific models
python scripts/run_benchmark.py --models lightgbm,catboost,xgboost

# Custom output directory
python scripts/run_benchmark.py --output-dir my_benchmarks/
```

## Features

### 1. Automated Model Comparison
- Train multiple models on same data
- Calculate standardized metrics (MAPE, RMSE, MAE)
- Measure training and prediction time
- Track model parameters

### 2. Comprehensive Reporting
- JSON results file
- Text report with rankings
- Visualization plots (PNG)
- Summary statistics

### 3. Flexible Configuration
- Choose datasets
- Select models to test
- Custom output directory
- Easy extensibility

## Available Models

The benchmark suite supports:

### Statistical Models
- ARIMA
- Prophet
- SARIMAX
- Exponential Smoothing

### Machine Learning Models
- **LightGBM** (fast, accurate)
- **CatBoost** (handles categoricals well)
- **XGBoost** (widely used)
- RandomForest
- Ridge Regression
- ElasticNet

### Deep Learning Models
- LSTM
- TCN (Temporal Convolutional Network)
- Transformer

## Usage

### Basic Usage

```python
from ml_portfolio.evaluation.benchmark import ModelBenchmark

# Initialize
benchmark = ModelBenchmark(output_dir="results/benchmarks")

# Run single model
result = benchmark.run_benchmark(
    model=lightgbm_model,
    model_name="LightGBM",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    dataset_name="walmart"
)

# Run multiple models
models = {
    "LightGBM": lgbm_model,
    "CatBoost": catboost_model,
    "XGBoost": xgb_model
}

results = benchmark.run_multiple_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    dataset_name="walmart"
)
```

### Advanced Usage

```python
# Get results as DataFrame
df = benchmark.get_results_dataframe()

# Get summary statistics
summary = benchmark.get_summary_statistics()

# Get model rankings
ranking = benchmark.get_ranking(metric="mape")

# Save results
benchmark.save_results("my_benchmark.json")

# Generate plots
benchmark.plot_comparison(metric="mape")
benchmark.plot_training_time_vs_accuracy(metric="mape")

# Generate report
report = benchmark.generate_report()
print(report)
```

## Output Files

### 1. benchmark_results.json
Complete results in JSON format:
```json
{
  "results": [
    {
      "model_name": "LightGBM",
      "dataset_name": "walmart",
      "mape": 0.0856,
      "rmse": 1234.56,
      "mae": 987.65,
      "training_time": 2.34,
      "prediction_time": 0.05,
      "n_samples": 1000,
      "n_features": 15,
      "params": {
        "n_estimators": 100,
        "max_depth": 5
      }
    }
  ],
  "timestamp": "2025-10-06T12:00:00",
  "n_models": 5,
  "n_datasets": 1
}
```

### 2. benchmark_report.txt
Text report with rankings and statistics:
```
================================================================================
BENCHMARK REPORT
================================================================================

Total Models Tested: 5
Total Datasets: 1
Total Runs: 5

--------------------------------------------------------------------------------
RANKINGS BY MAPE
--------------------------------------------------------------------------------
   rank model_name  avg_mape
1     1   LightGBM    0.0856
2     2   CatBoost    0.0923
3     3    XGBoost    0.0987

...
```

### 3. Visualization Plots

**benchmark_comparison_mape.png**:
- Bar chart of average MAPE by model
- Box plot showing metric distribution

**benchmark_tradeoff.png**:
- Scatter plot: training time vs accuracy
- Shows speed/accuracy trade-offs

## Metrics Explained

### MAPE (Mean Absolute Percentage Error)
```
MAPE = (1/n) * Σ |actual - predicted| / |actual| * 100
```
- **Lower is better**
- Measures average prediction error as percentage
- Good for comparing models across different scales

### RMSE (Root Mean Squared Error)
```
RMSE = √[(1/n) * Σ (actual - predicted)²]
```
- **Lower is better**
- Penalizes large errors more heavily
- Same units as target variable

### MAE (Mean Absolute Error)
```
MAE = (1/n) * Σ |actual - predicted|
```
- **Lower is better**
- Average absolute prediction error
- More robust to outliers than RMSE

## Interpreting Results

### Model Rankings
```python
ranking = benchmark.get_ranking("mape")
```

**How to interpret**:
- Rank 1: Best performing model
- Compare avg_mape values
- Lower MAPE = better predictions

### Training Time vs Accuracy
```python
benchmark.plot_training_time_vs_accuracy()
```

**Look for**:
- Models in bottom-left corner (fast + accurate)
- Trade-offs between speed and accuracy
- Diminishing returns on training time

### Summary Statistics
```python
summary = benchmark.get_summary_statistics()
```

**Key metrics**:
- **Mean**: Average performance
- **Std**: Consistency (lower = more stable)
- **Min**: Best case performance

## Adding Custom Models

### 1. Create Model Wrapper
```python
class MyCustomModel:
    def fit(self, X, y):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        return predictions
```

### 2. Add to Benchmark
```python
models = {
    "MyModel": MyCustomModel(),
    "LightGBM": lgbm_model
}

benchmark.run_multiple_models(models, X_train, y_train, X_test, y_test, "my_dataset")
```

## Best Practices

### 1. Use Consistent Data Splits
```python
# Use same random seed
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. Test Multiple Datasets
```python
datasets = ["walmart", "ola", "tsi"]

for dataset in datasets:
    # Load data
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    # Run benchmark
    benchmark.run_multiple_models(models, X_train, y_train, X_test, y_test, dataset)
```

### 3. Include Model Parameters
```python
result = benchmark.run_benchmark(
    model=model,
    model_name="LightGBM",
    params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
    },
    ...
)
```

### 4. Save Results Regularly
```python
# After each dataset
benchmark.save_results(f"benchmark_{dataset}.json")
```

## Comparison Examples

### Example 1: Gradient Boosting Comparison
```bash
python scripts/run_benchmark.py --models lightgbm,catboost,xgboost
```

**Expected output**:
- LightGBM: Fastest training
- CatBoost: Best with categoricals
- XGBoost: Most stable

### Example 2: Statistical vs ML
```python
models = {
    "ARIMA": arima_model,
    "Prophet": prophet_model,
    "LightGBM": lgbm_model
}
```

**Typical results**:
- Statistical: Faster, good for simple patterns
- ML: More accurate for complex patterns

### Example 3: Model Size Trade-off
```python
models = {
    "Ridge": ridge_model,           # Fast, simple
    "RandomForest": rf_model,       # Medium complexity
    "LightGBM": lgbm_model,         # High performance
    "LSTM": lstm_model              # Slow, complex
}
```

## Troubleshooting

### Issue: Model training fails
**Check**:
- Data has no NaN values
- Feature types are correct
- Target variable is numeric

### Issue: MAPE is inf or NaN
**Cause**: Zero values in actual data
**Solution**: Use RMSE or MAE instead, or add epsilon

### Issue: Training time too long
**Solutions**:
- Use subset of data
- Reduce model complexity
- Enable early stopping

### Issue: Memory error
**Solutions**:
- Reduce dataset size
- Use incremental learning
- Increase swap space

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Benchmark
  run: python scripts/run_benchmark.py --dataset walmart

- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: results/benchmarks/
```

### Automated Reporting
```python
# Compare with baseline
baseline_mape = 0.10
current_mape = results["mape"].min()

if current_mape < baseline_mape:
    print("✅ New best model found!")
else:
    print("⚠️ No improvement over baseline")
```

## Next Steps

1. **Expand model coverage**: Add more model types
2. **Cross-validation**: Implement k-fold CV
3. **Automated tuning**: Integrate with Optuna
4. **Statistical tests**: Add significance testing
5. **Ensemble methods**: Benchmark ensemble strategies

## References

- [Benchmark results format](../results/benchmarks/benchmark_results.json)
- [API documentation](../src/ml_portfolio/evaluation/benchmark.py)
- [Example scripts](../scripts/run_benchmark.py)
