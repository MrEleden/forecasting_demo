# 🔧 API Reference

Complete reference for the ML Portfolio forecasting library (`ml_portfolio` package).

## 📋 Quick Reference

```python
# Core imports
from ml_portfolio.models.statistical import ARIMAWrapper, ProphetWrapper
from ml_portfolio.models.deep_learning import LSTMForecaster, TCNForecaster
from ml_portfolio.data import TimeSeriesDataset, create_time_windows
from ml_portfolio.evaluation import TimeSeriesBacktester
from ml_portfolio.utils import DataCache, load_csv, save_parquet
```

## 📊 Data Module (`ml_portfolio.data`)

### **Datasets**

#### **TimeSeriesDataset**
```python
from ml_portfolio.data import TimeSeriesDataset

class TimeSeriesDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for time series forecasting."""
    
    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        forecast_horizon: int = 1,
        stride: int = 1,
        transforms: Optional[callable] = None
    ):
        """
        Initialize time series dataset.
        
        Args:
            data: Time series data (1D array)
            window_size: Input sequence length
            forecast_horizon: Number of steps to predict
            stride: Step size between windows
            transforms: Optional data transforms
        """
```

**Example Usage:**
```python
import numpy as np
from ml_portfolio.data import TimeSeriesDataset

# Create dataset
data = np.random.randn(1000)  # 1000 time steps
dataset = TimeSeriesDataset(
    data=data,
    window_size=24,      # Use 24 time steps as input
    forecast_horizon=6,  # Predict next 6 steps
    stride=1
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### **Time Series Utilities**

#### **create_time_windows**
```python
def create_time_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series data.
    
    Args:
        data: Input time series (1D array)
        window_size: Size of input windows
        stride: Step size between windows
        forecast_horizon: Number of future steps to predict
        
    Returns:
        Tuple of (X, y) arrays for training
    """
```

**Example:**
```python
from ml_portfolio.data import create_time_windows

# Create windows
X, y = create_time_windows(
    data=ts_data.values,
    window_size=24,
    forecast_horizon=6
)
print(f"X shape: {X.shape}, y shape: {y.shape}")
# Output: X shape: (n_windows, 24), y shape: (n_windows, 6)
```

### **Data Loaders**

#### **WalmartDataLoader**
```python
from ml_portfolio.data.loaders import WalmartDataLoader

class WalmartDataLoader:
    """Specialized loader for Walmart dataset."""
    
    def __init__(
        self,
        file_path: str,
        date_column: str = "Date",
        target_column: str = "Weekly_Sales",
        store_column: str = "Store",
        dept_column: str = "Dept"
    ):
        """Initialize Walmart data loader."""
        
    def load(self, aggregate_by: str = "total") -> pd.DataFrame:
        """
        Load and preprocess Walmart data.
        
        Args:
            aggregate_by: How to aggregate data ('total', 'store', 'dept')
            
        Returns:
            Preprocessed DataFrame
        """
```

### **Transforms**

#### **StandardScaler**
```python
from ml_portfolio.data.transforms import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(ts_data)
original_data = scaler.inverse_transform(scaled_data)
```

#### **LogTransform**
```python
from ml_portfolio.data.transforms import LogTransform

log_transform = LogTransform()
log_data = log_transform.fit_transform(ts_data)
```

## 🧠 Models Module (`ml_portfolio.models`)

### **Statistical Models**

#### **ARIMAWrapper**
```python
from ml_portfolio.models.statistical import ARIMAWrapper

class ARIMAWrapper(BaseEstimator, RegressorMixin):
    """ARIMA model with sklearn compatibility."""
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        """
        Initialize ARIMA wrapper.
        
        Args:
            order: (p, d, q) order of ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order
            trend: Trend component ('n', 'c', 't', 'ct')
            enforce_stationarity: Whether to enforce stationarity
            enforce_invertibility: Whether to enforce invertibility
        """
    
    def fit(self, X, y):
        """Fit ARIMA model to time series data."""
        
    def predict(self, X):
        """Make predictions for given number of steps."""
        
    def predict_with_intervals(self, steps: int = 1, alpha: float = 0.05):
        """
        Make predictions with confidence intervals.
        
        Args:
            steps: Number of steps to predict
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with predictions, lower_ci, upper_ci, prediction_dates
        """
```

**Example Usage:**
```python
from ml_portfolio.models.statistical import ARIMAWrapper
import pandas as pd

# Load data
ts_data = pd.read_csv("data.csv", index_col="date", parse_dates=True)["value"]

# Fit model
model = ARIMAWrapper(order=(2, 1, 2))
model.fit(None, ts_data)

# Simple predictions
pred = model.predict(10)  # Predict next 10 steps

# Predictions with confidence intervals
result = model.predict_with_intervals(steps=10, alpha=0.05)
print(result.keys())  # ['predictions', 'lower_ci', 'upper_ci', 'prediction_dates']
```

#### **ProphetWrapper**
```python
from ml_portfolio.models.statistical import ProphetWrapper

class ProphetWrapper(BaseEstimator, RegressorMixin):
    """Prophet model with sklearn compatibility."""
    
    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        yearly_seasonality: Union[bool, str] = "auto",
        weekly_seasonality: Union[bool, str] = "auto",
        daily_seasonality: Union[bool, str] = "auto"
    ):
        """Initialize Prophet wrapper."""
```

### **Deep Learning Models**

#### **LSTMForecaster**
```python
from ml_portfolio.models.deep_learning import LSTMForecaster

class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_size: Number of output predictions
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
```

#### **TCNForecaster**
```python
from ml_portfolio.models.deep_learning import TCNForecaster

class TCNForecaster(nn.Module):
    """Temporal Convolutional Network for forecasting."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        num_channels: List[int] = [32, 32, 32],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """Initialize TCN forecaster."""
```

### **Model Registry**

#### **ModelRegistry**
```python
from ml_portfolio.models import ModelRegistry

class ModelRegistry:
    """Central registry for trained models."""
    
    def __init__(self, base_path: str = "models/"):
        """Initialize model registry."""
        
    def save_model(
        self,
        model,
        name: str,
        version: str = None,
        metadata: dict = None
    ):
        """Save model with metadata."""
        
    def load_model(self, name: str, version: str = "latest"):
        """Load model by name and version."""
        
    def list_models(self) -> List[dict]:
        """List all registered models."""
```

**Example:**
```python
from ml_portfolio.models import ModelRegistry

registry = ModelRegistry()

# Save model
registry.save_model(
    model=fitted_arima,
    name="walmart_arima_baseline",
    version="v1.0",
    metadata={
        "dataset": "walmart",
        "performance": {"rmse": 1250.5, "mape": 8.3},
        "hyperparameters": {"order": [2, 1, 2]}
    }
)

# Load model
model = registry.load_model("walmart_arima_baseline", version="v1.0")

# List models
models = registry.list_models()
```

## 📈 Evaluation Module (`ml_portfolio.evaluation`)

### **Backtesting**

#### **TimeSeriesBacktester**
```python
from ml_portfolio.evaluation import TimeSeriesBacktester

class TimeSeriesBacktester:
    """Walk-forward validation for time series models."""
    
    def __init__(
        self,
        model,
        initial_train_size: int,
        step_size: int = 1,
        forecast_horizon: int = 1,
        gap: int = 0
    ):
        """
        Initialize backtester.
        
        Args:
            model: Model to evaluate (sklearn-compatible)
            initial_train_size: Initial training window size
            step_size: Number of steps to advance each iteration
            forecast_horizon: Number of steps to predict
            gap: Gap between training and prediction
        """
        
    def backtest(self, data: pd.Series) -> dict:
        """
        Perform walk-forward backtesting.
        
        Returns:
            Dictionary with predictions, actuals, and metrics
        """
```

**Example:**
```python
from ml_portfolio.evaluation import TimeSeriesBacktester
from ml_portfolio.models.statistical import ARIMAWrapper

# Set up model and backtester
model = ARIMAWrapper(order=(2, 1, 2))
backtester = TimeSeriesBacktester(
    model=model,
    initial_train_size=100,
    step_size=1,
    forecast_horizon=5
)

# Run backtest
results = backtester.backtest(ts_data)
print(f"RMSE: {results['rmse']:.2f}")
print(f"MAPE: {results['mape']:.2f}%")
```

### **Metrics**

#### **Time Series Metrics**
```python
from ml_portfolio.models.metrics import rmse, mae, mape, smape

# Calculate metrics
rmse_score = rmse(y_true, y_pred)
mae_score = mae(y_true, y_pred)
mape_score = mape(y_true, y_pred)
smape_score = smape(y_true, y_pred)
```

### **Plotting**

#### **Plotting Functions**
```python
from ml_portfolio.evaluation.plots import (
    plot_forecast,
    plot_residuals,
    plot_timeseries_decomposition,
    plot_prediction_intervals
)

# Plot forecast vs actual
fig = plot_forecast(
    actual=y_true,
    predicted=y_pred,
    dates=date_index,
    title="ARIMA Forecast"
)

# Plot residuals
fig = plot_residuals(residuals=y_true - y_pred)

# Plot time series decomposition
fig = plot_timeseries_decomposition(ts_data, model_type="additive")
```

## 🛠️ Utils Module (`ml_portfolio.utils`)

### **I/O Operations**

#### **Data Loading/Saving**
```python
from ml_portfolio.utils.io import load_csv, save_parquet, load_pickle

# Load CSV with date parsing
df = load_csv("data.csv", date_column="Date", date_format="%Y-%m-%d")

# Save as parquet
save_parquet(df, "processed_data.parquet")

# Pickle operations
save_pickle(model, "model.pkl")
loaded_model = load_pickle("model.pkl")
```

#### **DataCache**
```python
from ml_portfolio.utils.io import DataCache

cache = DataCache(cache_dir="cache/")

# Cache expensive computation
if cache.exists("processed_data"):
    data = cache.get("processed_data")
else:
    data = expensive_processing()
    cache.set("processed_data", data)
```

#### **Project Paths**
```python
from ml_portfolio.utils.io import get_project_root, get_data_path

# Get project root
root = get_project_root()

# Get data path for specific project
data_path = get_data_path("retail_sales_walmart", "processed")
```

### **Configuration**

#### **Config Loading**
```python
from ml_portfolio.utils.config import load_config

# Load Hydra configuration
config = load_config("conf/config.yaml")

# Access nested values
model_config = config.model
dataset_config = config.dataset
```

## 🔌 Training Module (`ml_portfolio.training`)

### **Training Engine**

#### **TrainingEngine**
```python
from ml_portfolio.training import TrainingEngine

class TrainingEngine:
    """Training loop with callbacks and early stopping."""
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "auto",
        callbacks: List = None
    ):
        """Initialize training engine."""
        
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        save_best: bool = True
    ) -> dict:
        """
        Train model with validation.
        
        Returns:
            Training history dictionary
        """
```

### **Callbacks**

#### **EarlyStopping**
```python
from ml_portfolio.training.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)
```

#### **ModelCheckpoint**
```python
from ml_portfolio.training.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath="models/checkpoints/best_model.pth",
    save_best_only=True,
    monitor="val_loss"
)
```

## 🚀 Pipeline Module (`ml_portfolio.pipelines`)

### **Classical Pipeline**

#### **ClassicalPipeline**
```python
from ml_portfolio.pipelines import ClassicalPipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = ClassicalPipeline([
    ("scaler", StandardScaler()),
    ("model", ARIMAWrapper(order=(2, 1, 2)))
])

# Fit and predict
pipeline.fit(None, ts_data)
predictions = pipeline.predict(10)
```

### **Hybrid Pipeline**

#### **HybridPipeline**
```python
from ml_portfolio.pipelines import HybridPipeline

# Sklearn preprocessing + PyTorch model
pipeline = HybridPipeline(
    preprocessor=StandardScaler(),
    model=LSTMForecaster(input_size=1, hidden_size=64)
)
```

## 🔧 Configuration Examples

### **Hydra Integration**

#### **Model Instantiation**
```yaml
# conf/model/arima.yaml
_target_: ml_portfolio.models.statistical.ARIMAWrapper
order: [2, 1, 2]
seasonal_order: [0, 0, 0, 0]
trend: null
```

```python
# In training script
from hydra.utils import instantiate

model = instantiate(config.model)
```

#### **Dataset Configuration**
```yaml
# conf/dataset/walmart.yaml
_target_: ml_portfolio.data.loaders.WalmartDataLoader
file_path: "data/raw/train.csv"
date_column: "Date"
target_column: "Weekly_Sales"
```

## 🚨 Error Handling

### **Common Exceptions**
```python
from ml_portfolio.exceptions import (
    ModelNotFittedError,
    DataValidationError,
    ConfigurationError
)

try:
    predictions = model.predict(10)
except ModelNotFittedError:
    print("Model must be fitted before prediction")
```

## 📊 Performance Guidelines

### **Memory Efficiency**
- Use `DataLoader` with appropriate `batch_size`
- Clear intermediate results with `del variable`
- Use `torch.no_grad()` for inference

### **Speed Optimization**
- Use vectorized operations in numpy/pandas
- Enable GPU for PyTorch models
- Cache expensive computations with `DataCache`

## 🔗 Related Documentation

- **[Development Guide](DEVELOPMENT_GUIDE.md)**: Setup and coding standards
- **[Configuration Guide](CONFIGURATION_GUIDE.md)**: Hydra configuration details
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Common issues and fixes

---

*API Reference current as of October 2025 | Package version: 0.1.0*