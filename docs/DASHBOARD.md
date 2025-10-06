# Dashboard User Guide

## Overview

The ML Forecasting Dashboard provides an interactive interface for:
- Comparing model performance
- Exploring predictions
- Analyzing benchmark results
- Visualizing time series forecasts

## Installation

### Option 1: Local Installation
```bash
pip install streamlit plotly
```

### Option 2: Docker
```bash
docker-compose up dashboard
```

Access at: http://localhost:8501

## Features

### 1. Overview Page

**Quick Statistics**:
- Total models available
- Test coverage metrics
- Best performing model
- Average training time

**Model Catalog**:
- Statistical models (ARIMA, Prophet)
- ML models (LightGBM, CatBoost, XGBoost)
- Deep learning models (LSTM, TCN)

### 2. Model Comparison

**Interactive Visualizations**:
- Bar charts comparing metrics (MAPE, RMSE, MAE)
- Box plots showing metric distributions
- Scatter plots for training time vs accuracy trade-offs

**Filters**:
- Select specific datasets
- Filter by model types
- Choose primary metric

**Features**:
- Highlight best/worst performers
- Download results as CSV
- Customizable views

### 3. Predictions Explorer

**Upload Predictions**:
- CSV format with actual and predicted values
- Automatic column detection
- Manual column selection

**Visualizations**:
- Time series plots (actual vs predicted)
- Residual analysis over time
- Residual distribution histograms

**Metrics**:
- MAPE, RMSE, MAE calculated automatically
- Visual error indicators

### 4. Benchmark Results

**Summary Statistics**:
- Mean, std, min, max for each model
- Grouped by model type
- Formatted tables with color coding

**Detailed Results**:
- Full benchmark data
- Sortable columns
- Filterable views
- Export functionality

## Usage Examples

### 1. Running the Dashboard

```bash
# Local
streamlit run src/ml_portfolio/dashboard/app.py

# With custom port
streamlit run src/ml_portfolio/dashboard/app.py --server.port 8502

# Docker
docker-compose up dashboard
```

### 2. Viewing Benchmark Results

1. Run benchmark suite first:
   ```bash
   python scripts/run_benchmark.py
   ```

2. Open dashboard and navigate to "Benchmark Results"

3. Explore interactive visualizations and tables

### 3. Exploring Predictions

1. Prepare CSV with columns:
   - Date/timestamp
   - Actual values
   - Predicted values

2. Navigate to "Predictions Explorer"

3. Upload CSV file

4. View automatic analysis and visualizations

### 4. Comparing Models

1. Navigate to "Model Comparison"

2. Select dataset from sidebar

3. Choose models to compare

4. Adjust metric (MAPE, RMSE, MAE)

5. Explore interactive charts

## Data Format

### Benchmark Results JSON
```json
{
  "results": [
    {
      "model_name": "LightGBM",
      "dataset_name": "walmart",
      "mape": 0.1234,
      "rmse": 5678.9,
      "mae": 4321.0,
      "training_time": 12.34,
      "prediction_time": 0.56
    }
  ],
  "timestamp": "2025-10-06T12:00:00",
  "n_models": 5,
  "n_datasets": 3
}
```

### Predictions CSV
```csv
date,actual,predicted
2024-01-01,1000,980
2024-01-02,1100,1050
2024-01-03,950,970
```

## Customization

### 1. Theme
Streamlit supports light/dark themes. Configure in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### 2. Page Configuration
Modify `app.py`:

```python
st.set_page_config(
    page_title="Custom Title",
    page_icon="📊",
    layout="wide",
)
```

### 3. Adding New Pages

Add new page to radio button:
```python
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Model Comparison", "Your New Page"],
)

if page == "Your New Page":
    # Your code here
    pass
```

## Performance Tips

### 1. Use Caching
```python
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)
```

### 2. Lazy Loading
Only load data when needed:
```python
if st.button("Load Data"):
    data = load_heavy_data()
```

### 3. Optimize Plots
Use `use_container_width=True` for responsive plots:
```python
st.plotly_chart(fig, use_container_width=True)
```

## Troubleshooting

### Issue: Dashboard won't start
**Solution**: Check if port 8501 is available
```bash
netstat -ano | findstr :8501  # Windows
lsof -i :8501                 # Linux/Mac
```

### Issue: Benchmark results not found
**Solution**: Run benchmark suite first
```bash
python scripts/run_benchmark.py
```

### Issue: Upload file error
**Solution**: Ensure CSV has proper format with headers

### Issue: Slow performance
**Solution**:
- Enable caching decorators
- Reduce data size
- Use sampling for large datasets

## Deployment

### 1. Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy

### 2. Docker
```bash
docker build -t forecasting-dashboard .
docker run -p 8501:8501 forecasting-dashboard
```

### 3. Production Server
```bash
# With nginx reverse proxy
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# nginx config
location /dashboard {
    proxy_pass http://localhost:8501;
}
```

## API Integration

The dashboard can connect to the FastAPI endpoint:

```python
import requests

# Get model info
response = requests.get("http://localhost:8000/models")
models = response.json()

# Make predictions
prediction_request = {
    "model_name": "lightgbm",
    "data": [...]
}
response = requests.post("http://localhost:8000/predict", json=prediction_request)
predictions = response.json()["predictions"]
```

## Screenshots

### Overview Page
- Quick metrics and statistics
- Model catalog
- System status

### Model Comparison
- Bar charts of metrics
- Box plots for distributions
- Scatter plots for trade-offs

### Predictions Explorer
- Time series visualization
- Residual analysis
- Metric cards

### Benchmark Results
- Detailed tables
- Summary statistics
- Export functionality

## Tips and Tricks

1. **Keyboard Shortcuts**:
   - `R`: Rerun app
   - `C`: Clear cache
   - `Ctrl+C`: Stop server

2. **URL Parameters**:
   ```
   http://localhost:8501/?page=Model%20Comparison
   ```

3. **Sidebar Width**:
   Use CSS to adjust:
   ```python
   st.markdown("""
   <style>
   [data-testid="stSidebar"] {width: 300px !important;}
   </style>
   """, unsafe_allow_html=True)
   ```

## Next Steps

1. Add real-time prediction updates
2. Integrate with MLflow for experiment tracking
3. Add model download functionality
4. Implement user authentication
5. Create custom model training interface

## Support

For issues or questions:
- GitHub Issues: https://github.com/MrEleden/forecasting_demo/issues
- Documentation: See `docs/` folder
- API Reference: http://localhost:8000/docs
