# Dashboard User Guide

## Overview

The ML Forecasting Portfolio Dashboard is a narrative-driven interface showcasing:

- Executive storyline and architecture overview
- Model library with selection guidance
- MLflow experiment tracking and visualization
- Benchmark leaderboards with trade-off analysis
- Engineering and data science perspectives
- Comprehensive portfolio documentation

## Installation

### Option 1: Local Installation

```bash
pip install streamlit mlflow altair
```

### Option 2: Docker

```bash
docker-compose up dashboard
```

Access at: http://localhost:8501

## Primary Dashboard

### ML Forecasting Portfolio App (src/ml_portfolio/dashboard/app.py)

The main narrative-driven dashboard with 6 comprehensive tabs:

#### 1. Overview Tab

- **Executive storyline**: Problem, approach, and outcomes
- **Architecture at a glance**: Shared library structure and demo projects
- **Quick start guide**: Installation, training, optimization, MLflow commands

#### 2. Model Library Tab

- Browse by family: Gradient Boosting, Deep Learning, Ensemble Trees
- Model metadata: When to use, data regime, strengths, trade-offs
- Hydra configuration snapshots for each model

#### 3. Experiments Tab (MLflow Integration)

- Select and inspect MLflow experiments
- Filter runs by model and status
- View metrics: MAPE, RMSE, MAE on validation and test sets
- Validation MAPE chart over time (interactive)

#### 4. Benchmarks & Trade-offs Tab

- Leaderboard sorted by validation MAPE
- Best run per model with full metrics
- Model selection guidance with context-aware recommendations

#### 5. Engineering POV Tab

- Architecture and packaging patterns
- Configuration and reproducibility strategy
- Data access and preprocessing pipelines
- Training infrastructure (engines, callbacks, checkpoints)
- Experiment tracking with MLflow and Optuna
- Quality, testing, CI/CD setup
- Data quality and governance approach
- Serving and deployment interfaces

#### 6. Data Science POV Tab

- Dataset characteristics and splitting strategy
- Feature engineering details (lags, rolling stats, cyclical encodings)
- Model portfolio overview by family
- Metrics and evaluation approach
- Model selection guidance by data regime
- Known gaps and backlog items

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
# Main portfolio dashboard (recommended)
streamlit run src/ml_portfolio/dashboard/app.py

# With custom port
streamlit run src/ml_portfolio/dashboard/app.py --server.port 8502

# With virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac
streamlit run src/ml_portfolio/dashboard/app.py

# Docker
docker-compose up dashboard
```

### 2. Exploring the Model Library

1. Launch dashboard: `streamlit run src/ml_portfolio/dashboard/app.py`
1. Navigate to "Model Library" tab
1. Select model family (Gradient Boosting, Deep Learning, etc.)
1. Expand models to view:
   - When to use each model
   - Data regime requirements
   - Strengths and trade-offs
   - Hydra configuration snapshots

### 3. Viewing MLflow Experiments

1. Train models with MLflow enabled:

   ```bash
   python src/ml_portfolio/training/train.py model=lightgbm use_mlflow=true
   ```

1. Run optimization to populate experiments:

   ```bash
   python src/ml_portfolio/scripts/run_optimization.py --models lightgbm xgboost --trials 10
   ```

1. Open dashboard and navigate to "Experiments" tab

1. Select experiment from dropdown

1. Filter runs by model and status

1. View metrics table and validation MAPE chart

### 4. Analyzing Benchmarks

1. Navigate to "Benchmarks & Trade-offs" tab
1. Select experiment to benchmark
1. View leaderboard sorted by validation MAPE
1. Expand model entries for:
   - Best run metrics
   - Model selection guidance
   - Contextual strengths and trade-offs

### 5. Understanding Engineering Decisions

1. Navigate to "Engineering POV" tab
1. Review architecture and packaging approach
1. Explore configuration strategy with Hydra
1. Understand training infrastructure and callbacks
1. See CI/CD, testing, and quality gates

### 6. Deep Diving into Data Science

1. Navigate to "Data Science POV" tab
1. Review dataset characteristics and splits
1. Explore feature engineering pipeline
1. Understand model selection by data regime
1. Check known gaps and backlog items

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
1. Go to https://streamlit.io/cloud
1. Connect repository
1. Deploy

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

1. **URL Parameters**:

   ```
   http://localhost:8501/?page=Model%20Comparison
   ```

1. **Sidebar Width**:
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
1. Integrate with MLflow for experiment tracking
1. Add model download functionality
1. Implement user authentication
1. Create custom model training interface

## Support

For issues or questions:

- GitHub Issues: https://github.com/MrEleden/forecasting_demo/issues
- Documentation: See `docs/` folder
- API Reference: http://localhost:8000/docs
