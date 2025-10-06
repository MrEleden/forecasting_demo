"""
FastAPI serving endpoint for forecasting models.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="ML Forecasting API", description="Production-ready time series forecasting API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ForecastRequest(BaseModel):
    """Forecast request schema."""

    store_id: int = Field(..., description="Store ID", ge=1)
    horizon: int = Field(..., description="Forecast horizon (days)", ge=1, le=365)
    features: Optional[Dict[str, float]] = Field(
        None, description="Additional features (temperature, fuel_price, etc.)"
    )
    model_name: str = Field("lightgbm", description="Model to use")
    model_version: str = Field("latest", description="Model version")
    include_intervals: bool = Field(True, description="Include prediction intervals")
    confidence_level: float = Field(0.9, description="Confidence level for intervals", ge=0.5, le=0.99)


class ForecastResponse(BaseModel):
    """Forecast response schema."""

    predictions: List[float] = Field(..., description="Point forecasts")
    timestamps: List[str] = Field(..., description="Forecast timestamps")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(None, description="Prediction intervals")
    model_name: str = Field(..., description="Model used")
    model_version: str = Field(..., description="Model version")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    timestamp: str = Field(..., description="Response timestamp")


class ModelInfo(BaseModel):
    """Model information schema."""

    name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    created_at: str
    description: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    models_loaded: int
    timestamp: str


# Global model cache
MODEL_CACHE = {}
PREPROCESSING_CACHE = {}


def load_model(model_name: str, model_version: str = "latest"):
    """Load model from MLflow or disk."""
    cache_key = f"{model_name}_{model_version}"

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    try:
        # Try MLflow first
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        MODEL_CACHE[cache_key] = model
        return model
    except Exception:
        # Fall back to disk
        model_path = Path(f"models/{model_name}_v{model_version}.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            MODEL_CACHE[cache_key] = model
            return model
        raise HTTPException(status_code=404, detail=f"Model {model_name} version {model_version} not found")


def load_preprocessing_pipeline(model_name: str):
    """Load preprocessing pipeline."""
    if model_name in PREPROCESSING_CACHE:
        return PREPROCESSING_CACHE[model_name]

    pipeline_path = Path(f"models/preprocessing_{model_name}.pkl")
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
        PREPROCESSING_CACHE[model_name] = pipeline
        return pipeline
    return None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "ML Forecasting API", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", version="1.0.0", models_loaded=len(MODEL_CACHE), timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    """
    Make a forecast.

    Example:
    ```json
    {
        "store_id": 1,
        "horizon": 7,
        "features": {
            "temperature": 75.5,
            "fuel_price": 3.2,
            "holiday": false
        },
        "model_name": "lightgbm",
        "include_intervals": true
    }
    ```
    """
    try:
        # Load model
        model = load_model(request.model_name, request.model_version)

        # Load preprocessing pipeline
        pipeline = load_preprocessing_pipeline(request.model_name)

        # Prepare input data
        input_data = pd.DataFrame(
            {
                "Store": [request.store_id] * request.horizon,
                "Date": pd.date_range(start=datetime.now(), periods=request.horizon, freq="D"),
                **(request.features or {}),
            }
        )

        # Apply preprocessing
        if pipeline is not None:
            input_data = pipeline.transform(input_data)

        # Make predictions
        predictions = model.predict(input_data)

        # Generate timestamps
        timestamps = (
            pd.date_range(start=datetime.now(), periods=request.horizon, freq="D").strftime("%Y-%m-%d").tolist()
        )

        # Compute prediction intervals (if requested and model supports it)
        confidence_intervals = None
        if request.include_intervals:
            try:
                # Try quantile prediction
                lower_q = (1 - request.confidence_level) / 2
                upper_q = 1 - lower_q

                if hasattr(model, "predict_quantiles"):
                    quantiles = model.predict_quantiles(input_data, quantiles=[lower_q, 0.5, upper_q])
                    confidence_intervals = {
                        "lower": quantiles[str(lower_q)].tolist(),
                        "upper": quantiles[str(upper_q)].tolist(),
                    }
                else:
                    # Estimate intervals from residuals (simple approach)
                    std_error = np.std(predictions) * 1.96  # 95% CI approximation
                    confidence_intervals = {
                        "lower": (predictions - std_error).tolist(),
                        "upper": (predictions + std_error).tolist(),
                    }
            except Exception as e:
                print(f"Could not compute intervals: {e}")

        # Get model metadata
        try:
            client = mlflow.tracking.MlflowClient()
            model_metadata = client.get_latest_versions(request.model_name, stages=[request.model_version])[0]

            run = client.get_run(model_metadata.run_id)
            metrics = run.data.metrics
        except Exception:
            metrics = None

        return ForecastResponse(
            predictions=predictions.tolist(),
            timestamps=timestamps,
            confidence_intervals=confidence_intervals,
            model_name=request.model_name,
            model_version=request.model_version,
            metrics=metrics,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    try:
        client = mlflow.tracking.MlflowClient()
        models = []

        for model in client.search_registered_models():
            for version in client.get_latest_versions(model.name):
                run = client.get_run(version.run_id)

                models.append(
                    ModelInfo(
                        name=model.name,
                        version=version.version,
                        stage=version.current_stage,
                        metrics=run.data.metrics,
                        created_at=datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                        description=version.description,
                    )
                )

        return models

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str, version: str = "latest"):
    """Get information about a specific model."""
    try:
        client = mlflow.tracking.MlflowClient()

        if version == "latest":
            versions = client.get_latest_versions(model_name)
            if not versions:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            model_version = versions[0]
        else:
            model_version = client.get_model_version(model_name, version)

        run = client.get_run(model_version.run_id)

        return ModelInfo(
            name=model_name,
            version=model_version.version,
            stage=model_version.current_stage,
            metrics=run.data.metrics,
            created_at=datetime.fromtimestamp(model_version.creation_timestamp / 1000).isoformat(),
            description=model_version.description,
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/models/{model_name}/promote")
async def promote_model(model_name: str, version: str, stage: str = "Production"):
    """
    Promote a model version to a stage.

    Stages: Staging, Production, Archived
    """
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=model_name, version=version, stage=stage)
        return {
            "message": f"Model {model_name} version {version} promoted to {stage}",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
