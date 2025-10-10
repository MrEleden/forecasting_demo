"""FastAPI service stub for Walmart forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from mlflow.tracking import MlflowClient

    MLFLOW_CLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MlflowClient = None  # type: ignore
    MLFLOW_CLIENT_AVAILABLE = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"


class PredictionRequest(BaseModel):
    store: int
    features: dict


class BatchRequest(BaseModel):
    store: int
    records: list[dict]


app = FastAPI(title="Walmart Forecasting API", version="0.1.0")


@app.on_event("startup")
def load_registry() -> None:
    """Ensure model artefacts exist on startup."""
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_model(store: int):
    model_path = MODEL_DIR / f"random_forest_store{store}.joblib"
    scaler_path = MODEL_DIR / f"random_forest_store{store}_scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model artefacts not found. Train the notebook or pipeline before serving predictions.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        model, scaler = _load_model(request.store)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    feature_df = pd.DataFrame([request.features])
    features_scaled = scaler.transform(feature_df)
    prediction = model.predict(features_scaled)[0]
    return {"store": request.store, "prediction": float(prediction)}


@app.post("/predict-batch")
async def predict_batch(request: BatchRequest):
    try:
        model, scaler = _load_model(request.store)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    feature_df = pd.DataFrame(request.records)
    features_scaled = scaler.transform(feature_df)
    predictions = model.predict(features_scaled)
    return {
        "store": request.store,
        "predictions": [float(pred) for pred in predictions],
        "count": len(predictions),
    }


@app.get("/health")
async def health() -> dict:
    response: dict[str, Optional[float | str]] = {"status": "ok"}

    if MLFLOW_CLIENT_AVAILABLE and MlflowClient is not None:
        tracking_dir = Path(__file__).resolve().parents[2] / "mlruns"
        if tracking_dir.exists():
            try:
                client = MlflowClient(tracking_uri=tracking_dir.resolve().as_uri())
                experiments = client.list_experiments()
                if experiments:
                    latest_run = None
                    for exp in experiments:
                        runs = client.search_runs(
                            [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
                        )
                        if runs:
                            run = runs[0]
                            if latest_run is None or run.info.start_time > latest_run.info.start_time:
                                latest_run = run

                    if latest_run is not None:
                        metrics = latest_run.data.metrics
                        response.update(
                            {
                                "latest_run": latest_run.info.run_name or latest_run.info.run_id,
                                "latest_experiment": latest_run.info.experiment_id,
                                "latest_mape": float(metrics.get("val_MAPE")) if "val_MAPE" in metrics else None,
                                "latest_rmse": float(metrics.get("val_RMSE")) if "val_RMSE" in metrics else None,
                            }
                        )
            except Exception:
                response["mlflow"] = "unavailable"

    return response
