"""
Hydra-based training script for Walmart forecasting.

This script uses Hydra configuration management to train models with the custom
WalmartTimeSeriesDataset following the inheritance pattern.

Usage:
    # Default training (Random Forest with custom dataset)
    python train.py

    # Train specific model
    python train.py model=lstm
    python train.py model=gradient_boosting
    python train.py model=svr

    # Multi-run experiments
    python train.py -m model=random_forest,gradient_boosting,lstm,svr

    # Override config parameters
    python train.py model=random_forest model.n_estimators=200 dataset.lookback_window=26

    # Sweep experiments
    python train.py -m model=random_forest model.max_depth=10,15,20 model.n_estimators=50,100,200
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "data"))
sys.path.insert(0, str(project_root.parent.parent / "src"))

# Direct imports - will fail fast if missing
from walmart_dataset import WalmartTimeSeriesDataset


def prepare_data_for_sklearn(dataset):
    """Convert dataset sequences to sklearn-compatible format."""
    X, y = [], []

    for x_seq, y_seq in dataset.sequences:
        # Flatten the sequence to features (lookback_window → features)
        X.append(x_seq.flatten())
        # Take the mean of forecast horizon as target
        y.append(y_seq.mean())

    return np.array(X), np.array(y)


def prepare_data_for_pytorch(dataset, scaler=None):
    """Convert dataset sequences to PyTorch-compatible format."""
    X, y = [], []

    for x_seq, y_seq in dataset.sequences:
        X.append(x_seq)
        y.append(y_seq)

    X = np.array(X)  # Shape: (n_samples, lookback_window, 1)
    y = np.array(y)  # Shape: (n_samples, forecast_horizon, 1)

    # Scale the data if needed
    if scaler is None:
        scaler = StandardScaler()
        X_flat = X.reshape(-1, 1)
        scaler.fit(X_flat)

    # Transform X and y
    X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    return torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled), scaler


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive forecasting metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Directional accuracy
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    else:
        directional_accuracy = 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Directional_Accuracy": directional_accuracy}


def is_deep_learning_model(model_name):
    """Check if model requires PyTorch/deep learning setup."""
    deep_learning_models = ["lstm", "transformer", "tcn"]
    return model_name.lower() in deep_learning_models


def train_sklearn_model(model, X_train, X_test, y_train, y_test, cfg):
    """Train sklearn-compatible model."""
    print(f"🔄 Training {cfg.model._target_.split('.')[-1]}...")

    # Scale data for models that need it
    model_name = cfg.model._target_.split(".")[-1].lower()
    if model_name in ["svr", "linearregression", "ridge"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    # Train model
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    return y_pred_train, y_pred_test, training_time


def train_pytorch_model(model, X_train, X_test, y_train, y_test, cfg):
    """Train PyTorch model (LSTM, etc.)."""
    print(f"🧠 Training {cfg.model._target_.split('.')[-1]}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert to PyTorch tensors
    X_train_tensor = X_train.to(device)
    y_train_tensor = y_train.to(device)
    X_test_tensor = X_test.to(device)

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    start_time = time.time()

    num_epochs = cfg.trainer.max_epochs
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.squeeze(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs}: Loss = {loss.item():.6f}")

    training_time = time.time() - start_time

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred_train_tensor = model(X_train_tensor)
        y_pred_test_tensor = model(X_test_tensor)

    # Convert back to numpy
    y_pred_train = y_pred_train_tensor.cpu().numpy().mean(axis=1)
    y_pred_test = y_pred_test_tensor.cpu().numpy().mean(axis=1)

    return y_pred_train, y_pred_test, training_time


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""

    print("🏪 Walmart Forecasting with Hydra Configuration")
    print("=" * 70)

    # Print configuration
    print("📋 Configuration:")
    print(f"   Dataset: {cfg.dataset._target_.split('.')[-1]}")
    print(f"   Model: {cfg.model._target_.split('.')[-1]}")
    print(f"   Lookback Window: {cfg.dataset.lookback_window}")
    print(f"   Forecast Horizon: {cfg.dataset.forecast_horizon}")

    # Create dataset using Hydra instantiation
    print(f"\n📊 Creating dataset...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    print(f"✅ Dataset: {len(dataset)} sequences")

    # Get dataset insights
    if hasattr(dataset, "get_walmart_insights"):
        insights = dataset.get_walmart_insights()
        print(f"📈 Average weekly sales: ${insights['sales_statistics']['mean_weekly_sales']:,.0f}")

    # Create model using Hydra instantiation
    print(f"\n🤖 Creating model...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"✅ Model: {cfg.model._target_.split('.')[-1]}")

    # Prepare data based on model type
    model_name = cfg.model._target_.split(".")[-1].lower()

    if is_deep_learning_model(model_name):
        # PyTorch data preparation
        X, y, scaler = prepare_data_for_pytorch(dataset)

        # Split data
        split_idx = int(cfg.dataset.train_ratio * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"🔄 PyTorch data prepared: {X.shape} → {y.shape}")

        # Train PyTorch model
        y_pred_train, y_pred_test, training_time = train_pytorch_model(model, X_train, X_test, y_train, y_test, cfg)

        # Convert back to original scale
        y_train_orig = scaler.inverse_transform(y_train.mean(axis=(1, 2), keepdims=True).reshape(-1, 1)).flatten()
        y_test_orig = scaler.inverse_transform(y_test.mean(axis=(1, 2), keepdims=True).reshape(-1, 1)).flatten()
        y_pred_train_orig = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
        y_pred_test_orig = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

    else:
        # Sklearn data preparation
        X, y = prepare_data_for_sklearn(dataset)

        # Split data
        split_idx = int(cfg.dataset.train_ratio * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"🔄 Sklearn data prepared: {X.shape} → {y.shape}")

        # Train sklearn model
        y_pred_train, y_pred_test, training_time = train_sklearn_model(model, X_train, X_test, y_train, y_test, cfg)

        # Use original values
        y_train_orig, y_test_orig = y_train, y_test
        y_pred_train_orig, y_pred_test_orig = y_pred_train, y_pred_test

    print(f"✅ Training completed in {training_time:.2f}s")

    # Calculate metrics
    train_metrics = calculate_metrics(y_train_orig, y_pred_train_orig)
    test_metrics = calculate_metrics(y_test_orig, y_pred_test_orig)

    # Display results
    print(f"\n📊 Results:")
    print("-" * 40)
    print(f"📚 Training Metrics:")
    print(f"   MAE:  ${train_metrics['MAE']:,.0f}")
    print(f"   RMSE: ${train_metrics['RMSE']:,.0f}")
    print(f"   MAPE: {train_metrics['MAPE']:.2f}%")

    print(f"\n🧪 Test Metrics:")
    print(f"   MAE:  ${test_metrics['MAE']:,.0f}")
    print(f"   RMSE: ${test_metrics['RMSE']:,.0f}")
    print(f"   MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"   Directional Accuracy: {test_metrics['Directional_Accuracy']:.1f}%")

    # Sample predictions
    print(f"\n🎯 Sample Predictions:")
    print("-" * 35)
    n_samples = min(5, len(y_test_orig))
    for i in range(n_samples):
        actual = y_test_orig[i]
        predicted = y_pred_test_orig[i]
        error_pct = abs(actual - predicted) / actual * 100
        print(f"Sample {i+1}: Actual=${actual:>10,.0f} | Predicted=${predicted:>10,.0f} | Error={error_pct:5.1f}%")

    # Feature importance for tree-based models
    if hasattr(model, "feature_importances_"):
        print(f"\n🔍 Top 10 Most Important Features:")
        print("-" * 35)
        feature_names = [f"week_{i+1}" for i in range(cfg.dataset.lookback_window)]
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:10s}: {row['importance']:.4f}")

    print(f"\n✅ Training Complete!")
    print(f"🎯 Best metric: {test_metrics['MAPE']:.2f}% MAPE")
    print(f"💡 Custom Walmart dataset with {cfg.model._target_.split('.')[-1]} model")

    # Return metrics for Hydra multirun
    return test_metrics["MAPE"]


if __name__ == "__main__":
    train()
