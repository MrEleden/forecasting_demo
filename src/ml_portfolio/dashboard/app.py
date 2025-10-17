"""ML Forecasting Portfolio Dashboard.

Executive storyline:
- Problem: Consistent, reproducible time-series forecasting across domains.
- Approach: A shared library (src/ml_portfolio) powering self-contained demos
  (projects/*) with Hydra configs, MLflow tracking, and Optuna search.
- Outcome: Reusable data/feature/model/training primitives, audited experiments,
  and deployable stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yaml

# Optional dependencies
try:
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = Any  # type: ignore[assignment]
    ViewType = None
    MLFLOW_AVAILABLE = False

try:
    import altair as alt

    ALTAIR_AVAILABLE = True
except ImportError:
    alt = None  # type: ignore[assignment]
    ALTAIR_AVAILABLE = False

# Path configuration
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src" / "ml_portfolio"
CONF_DIR = SRC_DIR / "conf"
MODEL_CONF_DIR = CONF_DIR / "model"
PROJECTS_DIR = REPO_ROOT / "projects"
MLFLOW_DIR = REPO_ROOT / "mlruns"

# Model metadata for narrative
MODEL_METADATA = {
    "lightgbm": {
        "family": "Gradient Boosting",
        "label": "LightGBM",
        "when_to_use": ("High-dimensional tabular data with complex feature interactions"),
        "strengths": [
            "Fast training on large datasets",
            "Handles categorical features natively",
            "Strong performance on structured time series",
            "Low memory footprint",
        ],
        "tradeoffs": [
            "Less interpretable than linear models",
            "Requires careful tuning to avoid overfitting",
            "Limited extrapolation beyond training range",
        ],
        "data_regime": "10k+ samples, mixed types, feature-rich",
    },
    "xgboost": {
        "family": "Gradient Boosting",
        "label": "XGBoost",
        "when_to_use": "Robust baseline for medium to large tabular datasets",
        "strengths": [
            "Battle-tested with strong community support",
            "Customizable objectives and evaluation metrics",
            "Good feature importance diagnostics",
            "Handles missing values well",
        ],
        "tradeoffs": [
            "Slower than LightGBM on very large data",
            "Sensitive to hyperparameter choices",
            "May overfit without regularization",
        ],
        "data_regime": "5k+ samples, structured features",
    },
    "catboost": {
        "family": "Gradient Boosting",
        "label": "CatBoost",
        "when_to_use": ("Data dominated by categorical features or minimal preprocessing time"),
        "strengths": [
            "Superior handling of categorical variables",
            "Strong default hyperparameters",
            "Built-in ordered target encoding",
            "Robust to missing data",
        ],
        "tradeoffs": [
            "Larger model size than competitors",
            "Slower inference than LightGBM",
            "Smaller ecosystem than XGBoost/LightGBM",
        ],
        "data_regime": "Any size with many categoricals",
    },
    "random_forest": {
        "family": "Ensemble Trees",
        "label": "Random Forest",
        "when_to_use": ("Interpretable baseline or when feature interactions matter " "more than temporal patterns"),
        "strengths": [
            "Simple to train and tune",
            "Good permutation-based feature importance",
            "Robust to outliers and noise",
            "Parallelizable",
        ],
        "tradeoffs": [
            "Cannot extrapolate trends",
            "Weaker on pure time-series vs GBMs",
            "Larger model size",
        ],
        "data_regime": "1k+ samples, sanity-check baseline",
    },
    "lstm": {
        "family": "Deep Learning (RNN)",
        "label": "LSTM",
        "when_to_use": "Long-term temporal dependencies with multivariate inputs",
        "strengths": [
            "Captures sequential patterns and memory",
            "Flexible input/output horizons",
            "Integrates external covariates naturally",
            "GPU acceleration available",
        ],
        "tradeoffs": [
            "Requires larger datasets (10k+ sequences)",
            "Slower to train and tune",
            "Prone to overfitting without regularization",
            "Less interpretable",
        ],
        "data_regime": "10k+ sequences, rich temporal structure",
    },
    "tcn": {
        "family": "Deep Learning (CNN)",
        "label": "Temporal Convolutional Network",
        "when_to_use": "Medium to long sequences with parallel training needs",
        "strengths": [
            "Stable gradients vs RNNs",
            "Parallel training (faster than LSTM)",
            "Captures local and medium-term patterns",
            "Flexible receptive field via dilations",
        ],
        "tradeoffs": [
            "Requires tuning dilation schedule",
            "Still data-hungry like other DL models",
            "Less mature ecosystem than LSTM",
        ],
        "data_regime": "5k+ sequences, moderate temporal depth",
    },
    "transformer": {
        "family": "Deep Learning (Attention)",
        "label": "Transformer",
        "when_to_use": ("Very long sequences or multi-horizon forecasting with " "cross-series learning"),
        "strengths": [
            "Captures global dependencies via attention",
            "Supports multi-horizon and probabilistic outputs",
            "Can learn from multiple time series jointly",
            "State-of-the-art on long sequences",
        ],
        "tradeoffs": [
            "Highest computational cost",
            "Requires large datasets or pretraining",
            "Complex hyperparameter space",
            "Slower inference",
        ],
        "data_regime": "50k+ samples or pretrained foundation models",
    },
}


def _get_mlflow_client() -> Optional[Any]:
    """Return MLflow tracking client if available."""
    if not MLFLOW_AVAILABLE or not MLFLOW_DIR.exists():
        return None
    try:
        tracking_uri = MLFLOW_DIR.resolve().as_uri()
        return MlflowClient(tracking_uri=tracking_uri)
    except Exception:
        return None


def _list_experiments() -> list:
    """List active MLflow experiments."""
    client = _get_mlflow_client()
    if client is None or ViewType is None:
        return []
    try:
        if hasattr(client, "search_experiments"):
            return client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        return client.list_experiments(view_type=ViewType.ACTIVE_ONLY)
    except Exception:
        return []


def _get_runs(experiment_id: str, limit: int = 100) -> pd.DataFrame:
    """Retrieve MLflow runs for an experiment."""
    client = _get_mlflow_client()
    if client is None or ViewType is None:
        return pd.DataFrame()

    try:
        runs = client.search_runs(
            [experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
    except Exception:
        return pd.DataFrame()

    records = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        records.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or run.info.run_id[:8],
                "status": run.info.status,
                "model": tags.get("model_type", params.get("model", "unknown")),
                "dataset": tags.get("dataset", params.get("dataset_name", "unknown")),
                "val_MAPE": metrics.get("val_MAPE", metrics.get("val_MAPEMetric")),
                "val_RMSE": metrics.get("val_RMSE", metrics.get("val_RMSEMetric")),
                "val_MAE": metrics.get("val_MAE", metrics.get("val_MAEMetric")),
                "test_MAPE": metrics.get("test_MAPE", metrics.get("test_MAPEMetric")),
                "test_RMSE": metrics.get("test_RMSE", metrics.get("test_RMSEMetric")),
                "test_MAE": metrics.get("test_MAE", metrics.get("test_MAEMetric")),
                "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in [
        "val_MAPE",
        "val_RMSE",
        "val_MAE",
        "test_MAPE",
        "test_RMSE",
        "test_MAE",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _load_model_config(model_name: str) -> Dict[str, Any]:
    """Load Hydra config for a model."""
    config_path = MODEL_CONF_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def render_overview_tab():
    """Overview: What the portfolio is and how it's organized."""
    st.header("ML Forecasting Portfolio")
    st.markdown(
        """
        ### Executive Storyline

        **Problem**: Consistent, reproducible time-series forecasting across domains.

        **Approach**: A shared library (`src/ml_portfolio`) powering self-contained
        demos (`projects/*`) with Hydra configs, MLflow tracking, and Optuna search.

        **Outcome**: Reusable data/feature/model/training primitives, audited
        experiments, and deployable stubs - demonstrated on Walmart weekly sales.
        """
    )

    st.markdown("---")
    st.subheader("Architecture at a glance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Shared Library** (`src/ml_portfolio/`)
            - `data/`: DatasetFactory, loaders, preprocessing pipelines
            - `models/`: Statistical (GBMs), deep learning (LSTM, TCN), base classes
            - `training/`: Engines (Statistical, PyTorch), callbacks, training loop
            - `evaluation/`: Metrics (MAPE, RMSE, MAE), plotting utilities
            - `conf/`: Hydra configs for models, datasets, features, engines
            - `utils/`: I/O, logging, config helpers
            """
        )

    with col2:
        st.markdown(
            """
            **Demo Projects** (`projects/*/`)
            - Self-contained demonstrations with own data and notebooks
            - `data/`: raw, interim, processed splits
            - `models/`: trained artifacts and checkpoints
            - `notebooks/`: EDA and training flows
            - `scripts/`: data download/generation
            - Inherit from shared library; override as needed
            """
        )

    st.markdown("---")
    st.subheader("Quick start")

    st.code(
        """
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a model with Hydra
python src/ml_portfolio/training/train.py model=lightgbm dataset_name=walmart

# 3. Run hyperparameter optimization
python src/ml_portfolio/scripts/run_optimization.py --models lightgbm --trials 50

# 4. View experiments
mlflow ui

# 5. Launch this dashboard
streamlit run src/ml_portfolio/dashboard/app.py
        """,
        language="bash",
    )

    st.info(
        "See `docs/` for detailed guides on experiment tracking, benchmarking, " "and adding new models or projects."
    )


def render_model_library_tab():
    """Model library: Browse families and configs."""
    st.header("Model Library")
    st.markdown(
        """
        Browse model families, understand their strengths and trade-offs, and
        view Hydra configurations.
        """
    )

    families = {}
    for model_key, meta in MODEL_METADATA.items():
        family = meta["family"]
        if family not in families:
            families[family] = []
        families[family].append((model_key, meta))

    selected_family = st.selectbox("Select model family", sorted(families.keys()))

    st.markdown("---")
    for model_key, meta in families[selected_family]:
        with st.expander(f"{meta['label']} (`{model_key}`)"):
            st.markdown(f"**When to use**: {meta['when_to_use']}")
            st.markdown(f"**Data regime**: {meta['data_regime']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Strengths**")
                for strength in meta["strengths"]:
                    st.write(f"- {strength}")

            with col2:
                st.markdown("**Trade-offs**")
                for tradeoff in meta["tradeoffs"]:
                    st.write(f"- {tradeoff}")

            st.markdown("**Hydra config snapshot**")
            config = _load_model_config(model_key)
            if config:
                st.json(config, expanded=False)
            else:
                st.warning(f"Config not found: `{MODEL_CONF_DIR / model_key}.yaml`")


def render_experiments_tab():
    """Experiments: Select experiment, inspect runs."""
    st.header("Experiments")
    st.markdown("Select an MLflow experiment to inspect runs, metrics, and artifacts.")

    if not MLFLOW_AVAILABLE:
        st.error("MLflow is not installed. Install it to view experiments.")
        return

    experiments = _list_experiments()
    if not experiments:
        st.info("No experiments found. Train a model with `use_mlflow=true` to populate.")
        return

    exp_names = {exp.name: exp.experiment_id for exp in experiments}
    selected_name = st.selectbox("Select experiment", sorted(exp_names.keys()))
    selected_id = exp_names[selected_name]

    st.markdown(f"**Experiment ID**: `{selected_id}`")

    runs = _get_runs(selected_id, limit=200)
    if runs.empty:
        st.info("No runs found for this experiment.")
        return

    st.markdown("---")
    st.subheader("Runs overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total runs", len(runs))
    col2.metric("Models", runs["model"].nunique())
    col3.metric("Datasets", runs["dataset"].nunique())

    st.markdown("---")
    st.subheader("Run table")

    # Warning about optimization runs
    st.warning(
        "⚠️ **Note**: Hyperparameter optimization saves ALL trial runs to MLflow. "
        "This can result in hundreds of runs per optimization session. "
        "The table below shows all runs including intermediate optimization trials."
    )

    # Filters
    col_model, col_status = st.columns(2)
    models = sorted(runs["model"].dropna().unique())
    statuses = sorted(runs["status"].dropna().unique())

    selected_models = col_model.multiselect("Filter by model", models, default=models)
    selected_statuses = col_status.multiselect("Filter by status", statuses, default=statuses)

    filtered = runs[runs["model"].isin(selected_models) & runs["status"].isin(selected_statuses)]

    if filtered.empty:
        st.warning("No runs match filters.")
        return

    display_cols = [
        "run_name",
        "model",
        "dataset",
        "status",
        "val_MAPE",
        "val_RMSE",
        "test_MAPE",
        "test_RMSE",
        "start_time",
    ]
    st.dataframe(
        filtered[display_cols].style.format(
            {
                "val_MAPE": "{:.2f}%",  # Already a percentage (not decimal), just add % symbol
                "val_RMSE": "{:.2f}",
                "test_MAPE": "{:.2f}%",  # Already a percentage (not decimal), just add % symbol
                "test_RMSE": "{:.2f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
    )

    if ALTAIR_AVAILABLE and not filtered.dropna(subset=["val_MAPE"]).empty:
        st.markdown("---")
        st.subheader("Validation MAPE over time")
        chart_df = filtered.dropna(subset=["val_MAPE", "start_time"])
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("start_time:T", title="Run start time"),
                y=alt.Y("val_MAPE:Q", title="Validation MAPE"),
                color=alt.Color("model:N", title="Model"),
                tooltip=[
                    "run_name",
                    "model",
                    alt.Tooltip("val_MAPE:Q", format=".2%"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)


def render_benchmarks_tab():
    """Benchmarks: Leaderboard and trade-offs."""
    st.header("Benchmarks & Trade-offs")
    st.markdown("View best runs per model and understand selection criteria.")

    if not MLFLOW_AVAILABLE:
        st.error("MLflow is not installed. Install it to view benchmarks.")
        return

    experiments = _list_experiments()
    if not experiments:
        st.info("No experiments found.")
        return

    exp_names = {exp.name: exp.experiment_id for exp in experiments}
    selected_name = st.selectbox("Select experiment", sorted(exp_names.keys()), key="bench_exp")
    selected_id = exp_names[selected_name]

    runs = _get_runs(selected_id, limit=200)
    if runs.empty:
        st.info("No runs found.")
        return

    st.markdown("---")
    st.subheader("Leaderboard (sorted by test MAPE)")

    # Filter for runs with test MAPE, fallback to validation MAPE
    leaderboard = runs.dropna(subset=["test_MAPE"]).sort_values("test_MAPE").drop_duplicates("model", keep="first")

    if leaderboard.empty:
        st.warning("⚠️ No runs with test metrics found. This usually means:")
        st.markdown(
            """
            - Training runs are still in progress (test evaluation happens at the end)
            - Optimization trials were interrupted before final evaluation
            - Test set evaluation was skipped in training config

            Showing runs sorted by validation MAPE instead:
            """
        )
        leaderboard = runs.dropna(subset=["val_MAPE"]).sort_values("val_MAPE").drop_duplicates("model", keep="first")
        if leaderboard.empty:
            st.error("No runs with validation MAPE either.")
            return

    st.dataframe(
        leaderboard[["model", "run_name", "val_MAPE", "val_RMSE", "test_MAPE", "test_RMSE"]].style.format(
            {
                "val_MAPE": "{:.2f}%",  # Already a percentage (not decimal), just add % symbol
                "val_RMSE": "{:.2f}",
                "test_MAPE": "{:.2f}%",  # Already a percentage (not decimal), just add % symbol
                "test_RMSE": "{:.2f}",
            },
            na_rep="None",
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Model selection guidance")

    guidance_shown = False
    for _, row in leaderboard.iterrows():
        model_key = row["model"]
        # Normalize model key to lowercase with underscores for lookup
        # RandomForest -> random_forest, LightGBM -> lightgbm, etc.
        model_key_normalized = model_key.lower()
        if "forest" in model_key_normalized and "_" not in model_key_normalized:
            model_key_normalized = "random_forest"
        meta = MODEL_METADATA.get(model_key_normalized, {})
        if not meta:
            # Show basic info even without metadata
            with st.expander(f"{model_key} — Best run: {row['run_name']}"):
                st.markdown(
                    f"**Validation MAPE**: {row.get('val_MAPE', 'N/A'):.2%}"
                    if pd.notna(row.get("val_MAPE"))
                    else "**Validation MAPE**: N/A"
                )
                st.markdown(
                    f"**Test MAPE**: {row.get('test_MAPE', 'N/A'):.2%}"
                    if pd.notna(row.get("test_MAPE"))
                    else "**Test MAPE**: N/A"
                )
                st.info(f"No detailed metadata available for model: {model_key}")
            guidance_shown = True
            continue

        with st.expander(f"{meta.get('label', model_key)} — Best run: {row['run_name']}"):
            if pd.notna(row.get("val_MAPE")):
                st.markdown(f"**Validation MAPE**: {row['val_MAPE']:.2f}%")
            if pd.notna(row.get("test_MAPE")):
                st.markdown(f"**Test MAPE**: {row['test_MAPE']:.2f}%")
            st.markdown(f"**When to use**: {meta.get('when_to_use', 'N/A')}")
            st.markdown(f"**Data regime**: {meta.get('data_regime', 'N/A')}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Strengths**")
                for s in meta.get("strengths", []):
                    st.write(f"- {s}")
            with col2:
                st.markdown("**Trade-offs**")
                for t in meta.get("tradeoffs", []):
                    st.write(f"- {t}")

            # Add hyperparameter analysis
            st.markdown("---")
            st.markdown("**🔍 Hyperparameter Analysis**")

            # Get all runs for this model
            model_runs = runs[runs["model"] == model_key].dropna(subset=["val_MAPE"]).copy()

            if len(model_runs) >= 3:
                st.markdown(f"Analyzed **{len(model_runs)}** runs for this model:")

                # Extract hyperparameters from MLflow
                client = _get_mlflow_client()
                if client:
                    try:
                        # Get parameters for best run
                        best_run_id = row["run_id"]
                        best_run_data = client.get_run(best_run_id)
                        best_params = best_run_data.data.params

                        # Filter to model-specific parameters
                        model_params = {k: v for k, v in best_params.items() if k.startswith("model.")}

                        if model_params:
                            st.markdown("**Best run parameters**:")
                            param_cols = st.columns(3)
                            for idx, (param_name, param_value) in enumerate(sorted(model_params.items())):
                                col_idx = idx % 3
                                param_cols[col_idx].metric(label=param_name.replace("model.", ""), value=param_value)
                        else:
                            st.info("No hyperparameters logged for this model.")

                    except Exception as e:
                        st.warning(f"Could not retrieve hyperparameter details: {e}")
                else:
                    st.info("MLflow client not available for parameter comparison.")
            else:
                st.info(
                    f"Only {len(model_runs)} run(s) found for this model. "
                    f"Need at least 3 runs for meaningful hyperparameter analysis."
                )

        guidance_shown = True

    if not guidance_shown:
        st.info("No model guidance available. Train some models to see recommendations.")


def render_engineering_pov_tab():
    """Engineering POV: Architecture, CI/CD, data quality."""
    st.header("Engineering Perspective")
    st.markdown("How the system is built, tested, and deployed.")

    st.subheader("Architecture & packaging")
    st.markdown(
        """
        - **Shared library** (`src/ml_portfolio/`): reusable data, models,
          training, evaluation.
        - **Demo projects** (`projects/*/`): self-contained with own data
          and notebooks.
        - **Clear module boundaries**: data loaders, feature pipelines,
          engines, metrics.
        - **Hydra configs**: reproducible experiments with structured overrides.
        """
    )

    st.markdown("---")
    st.subheader("Configuration & reproducibility")
    st.markdown(
        """
        - **Hydra**: structured configs for datasets, features, models,
          engines, metrics.
        - **Deterministic seeds**: `seed=42` for numpy, random, torch.
        - **Run artifacts**: `hydra.run.dir` captures configs, logs, outputs.
        - **Override system**: `model=lightgbm dataset_name=walmart optimizer=adam`.
        """
    )

    st.markdown("---")
    st.subheader("Data access & preprocessing")
    st.markdown(
        """
        - **DatasetFactory**: chronological train/val/test splits,
          store filtering.
        - **Static feature pipeline**: lags (1,2,4,8,13,26,52), rolling stats,
          cyclical encodings (month, week, day), scaling.
        - **Loaders**: `SimpleDataLoader` (statistical),
          `PyTorchDataLoader` (DL).
        - **Parquet caching**: speed up repeated runs.
        """
    )

    st.markdown("---")
    st.subheader("Training infrastructure")
    st.markdown(
        """
        - **StatisticalEngine**: sklearn-compatible, single-pass fit
          for GBMs/forests.
        - **PyTorchEngine**: multi-epoch training, gradient clipping,
          early stopping.
        - **Early stopping**: native for GBMs; patience-based for
          PyTorch models.
        - **Checkpoints**: best model (by val metric), final model
          after training.
        - **Logs**: train/val/test metrics, runtime, convergence.
        """
    )

    st.markdown("---")
    st.subheader("Experiment tracking & optimization")
    st.markdown(
        """
        - **MLflow**: logs params, metrics, artifacts, models to
          local `mlruns/`.
        - **Optuna**: hyperparameter search with Hydra integration
          (`run_optimization.py`).
        - **Benchmarks**: aggregated by primary metric (MAPE for Walmart).
        """
    )

    st.markdown("---")
    st.subheader("Quality, testing, CI/CD")
    st.markdown(
        """
        - **Pytest**: unit tests with coverage gate (~40% currently).
        - **Linting**: Ruff, Black, Mypy in CI matrix (Ubuntu, Windows).
        - **Security**: Bandit, Safety scans.
        - **Docs**: Sphinx build check, Codecov reports.
        - **CI matrix**: tests run on Python 3.9, 3.10, 3.11.
        """
    )

    st.markdown("---")
    st.subheader("Data quality & governance")
    st.markdown(
        """
        - **Great Expectations**: optional pre-training validation hook.
        - **Fail-open**: logs warnings if GE not installed; can enforce in CI.
        - **Next steps**: wire into CI with non-blocking alerts.
        """
    )

    st.markdown("---")
    st.subheader("Serving & interfaces")
    st.markdown(
        """
        - **FastAPI**: stubs for `/predict` (single/batch) and
          `/health` endpoints.
        - **Streamlit**: this dashboard showcases portfolio, experiments,
          benchmarks.
        - **Deployment**: Dockerfiles, docker-compose ready; monitoring TBD.
        """
    )


def render_data_science_pov_tab():
    """Data Science POV: Dataset, features, evaluation, insights."""
    st.header("Data Science Perspective")
    st.markdown("Understanding the data, features, models, and evaluation strategy.")

    st.subheader("Dataset & splitting")
    st.markdown(
        """
        - **Walmart weekly sales**: ~6.5k samples across 45 stores
          and 81 departments.
        - **Chronological split**: 70% train, 15% val, 15% test.
        - **No temporal leakage**: validation and test strictly follow
          training period.
        - **Store filtering**: can focus on specific stores for faster iteration.
        """
    )

    st.markdown("---")
    st.subheader("Feature engineering")
    st.markdown(
        """
        **Time-aware statics**:
        - Lags: 1, 2, 4, 8, 13, 26, 52 weeks (short-term + seasonal)
        - Rolling statistics: mean and std over 4, 8, 13, 26, 52 week windows
        - Cyclical encodings: month, week, dayofweek, quarter
        - Scaling: StandardScaler for features and target

        **Hooks for enrichment**:
        - Holidays (e.g., Thanksgiving, Christmas)
        - Promotions and markdowns
        - Weather, fuel prices (already in raw data)
        - Store/department hierarchies
        """
    )

    st.markdown("---")
    st.subheader("Model portfolio")
    st.markdown(
        """
        **Gradient Boosting** (LightGBM, XGBoost, CatBoost):
        - Strong tabular baselines, fast iteration
        - Less temporal extrapolation; best for feature-rich regimes

        **Deep Learning** (LSTM, TCN, Transformer):
        - Capture long-term dependencies, multivariate inputs
        - Require larger datasets (10k+ sequences) and careful tuning

        **Ensemble Trees** (Random Forest):
        - Interpretable baseline, sanity check
        - Cannot extrapolate trends; weaker on pure time-series

        **Statistical** (ARIMA, Prophet - scaffolded):
        - Fast, interpretable, good for univariate slices
        - Limited multivariate capacity
        """
    )

    st.markdown("---")
    st.subheader("Metrics & evaluation")
    st.markdown(
        """
        - **Primary metric**: MAPE (business-friendly, scale-free)
        - **Secondary**: RMSE (penalizes large errors),
          MAE (robust to outliers)
        - **Diagnostics**: feature importances (trees),
          forecast vs actuals plots
        - **Benchmark tab**: leaderboard sorted by validation MAPE
        """
    )

    st.markdown("---")
    st.subheader("Insights & trade-offs")
    st.markdown(
        """
        **Model selection by data regime**:
        - **< 1k samples**: statistical (ARIMA, Prophet) or simple baselines
        - **1k-10k samples**: GBMs (LightGBM, XGBoost, CatBoost) dominate
        - **10k+ samples**: consider LSTM/TCN if temporal depth matters
        - **50k+ samples**: Transformers become viable with pretraining

        **Current best practices**:
        - Start with LightGBM baseline (fast, reliable)
        - Add LSTM if long-term dependencies observed
        - Ensemble top-3 models for production robustness
        """
    )

    st.markdown("---")
    st.subheader("Known gaps & backlog")
    st.warning(
        """
        - Add WMAE (weighted MAE) as primary metric for Walmart hierarchy
        - Implement rolling-origin backtesting for realistic evaluation
        - Incorporate holiday/promo features (data available, not yet engineered)
        - Hierarchical reconciliation across Store → Department
        - Scenario simulations (price elasticity, supply shocks)
        """
    )


def main():
    """Main entry point for the dashboard."""
    st.set_page_config(
        page_title="ML Forecasting Portfolio",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("ML Forecasting Portfolio")
    st.sidebar.markdown("**Navigate the story**")

    tab = st.sidebar.radio(
        "Select view",
        [
            "Overview",
            "Model Library",
            "Experiments",
            "Benchmarks & Trade-offs",
            "Engineering POV",
            "Data Science POV",
        ],
    )

    if tab == "Overview":
        render_overview_tab()
    elif tab == "Model Library":
        render_model_library_tab()
    elif tab == "Experiments":
        render_experiments_tab()
    elif tab == "Benchmarks & Trade-offs":
        render_benchmarks_tab()
    elif tab == "Engineering POV":
        render_engineering_pov_tab()
    elif tab == "Data Science POV":
        render_data_science_pov_tab()

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **For leaders**: Replicable, tracked process from data to model to benchmark.

        **For engineers**: Configurable, testable components with CI and deployment hooks.

        **For scientists**: Credible features, fair splits, transparent metrics,
        and comparable baselines.
        """
    )


if __name__ == "__main__":
    main()
