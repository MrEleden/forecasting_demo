---
title: Model Selection Playbook
description: Step-by-step guidance for choosing the right forecasting model based on data constraints and business objectives.
---

# Model Selection Playbook

Choosing the correct forecasting approach depends on data volume, feature richness, required interpretability, and latency constraints. This playbook distils best practices gathered from the portfolio and top-tier ML libraries.

## 1. Decision flow

1. **How much history do you have?**
   - \< 500 observations → start with statistical baselines.
   - 500–10,000 observations → gradient boosting or hybrid models.
   - > 10,000 observations → consider deep learning or ensembles.
1. **Need explanatory features?**
   - Mostly seasonal patterns → statistical models (SARIMAX, Prophet).
   - Rich covariates → tree-based boosting (LightGBM, CatBoost, XGBoost).
   - Long-term dependencies → sequence models (LSTM, TCN, Transformer).
1. **Is interpretability critical?**
   - High → statistical / tree-based models with feature importance.
   - Medium → monotonic gradient boosting with SHAP explanations.
   - Low → neural nets with attention or latent representations.
1. **Latency and deployment profile?**
   - Real-time (\<50 ms) → LightGBM, CatBoost, or cached statistical models.
   - Batch predictions → any architecture; prioritise accuracy and stability.

## 2. Comparison matrix

| Model                 | Data Size       | Feature Support                  | Training Cost | Interpretability | When to prefer                                        |
| --------------------- | --------------- | -------------------------------- | ------------- | ---------------- | ----------------------------------------------------- |
| SARIMAXForecaster     | Tiny → Medium   | Date-derived only                | Very low      | High             | Strong seasonality, limited covariates                |
| ProphetForecaster     | Tiny → Medium   | Date-derived + holidays          | Low           | Medium           | Marketing calendars, irregular seasonality            |
| LightGBMForecaster    | Small → Large   | Tabular, categorical, continuous | Low           | Medium (SHAP)    | General-purpose baseline, fast iteration              |
| CatBoostForecaster    | Small → Large   | Strong categorical support       | Medium        | Medium (SHAP)    | High-cardinality categorical features                 |
| XGBoostForecaster     | Small → Large   | Numeric-heavy datasets           | Medium        | Medium           | Sparse signals, custom loss functions                 |
| TCNForecaster         | Medium → Huge   | Static + dynamic covariates      | High          | Low              | Fast convolutions, medium-length sequences            |
| LSTMForecaster        | Medium → Huge   | Full covariate support           | High          | Low              | Long-range dependencies, uneven intervals             |
| TransformerForecaster | Large → Massive | Full covariate support           | Very high     | Low              | Multi-horizon forecasting, attention insights         |
| EnsembleForecaster    | Any             | Aggregated                       | High          | Medium           | When mixing statistical + ML models boosts robustness |

## 3. Data readiness checklist

Before committing to a model, ensure:

- **Data integrity**: Missing timestamps are imputed or flagged; duplicates resolved.
- **Feature engineering**: Employ the `StaticTimeSeriesPreprocessingPipeline` for deterministic features; consider domain-specific signals (promotions, events).
- **Target scaling**: Use log or Box-Cox transforms for skewed targets when supported by the model.
- **Temporal validation**: Adopt time-based splits with a holdout horizon (see [DatasetFactory](../../src/ml_portfolio/data/dataset_factory.py)).

## 4. Experiment design tips

- Start with a **statistical baseline** to quantify uplift from complex models.
- Run **Optuna sweeps** on the top two candidates with consistent metric targets.
- When comparing families, log runtime, memory, and accuracy in MLflow for full context.
- Use the [benchmark suite](../BENCHMARK.md) to rerun comparisons after data refreshes.

## 5. Deployment guidance

| Scenario              | Recommendation                                                                   |
| --------------------- | -------------------------------------------------------------------------------- |
| Real-time scoring API | Lightweight gradient boosting exported with `joblib` or `onnx`.                  |
| Batch analytics       | Deep models or ensembles serialized to disk; orchestrate with scheduled scripts. |
| Dashboard drill-down  | Models with feature importance support (LightGBM, CatBoost, SARIMAX).            |
| Edge deployment       | Compact statistical models or distilled gradient boosting variants.              |

## 6. Next steps

- Use the [experiment tracking playbook](experiment_tracking.md) to record decisions and metrics.
- Promote the winning model through the [model promotion guide](model_promotion.md).
- Document learnings in project-specific READMEs for future collaborators.
