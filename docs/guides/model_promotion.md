---
title: Model Promotion Checklist
description: Steps for promoting a trained forecasting model from experimentation to production within the portfolio.
---

# Model Promotion Checklist

Use this checklist when a model graduates from experimentation to production. It complements the benchmark suite and dashboard workflows.

## 1. Gather evidence

- ✅ **Benchmark results**: Produce `benchmark_results.json` and `benchmark_report.txt` for the candidate using `ModelBenchmark`.
- ✅ **MLflow tracking**: Log parameter sets, metrics, and artefacts (`mlruns/`) for reproducibility.
- ✅ **Notebooks → scripts**: Convert exploratory notebooks into scripted pipelines stored under `src/ml_portfolio` or project-specific `scripts/` folders.

## 2. Package the artefacts

- Save the trained estimator with `joblib.dump(model, "models/{name}_v{version}.pkl")`.
- Persist preprocessing pipelines (static + statistical) alongside the model.
- Record metadata (training window, metrics, feature set) in a JSON manifest stored in `models/metadata/{name}_v{version}.json`.

## 3. Register and document

- Update the project README with the promoted model, including performance metrics and data coverage.
- If using the central registry (`ml_portfolio.models.registry`), add an entry mapping the registry name to the saved artefacts and version.
- Document inference requirements (feature order, scaling steps, expected cadence) in `docs/` or project `reports/`.

## 4. Wire into serving surfaces

- **API**: Expose the model through the FastAPI blueprint under `projects/{project}/api/`. Ensure dependency injection uses the registry or explicit file paths.
- **Dashboard**: Point the Streamlit app to the latest benchmark JSON or MLflow experiment so stakeholders see the promoted model immediately.
- **Batch pipelines**: Update scheduling scripts or notebooks that rely on the previous model version.

## 5. Quality gates before release

1. Run the full test suite: `pytest tests/ -v`.
1. Execute linting hooks: `pre-commit run --all-files` (now including Markdown checks; see `docs/STYLE_GUIDE.md`).
1. Re-run the benchmark to confirm the promoted model still wins against baselines.
1. If applicable, run backtests or hold-out evaluations covering edge cases (holiday periods, cold starts).

## 6. Post-deployment monitoring hooks

- Log live metrics back into MLflow or a monitoring store.
- Set alert thresholds for drift in directional accuracy or MAPE.
- Schedule periodic re-benchmarking (weekly or monthly) to catch regressions early.

Following this checklist keeps promotions auditable and lets downstream consumers know exactly which artefacts and metrics back each release.
