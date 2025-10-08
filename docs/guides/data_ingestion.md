---
title: Data Ingestion Playbook
description: Practical guidance for sourcing, validating, and staging datasets across the ML forecasting portfolio.
---

# Data Ingestion Playbook

This guide standardises how raw data enters the portfolio so every project can reuse loaders, pipelines, and benchmarks without rework.

## 1. Sources and orchestration

| Project               | Script                                                              | Command                                                                    |
| --------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Walmart retail sales  | `projects/retail_sales_walmart/scripts/download_data.py`            | `python projects/retail_sales_walmart/scripts/download_data.py`            |
| Ola rideshare demand  | `src/ml_portfolio/scripts/download_all_data.py --dataset ola`       | `python src/ml_portfolio/scripts/download_all_data.py --dataset ola`       |
| Inventory forecasting | `src/ml_portfolio/scripts/download_all_data.py --dataset inventory` | `python src/ml_portfolio/scripts/download_all_data.py --dataset inventory` |
| Transportation TSI    | `src/ml_portfolio/scripts/download_all_data.py --dataset tsi`       | `python src/ml_portfolio/scripts/download_all_data.py --dataset tsi`       |

Best practice is to run the orchestrator once per environment:

```powershell
python src/ml_portfolio/scripts/download_all_data.py --dataset all
```

This fans out to project-level scripts while keeping credentials and bespoke logic near each domain.

## 2. Directory conventions

Each project follows the same data layout:

```
projects/{project_name}/data/
├── raw/        # Unmodified downloads
├── interim/    # Temporary processing artefacts
├── processed/  # Feature-ready parquet or CSV tables
└── external/   # Third-party enrichment files
```

- **Raw** directories are immutable—treat them as append-only.
- Use **interim** for expensive feature engineering steps you may wish to cache.
- Ship models and benchmarks from the **processed** tier only.

## 3. Validation and schema checks

- Run Pandera validation utilities inside `ml_portfolio.data.validation` (e.g., `validate_walmart_data`) immediately after download.
- Fail fast if critical columns are missing (`Weekly_Sales`, `Date`, `Store`, etc.).
- Store validation reports alongside processed artefacts (`projects/{project}/reports/`).

## 4. Feature engineering guardrails

- Apply deterministic, backward-looking transforms through `StaticTimeSeriesPreprocessingPipeline`.
- Keep stochastic or learned preprocessing in the statistical pipeline (`StatisticalPreprocessingPipeline`) that is fitted on the training split only.
- Record feature settings (lags, rolling windows, encodings) in configuration files so benchmarks and dashboard components can reproduce them.

## 5. Reproducibility checklist

1. Capture download commands (and any credentials requirements) in the project README.
1. Hash raw files or store checksums to detect drift.
1. Version processed datasets by date or git commit if you regenerate features.
1. Keep notebook explorations in `projects/{project}/notebooks/` and export final transformations to scripts or pipelines.

Adhering to this flow ensures every dataset enters the portfolio in a predictable, automation-friendly shape.
