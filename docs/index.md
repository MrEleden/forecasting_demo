---
title: Forecasting Portfolio Documentation
description: Central entry point for tutorials, guides, and API reference material.
---

# Forecasting Portfolio Documentation

Welcome! This site consolidates tutorials, guides, and references for the forecasting portfolio. The content is written in Markdown and rendered with Sphinx + MyST, so you can keep documentation close to the codebase while still generating a polished documentation site.

## Navigation

```{toctree}
:maxdepth: 1
:caption: Getting Started

SETUP
getting_started/ten_minute_tour
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/01_first_forecast
```

```{toctree}
:maxdepth: 2
:caption: How-to Guides

guides/data_ingestion
guides/model_selection
guides/experiment_tracking
guides/testing_strategy
guides/model_promotion
guides/troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Reference

BENCHMARK
api_reference/index
STYLE_GUIDE
```

```{toctree}
:maxdepth: 1
:caption: Developer Reference

developer/architecture
developer/adding_model
developer/testing_best_practices
```

## Build locally

Generate the static site using the Sphinx CLI:

```powershell
.\.venv\Scripts\sphinx-build docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser to preview the site.
