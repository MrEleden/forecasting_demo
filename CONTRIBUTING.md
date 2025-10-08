# Contributing to the Forecasting Demo

Thank you for investing time in improving this portfolio. The project combines reusable forecasting components with domain-specific demos, so contributions should keep both layers in sync.

## 1. Prerequisites

1. Fork the repository and clone your fork.
1. Create a virtual environment and install the project in editable mode:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate
   pip install -r requirements-dev.txt
   pip install -e .
   ```
1. Download baseline datasets as needed:
   ```powershell
   python src/ml_portfolio/scripts/download_all_data.py --dataset all
   ```

## 2. Development workflow

1. Create a descriptive branch name (`feature/benchmark-cli`, `fix/lightgbm-docs`, etc.).
1. Make focused changes and keep commits scoped. Use present-tense, imperative commit messages ("Add LightGBM benchmark example").
1. Run formatting and tests before pushing:
   ```powershell
   pre-commit run --all-files
   pytest tests/ -v
   ```
1. Update or add documentation alongside code changes. The [`docs/STYLE_GUIDE.md`](docs/STYLE_GUIDE.md) outlines formatting expectations and requires YAML front matter for new Markdown files.

## 3. Pull request checklist

- [ ] Linked or described the issue the PR addresses.
- [ ] Added unit/integration tests when changing behaviour or fixing bugs.
- [ ] Updated relevant documentation (tutorials, guides, benchmark docs, etc.).
- [ ] Built documentation with `sphinx-build docs docs/_build/html` (ensure it finishes without errors).
- [ ] Verified benchmark scripts or dashboards if changes affect them.
- [ ] Confirmed `git status` shows no unintended file deletions.

## 4. Documentation contributions

High-quality docs are as important as code:

- Tutorials live under `docs/tutorials/` and should walk through end-to-end workflows.
- Reference content belongs in `docs/api_reference/` and should match the actual interfaces.
- Use `docs/guides/` for playbooks (data ingestion, model promotion, experiment tracking, testing strategy).
- Mention new docs in the relevant README or linking pages when appropriate.

## 5. Reporting issues

When filing issues, include:

- Environment details (OS, Python version, GPU availability).
- Exact command that reproduces the problem.
- Stack trace or screenshots for dashboard issues.
- Expected vs. actual behaviour.

## 6. Communication

We follow the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Be respectful and collaborative in discussions, reviews, and issue threads.

Happy forecasting! 📈
