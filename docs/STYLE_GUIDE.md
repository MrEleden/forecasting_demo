---
title: Documentation Style Guide
description: Conventions for writing and validating Markdown content in the ML forecasting portfolio.
---

# Documentation Style Guide

Consistent documentation keeps the portfolio approachable and ready for static site generation. Follow these rules whenever you touch Markdown files.

## Front matter and headings

- Every `.md` file must begin with YAML front matter containing at minimum `title` and `description`.
- Start content with a single `#` heading that mirrors the `title`.
- Use `##` and `###` for subsequent sections—avoid skipping levels.

## Formatting conventions

- Favour short paragraphs and bullet lists for readability.
- Label code fences with the correct language (`powershell`, `bash`, `python`, `yaml`).
- Use KaTeX notation for mathematics: inline `$x^2$`, block `$$ x^2 $$`.
- Keep lines UTF-8/ASCII compatible; emoji are welcome in Markdown but avoid them in non-Markdown artefacts.
- Reference workspace files with backticks (e.g., `` `docs/BENCHMARK.md` ``).

## Cross-linking

- Link to related docs when relevant (e.g., from tutorials to benchmarks or dashboard guides).
- When describing scripts or configs, provide relative paths under `src/` or `projects/`.

## Change control

- Run `pre-commit run --all-files` before pushing to validate Markdown formatting and lint checks.
- Update tables of contents manually only when the structure stabilises; automated TOC tooling is currently optional.

## Voice and tone

- Write in the second person (“you”) for tutorials and how-tos; use neutral tone for references.
- Be explicit about prerequisites and expected results.
- Prefer actionable steps over lengthy theory—link to external resources when deeper context is required.

Adhering to these guidelines ensures the documentation set stays cohesive as the portfolio grows.
