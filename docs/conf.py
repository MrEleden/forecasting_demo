"""Sphinx configuration for the forecasting portfolio documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Add project root so autodoc can locate the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "Forecasting Demo"
author = "Forecasting Demo Contributors"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "alabaster"
html_static_path = ["_static"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]
myst_heading_anchors = 3
myst_front_matter_enable = True

highlight_language = "python"
pygments_style = "sphinx"
