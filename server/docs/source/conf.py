# pylint: disable=all
# mypy: ignore-errors
import os, sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Mila - Demand Response"
copyright = "2023, Polytechnique Montréal"
author = "Polytechnique Montréal"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinxcontrib.autodoc_pydantic"]
sys.path.insert(0, os.path.abspath("../../.."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../app/**"))
sys.path.insert(0, os.path.abspath("../v0/**"))
sys.path.insert(0, os.path.abspath("./app/**"))
sys.path.insert(0, os.path.abspath("./v0/**"))
sys.path.insert(0, os.path.abspath("."))
autodoc_typehints = "both"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "insipid"
html_static_path = ["_static"]
