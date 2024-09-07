# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os
from pathlib import Path
HERE = Path(__file__).parent
PARENT_PARENT_HERE = HERE.parents[1]
SRC_PATH = PARENT_PARENT_HERE / 'src'
sys.path.insert(0, str(SRC_PATH))
for x in os.walk(str(SRC_PATH)):
  sys.path.insert(0, x[0])
print(sys.path)
try:
    import smftools
    print("smftools imported successfully.")
except ImportError:
    print("smftools is not imported.")
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'smftools'
copyright = '2024, Joseph McKenna'
author = 'Joseph McKenna'
release = '0.1.0'
repository_url = 'https://github.com/jkmckenna/smftools'

# -- General configuration ---------------------------------------------------
# Bibliography settings
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
nitpicky = True
needs_sphinx = "4.0"

master_doc = "index"
templates_path = ['_templates']
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # exclude version md files
    "release-notes/[!i]*.md"
]
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_search.extension",
    "sphinxext.opengraph",
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
api_dir = HERE / "api" 
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto", "ftp")
myst_heading_anchors = 3
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True

suppress_warnings = [
    "myst.header"
]

typehints_defaults = "braces"

# html_context = {
#     "display_github": True,
#     "github_user": "jkmckenna",
#     "github_repo": project,
#     "github_version": "main",
#     "conf_py_path": "/docs/source/",
# }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "show_toc_level": 1,
    "path_to_docs": "docs/",
    "repository_branch": release,
}

html_static_path = ['_static']