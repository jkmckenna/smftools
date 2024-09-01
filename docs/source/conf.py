# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'smftools'
copyright = '2024, Joseph McKenna'
author = 'Joseph McKenna'
release = '0.1.0'
repository_url = 'https://github.com/jkmckenna/smftools'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_nb"]

templates_path = ['_templates']
exclude_patterns = []

html_context = {
    "display_github": True,
    "github_user": "jkmckenna",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

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

