# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
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
nitpicky = False
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
    "sphinx_click",
    "sphinx_design",
    "sphinx_search.extension",
    "sphinxext.opengraph",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_preprocess_types = True
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

smartquotes = False

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


def _generate_schema_tables() -> None:
    try:
        import yaml
    except ModuleNotFoundError:
        print("PyYAML not installed; skipping schema table generation.")
        return

    try:
        from smftools.schema import get_schema_registry_path
    except Exception as exc:
        print(f"Unable to import smftools schema registry: {exc}")
        return

    registry_path = get_schema_registry_path()
    if not registry_path.exists():
        print(f"Schema registry not found at {registry_path}; skipping.")
        return

    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    output_path = HERE / "schema" / "_generated_schema_tables.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    version = data.get("schema_version", "unknown")
    lines.append(f"Schema registry version: `{version}`\n")

    stages = data.get("stages", {})
    section_order = ["obs", "var", "obsm", "varm", "layers", "obsp", "uns"]

    for stage_name, stage_data in stages.items():
        lines.append(f"## {stage_name}\n")
        stage_requires = ", ".join(stage_data.get("stage_requires", []) or [])
        lines.append(f"Stage requires: {stage_requires or 'None'}\n")
        for section in section_order:
            entries = stage_data.get(section, {})
            lines.append(f"### {section}\n")
            if not entries:
                lines.append("_No entries defined._\n")
                continue
            lines.append(
                "| Key | Dtype | Created by | Modified by | Requires | Optional inputs | Notes |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
            )
            for key, meta in entries.items():
                dtype = meta.get("dtype", "")
                created_by = meta.get("created_by", "")
                modified_by = ", ".join(meta.get("modified_by", []) or [])
                requires = meta.get("requires", []) or []
                if requires and any(isinstance(item, list) for item in requires):
                    groups = [
                        " + ".join(group) if isinstance(group, list) else str(group)
                        for group in requires
                    ]
                    requires_text = " OR ".join(groups)
                else:
                    requires_text = ", ".join(requires)
                optional_inputs = ", ".join(meta.get("optional_inputs", []) or [])
                notes = meta.get("notes", "")
                lines.append(
                    f"| `{key}` | `{dtype}` | `{created_by}` | `{modified_by}` | "
                    f"{requires_text or 'None'} | {optional_inputs or 'None'} | {notes} |\n"
                )
            lines.append("\n")

    output_path.write_text("".join(lines), encoding="utf-8")


_generate_schema_tables()
