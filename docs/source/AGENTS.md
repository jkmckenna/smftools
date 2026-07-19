# AGENTS.md — docs/source

Sphinx documentation source. See root AGENTS.md first. This file exists because several docs-CI
failures this session came from patterns that are easy to write by accident and don't fail until
`sphinx-build -W` runs — not at commit time, not at `pytest` time.

## Before committing anything that touches a docstring, conf.py, or docs/source/

Run the actual doc build locally:

```bash
pip install -e ".[docs]"   # in whatever venv you're using, or a scratch one
sphinx-build -W -b html docs/source docs/_build/html
```

`-W` treats every Sphinx warning as a build failure — this matches both CI's `docs` job and Read
the Docs' `fail_on_warning: true`. A docstring that imports and runs fine can still fail this.
Clean up any `docs/_build/` and `docs/source/api/generated/` / `docs/source/schema/_generated_schema_tables.md`
this produces before committing — those are build artifacts, not source.

## Docstring pitfalls that pass everywhere except this build

This project uses Napoleon with `napoleon_google_docstring = True`,
`napoleon_numpy_docstring = False`. Google-style `Args:`/`Returns:`/etc. sections get converted
to proper RST before parsing. Everything else in a docstring — including NumPy-style
`Parameters\n----------` blocks and any custom section header like `Channels:` or `Modules:` — is
parsed as **raw RST**, and RST is stricter than it looks:

1. **A bullet/numbered list glued to the line above with no blank line breaks.** RST requires a
   blank line before a list starts; without one, the list line and everything above it merge into
   one paragraph, and the first genuinely-indented continuation line in that merged paragraph
   raises `Unexpected indentation`.

   ```
   # Wrong — no blank line before the list:
   Writes:
     - thing one
     - thing two

   # Right:
   Writes:

     - thing one
     - thing two
   ```

2. **A NumPy-style parameter with no description, followed by another bare parameter, followed by
   one that does have a description** — the bare ones merge into a paragraph with the next
   parameter name for the same reason as #1, then break when the indented description appears.
   Give every parameter in a block either all bare names or all `name : type\n    description`
   pairs, not a mix.

3. **Any word ending in a trailing underscore in prose** (e.g. sklearn's `PCA.explained_variance_ratio_`
   convention) is parsed by RST as a named hyperlink reference, and fails as "Unknown target name"
   since nothing defines that target. Wrap it in double backticks: `` ``PCA.explained_variance_ratio_`` ``.

4. **A `@dataclass` docstring using Napoleon's `Attributes:` section** generates its own
   `.. attribute::` directives, which collide with the ones autodoc introspects from the real
   dataclass fields ("duplicate object description"). Use a different header (e.g. `Fields:`) so
   Napoleon leaves it as plain prose instead.

5. **A pseudo-code dict/JSON literal in a docstring** (e.g. showing a return shape) gets its
   colons and braces parsed as RST definition-list syntax. Use a real literal block instead
   (`::` at the end of the preceding line) so the content isn't parsed as RST at all.

## `TYPE_CHECKING` imports and `autodoc_mock_imports`

Sphinx's `autodoc-typehints` extension flips `typing.TYPE_CHECKING` to `True` and *actually
executes* those guarded imports while building docs (this is what makes the
`if TYPE_CHECKING: import anndata as ad` pattern work for annotation resolution). That means:

- If a function's return type is a forward-referenced string (e.g. `"umap.UMAP"`, `"pd.DataFrame"`),
  the module owning that name must be imported somewhere under `TYPE_CHECKING` in that file, or
  the build fails with `Cannot resolve forward reference ... name 'X' is not defined`.
- That import will genuinely execute during the doc build. If the package isn't actually
  installed in the docs environment (most optional extras aren't — see `.[docs]` in
  `pyproject.toml`), it must be added to `autodoc_mock_imports` in `conf.py`, the same way
  `pod5`, `sklearn`, `anndata`, `torch`, etc. already are.

## `autosummary` structure

`docs/source/api/*.md` (`analysis.md`, `informatics.md`, `plotting.md`, `preprocessing.md`,
`tools.md`, `datasets.md`) each hand-list the submodules they document via
`.. autosummary:: :toctree: generated/...`. Adding a new module to a subpackage doesn't
automatically document it — add it to the relevant `api/*.md` file's list.