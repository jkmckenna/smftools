# AGENTS.md / CLAUDE.md file plan

Staging doc for restructuring the repo's agent-instruction files. This is **not** deployed
anywhere itself (`dev/` is gitignored) — it's the source to copy from. AGENTS.md/CLAUDE.md
files cannot be created or edited by an agent (root `AGENTS.md` says so explicitly), so a human
has to place each section below at its destination path.

## Why

- Root `AGENTS.md`'s "Click commands" section is the fastest-growing, most repo-structure-coupled
  part of a file that's supposed to be stable global policy — every new CLI subcommand means
  editing it.
- Several real bugs this session (docs build failures, CI collection failures) came from
  conventions that exist nowhere in `AGENTS.md`: doc-build gotchas, docstring RST pitfalls,
  missing optional-dependency extras in CI.
- `src/smftools/analysis/CLAUDE.md` is a validated example of a subpackage-local file working
  well — this plan generalizes that pattern to `cli/`, `docs/`, and `tests/`.

## File map

| Path | Status | Purpose |
|---|---|---|
| `AGENTS.md` (root) | **revise** | Trim to stable, global policy. Drop the Click-command deep-dive. |
| `src/smftools/cli/AGENTS.md` | **new** | CLI group map, the `_core()` pattern, shared helpers. |
| `src/smftools/analysis/CLAUDE.md` | **keep as-is** | Already good. Rename to `AGENTS.md` for tool-agnostic consistency (see note at bottom). |
| `docs/source/AGENTS.md` | **new** | Napoleon/RST pitfalls, mock-imports rule, local doc-build check. |
| `tests/AGENTS.md` | **new** | Marker taxonomy, venv/extras gotcha, doctest-modules note. |
| `src/smftools/informatics/AGENTS.md`, `src/smftools/hmm/AGENTS.md` | **deferred** | Real candidates (big, non-obvious subpackages) but not drafted yet — write these once they've actually caused repeated confusion, not preemptively. |

Each section below is the literal proposed content for that file.

---

## 1. `AGENTS.md` (root) — revised

```markdown
# Claude Code Agent Instructions

You are the implementation agent defined in smftools/AGENTS.md

- For AGENTS.md or CLAUDE.md files (this one, and any nested ones under `src/`, `docs/`, `tests/`):
  - Agents can read from these files.
  - Agents can never edit these files.

## Goals
- Make minimal, correct changes.
- Prefer small PRs / diffs.
- Keep behavior stable unless the task explicitly requests changes.
- Generate production grade, scalable code.

## Prompt interface
- When asked about a problem or task, first read all files relevent to the task's scope.
- Describe the problem given the context.
- Formulate a plan to address the problem within scope.
- Refine the plan with user input.
- Implement code after being told to proceed.

## Repo orientation
- Read existing patterns before inventing new ones.
- Don't refactor broadly unless asked.
- If you're unsure about intended behavior, look for tests or docs first.
- Ignore all files in any directory named "archived".
- User defined parameters exist within src/smftools/config.
- Parameters are inherited from default.yaml -> MODALITY.yaml -> user_defined_config.csv
- Frequently used non user defined variables should exist within src/smftools/constants.py
- Logging functionality is defined within src/smftools/logging_utils.py
- Optional dependency handling is defined within src/smftools/optional_imports.py
- Frequently used I/O functionality is defined within src/smftools/readwrite.py
- CLI functionality is provided through click; see src/smftools/cli/AGENTS.md for the command
  map and conventions before editing anything under src/smftools/cli/.
- RTD documentation organization through smftools/docs; see docs/source/AGENTS.md before editing
  docstrings or anything under docs/.
- Pytest testing within smftools/tests; see tests/AGENTS.md for markers and known gotchas before
  running or writing tests.
- smftools.analysis (downstream analysis library) has its own design contract at
  src/smftools/analysis/AGENTS.md — read it before adding to compute/plot/filters/config.
- Nested AGENTS.md/CLAUDE.md files exist to keep this file from re-growing per-subpackage detail.
  If you find yourself wanting to document something specific to one subpackage here, it
  probably belongs in that subpackage's own file instead (create one if it doesn't exist, and
  flag it to the user since agents can't create AGENTS.md/CLAUDE.md files themselves).

## Project dependencies
- A core set of dependencies is required for the project.
- Various optional dependencies are provided for:
    - Optional functional modules of the package (ont, plotting, ml-base, ml-extended, umap, qc,
      pysam, catalog, cluster, ...) — see pyproject.toml's [project.optional-dependencies] for
      the full, current list; do not enumerate it here, it changes often.
    - If available, a Python version of a CLI tool is preferred (Such as for Samtools, Bedtools,
      BedGraphToBigWig).
    - torch is listed as an extra dependency, but is currently required.
    - All functional extras can be installed with `pip install -e ".[all_2]"` (the more complete
      of two overlapping "everything" extras — `all` predates `pybedtools`/`pyBigWig` being added
      and is missing them; prefer `all_2`). See Setup below for the canonical dev venv this
      produces.
- Certain command line tools are currently needed for certain functionalities within smftools load:
  - dorado: Used for nanopore basecalling from POD5/FAST5 files to BAM.
  - dorado/minimap2: Used for alignment of reads to reference.
  - dorado: Used for demultiplexing of nanopore derived BAMs.
  - modkit: Used for extracting modification probabilities from MM/ML BAM tags for native smf modality.

## Setup

Which interpreter to use, in priority order:

1. **User- or task-specified venv/interpreter** — if one is given, use it, full stop.
2. **The currently active environment**, if it already satisfies what the task needs (e.g. it's
   already running with the right packages importable) — don't switch environments just because
   a canonical one exists.
3. **`venvs/venv-all`** — the canonical, fully-provisioned dev venv (editable install, every
   functional extra: `pip install -e ".[all_2,dev,docs]"`). This is the default when neither of
   the above applies. Being editable, it always reflects whatever branch is currently checked
   out — it does not need to be recreated per branch, only re-installed
   (`venvs/venv-all/bin/pip install -e ".[all_2,dev,docs]"`) if `pyproject.toml`'s dependencies
   changed since it was last built.

If `venvs/venv-all` doesn't exist yet or a narrower environment is wanted, create one:
- `python3 -m venv venvs/<name> && venvs/<name>/bin/pip install -e ".[dev,torch]"` (core +
  dev/test tooling), then add extras as needed: `venvs/<name>/bin/pip install -e ".[EXTRA_NAME]"`.

**Common trap**: a venv/interpreter that's missing an optional extra (e.g. `pod5`, `pysam`,
`umap`) will fail *test collection*, not just individual tests, for any file that imports it
at module level — this looks like a code regression but usually isn't. Before debugging, check
which interpreter you're actually running and whether it has the extras the failing files need.
`venvs/venv-all` exists specifically to make this class of bug not happen in the first place.

## How to run checks
- Smoke tests: `pytest -m smoke -q`
- Unit tests: `pytest -m unit -q`
- Integration tests: `pytest -m integration -q`
- E2E tests: `pytest -m e2e -q`
- Coverage (if configured): `pytest --cov`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type-check (if configured): `mypy .`
- **Docs build** (before committing anything that touches a docstring or `docs/`):
  `sphinx-build -W -b html docs/source docs/_build/html`. `-W` treats warnings as errors, matching
  CI's `docs` job and Read the Docs' `fail_on_warning: true` — a docstring that imports fine can
  still fail this. See docs/source/AGENTS.md for the specific pitfalls that trip this up.

## Coding conventions
- Follow existing style and module layout.
- Prefer clear, explicit code over cleverness.
- Prefer modular functionality to facilitate testing and future development.
- Do not over-parametize functions when possible.
- For function parameters that a user may want to tune, use the config management strategy.
- Use constants.py when appropriate.
- Annotate code blocks to describe functionality.
- Add/adjust tests for bug fixes and new behavior.
- Keep public APIs backward compatible unless explicitly changing them.
- Python:
  - Use type hints for new/modified functions where reasonable.
  - Use Google style docstring format.
  - Avoid heavy dependencies unless necessary.
  - Use typing.TYPE_CHECKING and annotations.
  - In docstring of new functions, define the purpose of the function and what it does.
  - If a function's return-type annotation (or any forward-referenced type) names a symbol that
    is only ever imported inside a `TYPE_CHECKING` block, that symbol's *top-level package* must
    also be in `docs/source/conf.py`'s `autodoc_mock_imports` — otherwise the docs build breaks
    even though the code runs fine. See docs/source/AGENTS.md.

## Testing expectations
- New functionality must include tests.
- If tests are flaky or slow, note it and scope the change.
- There is currently no "regression" test marker/category, despite the concept coming up in
  practice — if you need one, propose it explicitly rather than assuming a convention exists.

## Logging & secrets
- Don't log secrets, tokens, or PII.
- Never hardcode credentials.
- If sample keys are needed, use obvious placeholders like `YOUR_API_KEY_HERE`.

## Git / PR hygiene
- Keep commits focused.
- Update docs/changelog if behavior or user-facing CLI changes.
- If you change a CLI flag or config schema, add a migration note.
- Cut a new `<minor>.0-<description>` branch and bump `src/smftools/_version.py` before each
  distinct track of work, not one branch for everything.
- When cutting a new version branch: tag the outgoing branch's HEAD locally
  (`git tag -a vMAJOR.MINOR.PATCH <branch> -m "..."`), run `python -m build && twine check dist/*`
  as a local sanity check, and summarize `git log <prev-tag>..HEAD` for the user (a candidate
  docs/source/release-notes/<version>.md entry). Do not push tags, delete branches, or publish
  build artifacts without explicit confirmation each time — these are shared-state/irreversible
  actions on a public repo.

## If something fails
- If a command fails, paste the full error and summarize likely causes.
- Don't "fix" by deleting tests or weakening assertions unless explicitly instructed.
```

**Removed from the original**: the entire "Click commands and their primary intent" section
(moved to `src/smftools/cli/AGENTS.md` below, where it's actually adjacent to what it describes).

---

## 2. `src/smftools/cli/AGENTS.md` — new

```markdown
# AGENTS.md — src/smftools/cli

CLI implementation for the `smftools` command. See root AGENTS.md first for repo-wide policy;
this file covers conventions specific to this subpackage.

## Structure

`src/smftools/cli_entry.py` defines the Click groups and thin command wrappers.
`src/smftools/cli/*.py` holds the actual per-stage implementation, one module per stage/concern.

Every pipeline-stage command follows the same split:

- `<stage>_adata(config_path)` — the Click-facing wrapper. Resolves config, decides whether the
  stage needs to run at all (stage-skip / already-done checks), and if so calls
  `<stage>_adata_core(...)`.
- `<stage>_adata_core(...)` — the real logic. This is almost always the function you actually
  want to read or edit; the outer wrapper is boilerplate.

Exception: `raw_adata()` (in `raw_adata.py`) does not have its own `_core` — it delegates to
`load_adata_core(cfg, paths, config_path=config_path, raw_only=True)` in `load_adata.py`, since
raw ingestion and dense-cache loading share the same underlying function.

## Command map

`smftools --help` is authoritative; this is a summary. Two top-level groups:

### `smftools experiment <config_path>` — pipeline stages for a single experiment

| Command | Module | Core function | Purpose |
|---|---|---|---|
| `raw` | `raw_adata.py` | `load_adata_core(..., raw_only=True)` | Prepare BAM artifacts and write the ragged raw store. |
| `load` | `load_adata.py` | `load_adata_core` | Optionally pre-build the dense zarr cache from raw artifacts. |
| `preprocess` | `preprocess_adata.py` | `preprocess_adata_core` | QC, filtering, read-level preprocessing. |
| `variant` | `variant_adata.py` | `variant_adata_core` | Sequence variation analyses. |
| `chimeric` | `chimeric_adata.py` | `chimeric_adata_core` | Detect putative PCR chimeras. |
| `spatial` | `spatial_adata.py` | `spatial_adata_core` | Spatial signal analysis. |
| `hmm` | `hmm_adata.py` | `hmm_adata_core` | HMM feature annotation and plotting. |
| `latent` | `latent_adata.py` | `latent_adata_core` | Latent representations (PCA/UMAP/NMF/CP). |
| `full` | `recipes.py` | — | Composed workflow: raw, preprocess, spatial, hmm. |
| `batch` | `cli_entry.py` | — | Run one stage across many experiments from a CSV/TSV/TXT. |
| `concatenate` | `cli_entry.py` | — | Merge multiple `.h5ad` files into one. |
| `export-fastq` | `export_fastq.py` | `export_fastq_for_experiment` | FASTQ export of QC-passed reads, per experiment. |
| `plot-current` | `plot_current.py` | — | Plot nanopore current traces for specified reads. |

### `smftools project <project_dir>` — registering and querying across experiments

| Command | Purpose |
|---|---|
| `init` | Initialize a project directory + registry. |
| `add` / `remove` | Register or deactivate an experiment in the project. |
| `list` | List registered experiments and harmonized references. |
| `materialize` | Pool a canonical reference across matching experiments into one AnnData. |
| `sample-store-list` | List cataloged per-sample-store partitions. |
| `export-fastq` | FASTQ export of QC-passed reads, across every registered experiment. |

All `project_*` commands live in `project_cmd.py`.

## Shared helpers

- `helpers.py` — `AdataPaths`/`ArtifactPaths` dataclasses (canonical per-stage file paths) and
  `resolve_adata_stage()` (stage-fallback resolution: hmm > latent > spatial > chimeric > variant
  > pp_dedup > pp > raw).
- `stage_input.py` — `StageSlice` dataclass for partition-scoped stage inputs.
- `stage_artifacts.py` — `StagePlotPaths` dataclass for per-stage output figure paths.

## When adding a new CLI command

1. Decide if it's `experiment`-scoped (one config, one experiment) or `project`-scoped (crosses
   experiments) — this determines which Click group it joins in `cli_entry.py`.
2. If it does real work beyond a thin wrapper, follow the `<name>(...)` / `<name>_core(...)`
   split so the logic is testable independent of Click.
3. Update the table above and `docs/source/cli.md`.
```

---

## 3. `docs/source/AGENTS.md` — new

```markdown
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
```

---

## 4. `tests/AGENTS.md` — new

```markdown
# AGENTS.md — tests

See root AGENTS.md first for repo-wide policy. This file covers conventions specific to running
and writing tests in this repo.

## Markers

Defined in `pyproject.toml`'s `[tool.pytest.ini_options]`:

- `smoke` — rapid, runtime and import tests.
- `unit` — fast, function tests without external dependencies.
- `integration` — slower, functional tests with external dependencies.
- `e2e` — slowest, end-to-end workflow testing.

Run a subset with `pytest -m <marker> -q`. There is no `regression` marker despite the concept
coming up in practice — if you need one, propose it, don't assume it exists.

## `--doctest-modules` is on

`pyproject.toml` sets `addopts = [..., "--doctest-modules", ...]`. Doctests in module docstrings
are collected and run, not just files under `tests/`.

## The most common false-alarm: wrong interpreter, not a real failure

If `pytest --collect-only` fails with `ModuleNotFoundError` for something like `pod5` or `pysam`
for files that otherwise look unrelated to your change, this is almost always an interpreter
missing an optional extra, not a code regression — those packages are imported at module level in
files like `informatics/pod5_functions.py`, so any test file that imports that module (even
transitively) fails to *collect*, not just to run. Check which Python you're actually invoking
and whether it has the extras those files need (see the relevant `[project.optional-dependencies]`
entry in `pyproject.toml`, e.g. `ont` for `pod5`, `pysam` for `pysam`). This exact issue caused
three separate CI failures in one session before the root cause (missing extras in the CI
install step, not a code bug) was found.

## Before assuming a failure is yours

Some tests can be flaky or have pre-existing failures unrelated to your change. Before debugging
deeply, `git stash` your changes, rerun the specific failing test(s), and confirm whether they
fail on a clean checkout too. If they do, note it and scope your change rather than trying to fix
unrelated pre-existing issues as a side effect.
```

---

## Note: `analysis/CLAUDE.md` naming

`src/smftools/analysis/CLAUDE.md` already exists and is good — no content changes proposed. But
AGENTS.md's own stated goal is multi-tool support (Codex, Gemini, Claude), and those tools look
for `AGENTS.md`, not `CLAUDE.md` — `CLAUDE.md` is a Claude Code-specific convention (matching the
root, where `CLAUDE.md` is a one-line pointer to `AGENTS.md`, not the real content). Recommend:
rename `src/smftools/analysis/CLAUDE.md` → `src/smftools/analysis/AGENTS.md`, content unchanged,
for consistency with everything else in this plan.

## Deferred: `informatics/` and `hmm/`

Both are large, architecturally dense subpackages (partitioned store, ragged store, raw
ingestion, the HMM class hierarchy) that would benefit from their own file eventually. Not
drafted here — write these once they've caused actual repeated confusion in a session, the same
bar used for deciding what's worth writing to memory, rather than pre-emptively guessing at
what future agents will need.
