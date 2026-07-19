# AGENTS.md - root

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
      produces.- Certain command line tools are currently needed for certain functionalities within smftools load:
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