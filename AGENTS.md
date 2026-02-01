# AGENTS.md

This file tells coding agents (including OpenAI Codex and Claude Code) how to work in this repo.

## Goals
- Make minimal, correct changes.
- Prefer small PRs / diffs.
- Keep behavior stable unless the task explicitly requests changes.

## Repo orientation
- Read existing patterns before inventing new ones.
- Don’t refactor broadly unless asked.
- If you’re unsure about intended behavior, look for tests/docs first.
- Ignore all files in any directory named "archived".
- User defined parameters exist within src/smftools/config.
- Parameters are herited from default.yaml -> MODALITY.yaml -> user_defined_config.csv
- Frequently used non user defined variables should exist within src/smftools/constants.py
- Logging functionality is defined within src/smftools/logging_utils.py
- Optional dependency handling is defined within src/smftools/optional_imports.py
- Frequently used I/O functionality is defined within src/smftools/readwrite.py
- CLI functionality is provided through click and is defined within:
  - src/smftools/cli_entry.py
  - Modules of the src/smtools/cli subpackage
- RTD documentation organization through smftools/docs
- Pytest testing within smftools/tests

## Project dependencies
- A core set of dependencies is required for the project.
- Various optional dependencies are provided for:
    - Optional functional modules of the package (ont, plotting, ml-base, ml-extended, umap, qc)
    - If available, a Python version of a CLI tool is preferred (Such as for Samtools, Bedtools, BedGraphToBigWig).
    - torch is listed as an extra dependency, but is currently required.
    - All dependencies can be installed with `pip install -e ".[all]"`

## Setup
- Use current environment if the core dependencies are installed.
- If dependencies are not found, create a venv in smftools/venvs/ directory:
  - `python3 -m venv .temp-venv && source .temp-venv/bin/activate`
- Install the core dependencies and development dependencies for testing/formatting/linting:
  - `pip install -e ".[dev,torch]"`
- If code is raising dependencies errors and they are in the optional dependencies:
  - `pip install -e ".[EXTRA_DEPENDENCY_NAME]"`

## How to run checks
- Smoke tests: `pytest -m smoke -q`
- Unit tests: `pytest -m unit -q`
- Integration tests: `pytest -m integration -q`
- E2E tests: `pytest -m e2e -q`
- Coverage (if configured): `pytest --cov`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type-check (if configured): `mypy .`

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

## Testing expectations
- New functionality must include tests.
- If tests are flaky or slow, note it and scope the change.

## Logging & secrets
- Don’t log secrets, tokens, or PII.
- Never hardcode credentials.
- If sample keys are needed, use obvious placeholders like `YOUR_API_KEY_HERE`.

## Git / PR hygiene
- Keep commits focused.
- Update docs/changelog if behavior or user-facing CLI changes.
- If you change a CLI flag or config schema, add a migration note.

## If something fails
- If a command fails, paste the full error and summarize likely causes.
- Don’t “fix” by deleting tests or weakening assertions unless explicitly instructed.
