# AGENTS.md

This file tells coding agents (including OpenAI Codex) how to work in this repo.

## Goals
- Make minimal, correct changes.
- Prefer small PRs / diffs.
- Keep behavior stable unless the task explicitly requests changes.

## Repo orientation
- Read existing patterns before inventing new ones.
- Don’t refactor broadly unless asked.
- If you’re unsure about intended behavior, look for tests/docs first.

## Setup
- Create env (pick one):
  - `python -m venv .venv && source .venv/bin/activate`
  - or `conda env create -f environment.yml && conda activate <env>`
- Install:
  - `pip install -e ".[dev]"`

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
- Add/adjust tests for bug fixes and new behavior.
- Keep public APIs backward compatible unless explicitly changing them.
- Python:
  - Use type hints for new/modified functions where reasonable.
  - Use Google style docstring format.
  - Avoid heavy dependencies unless necessary.

## Testing expectations
- New functionality must include tests.
- Bug fix PRs should include a regression test.
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
