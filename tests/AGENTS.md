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