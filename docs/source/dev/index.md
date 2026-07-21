(contribution-guide)=

# Contributing

## Development environment

Install runtime features and contributor tooling separately:

```shell
python -m pip install -e ".[all]"
python -m pip install --group dev --group docs
```

## Test tiers

The test markers represent execution boundaries, not levels of importance:

| Tier | Command | Purpose |
| --- | --- | --- |
| Smoke | `pytest -m smoke -q` | Rapid import and runtime checks |
| Unit | `pytest -m unit -q` | Isolated function and component regressions |
| Integration | `pytest -m integration -q` | Functional multiprocessing and storage paths using checked-in fixtures |
| E2E | `pytest -m e2e -q` | Full workflows that require external tools and configured experiment outputs |

The required pull-request workflow runs smoke tests on Python 3.11 and the full
smoke plus unit suite on Python 3.12. It also checks formatting, lint, minimum
storage dependency versions, documentation, and distribution builds.

The `Extended CI` workflow runs weekly and can be started manually from GitHub
Actions. It runs the full smoke plus unit suite on the minimum Python 3.11 and
the checked-in-fixture integration suite on Python 3.12. This preserves minimum-
version regression coverage without duplicating every unit test on every pull
request. Python 3.11 and 3.12 are the currently tested interpreter versions.

E2E tests remain local because they require tools such as Dorado, minimap2, and
modkit and write outputs specified by their experiment configurations. Run them
only in a disposable or explicitly selected output directory.

## Local checks

Before opening a pull request, run the tiers relevant to the change. Changes to
documentation or docstrings also require the warning-as-error documentation
build:

```shell
ruff check .
ruff format --check .
pytest -m smoke -q
pytest -m unit -q
sphinx-build -W -b html docs/source docs/_build/html
```
