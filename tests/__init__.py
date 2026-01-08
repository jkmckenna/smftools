"""Test suite configuration and marker guidance.

Use ``pytest -m smoke`` for rapid feedback on imports and runtime errors. Can be done in CI through github actions.
Use ``pytest -m unit`` for fast feedback on unit tests. Can be done in CI through github actions.
Use ``pytest -m integration`` for feedback on tests that require samtools, dorado, and minimap2. Run local.
Use ``pytest -m e2e`` to run slower, end-to-end style workflows. Run local.
Use ``pytest`` to run everything. Run local.
"""
