"""Test-suite marker guidance.

Use ``pytest -m smoke`` for rapid import and runtime feedback.
Use ``pytest -m unit`` for isolated function and component tests.
Use ``pytest -m integration`` for slower functional tests using checked-in fixtures.
Use ``pytest -m e2e`` for external-tool end-to-end workflows; run these locally.
Use ``pytest`` locally to run every collectable test.
"""
