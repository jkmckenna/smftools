from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark all tests under tests/unit as unit tests."""
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/unit/" in f"/{path}":
            item.add_marker(pytest.mark.unit)
