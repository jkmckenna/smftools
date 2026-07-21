from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.partitioned_pipeline import (
    PartitionedPipelineFixture,
    build_partitioned_pipeline_fixture,
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark all tests under tests/unit as unit tests."""
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/unit/" in f"/{path}":
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def partitioned_pipeline_store(tmp_path: Path) -> PartitionedPipelineFixture:
    """Write a compact two-reference/two-barcode raw partition store."""
    return build_partitioned_pipeline_fixture(tmp_path / "partitioned_pipeline")
