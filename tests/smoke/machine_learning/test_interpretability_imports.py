"""Import smoke tests for backend-neutral interpretability contracts."""

from __future__ import annotations

import pytest
from tests.smoke.import_helpers import import_module_or_skip

MODULES = [
    "smftools.machine_learning.interpretability",
    "smftools.machine_learning.interpretability.artifacts",
    "smftools.machine_learning.interpretability.background",
    "smftools.machine_learning.interpretability.contracts",
]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.smoke
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)


@pytest.mark.smoke
def test_top_level_lazy_module_export() -> None:
    from smftools import machine_learning

    assert machine_learning.interpretability.InterpretabilityRequest is not None
