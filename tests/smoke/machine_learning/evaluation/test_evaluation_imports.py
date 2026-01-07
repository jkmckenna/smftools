"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest
from tests.smoke.import_helpers import import_module_or_skip

MODULES = [
    "smftools.machine_learning.evaluation.eval_utils",
    "smftools.machine_learning.evaluation.evaluators",
]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.smoke
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
