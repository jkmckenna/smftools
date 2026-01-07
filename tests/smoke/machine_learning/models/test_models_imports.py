"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest
from tests.smoke.import_helpers import import_module_or_skip

MODULES = [
    "smftools.machine_learning.models.base",
    "smftools.machine_learning.models.cnn",
    "smftools.machine_learning.models.lightning_base",
    "smftools.machine_learning.models.mlp",
    "smftools.machine_learning.models.positional",
    "smftools.machine_learning.models.rnn",
    "smftools.machine_learning.models.sklearn_models",
    "smftools.machine_learning.models.transformer",
    "smftools.machine_learning.models.wrappers",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
