"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.machine_learning.inference.inference_utils",
    "smftools.machine_learning.inference.lightning_inference",
    "smftools.machine_learning.inference.sklearn_inference",
    "smftools.machine_learning.inference.sliding_window_inference",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
