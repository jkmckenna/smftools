"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest
from tests.smoke.import_helpers import import_module_or_skip

MODULES = [
    "smftools.plotting.autocorrelation_plotting",
    "smftools.plotting.classifiers",
    "smftools.plotting.general_plotting",
    "smftools.plotting.hmm_plotting",
    "smftools.plotting.position_stats",
    "smftools.plotting.qc_plotting",
]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.smoke
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
