"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.hmm.archived.apply_hmm_batched",
    "smftools.hmm.archived.calculate_distances",
    "smftools.hmm.archived.call_hmm_peaks",
    "smftools.hmm.archived.train_hmm",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
