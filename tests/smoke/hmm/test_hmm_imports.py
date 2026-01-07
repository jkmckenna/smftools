"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.hmm.HMM",
    "smftools.hmm.call_hmm_peaks",
    "smftools.hmm.display_hmm",
    "smftools.hmm.hmm_readwrite",
    "smftools.hmm.nucleosome_hmm_refinement",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
