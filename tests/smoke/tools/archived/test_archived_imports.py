"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.tools.archived.apply_hmm",
    "smftools.tools.archived.classifiers",
    "smftools.tools.archived.classify_methylated_features",
    "smftools.tools.archived.classify_non_methylated_features",
    "smftools.tools.archived.subset_adata_v1",
    "smftools.tools.archived.subset_adata_v2",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
