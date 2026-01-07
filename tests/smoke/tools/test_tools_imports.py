"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.tools.calculate_umap",
    "smftools.tools.cluster_adata_on_methylation",
    "smftools.tools.general_tools",
    "smftools.tools.position_stats",
    "smftools.tools.read_stats",
    "smftools.tools.spatial_autocorrelation",
    "smftools.tools.subset_adata",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
