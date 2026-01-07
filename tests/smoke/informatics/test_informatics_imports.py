"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.informatics.bam_functions",
    "smftools.informatics.basecalling",
    "smftools.informatics.bed_functions",
    "smftools.informatics.binarize_converted_base_identities",
    "smftools.informatics.complement_base_list",
    "smftools.informatics.converted_BAM_to_adata",
    "smftools.informatics.fasta_functions",
    "smftools.informatics.h5ad_functions",
    "smftools.informatics.modkit_extract_to_adata",
    "smftools.informatics.modkit_functions",
    "smftools.informatics.ohe",
    "smftools.informatics.pod5_functions",
    "smftools.informatics.run_multiqc",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
