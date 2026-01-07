"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.informatics.archived.bam_conversion",
    "smftools.informatics.archived.bam_direct",
    "smftools.informatics.archived.basecall_pod5s",
    "smftools.informatics.archived.basecalls_to_adata",
    "smftools.informatics.archived.conversion_smf",
    "smftools.informatics.archived.deaminase_smf",
    "smftools.informatics.archived.direct_smf",
    "smftools.informatics.archived.fast5_to_pod5",
    "smftools.informatics.archived.print_bam_query_seq",
    "smftools.informatics.archived.subsample_fasta_from_bed",
    "smftools.informatics.archived.subsample_pod5",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
