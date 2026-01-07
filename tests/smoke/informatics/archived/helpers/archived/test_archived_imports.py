"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest

from tests.smoke.import_helpers import import_module_or_skip


MODULES = [
    "smftools.informatics.archived.helpers.archived.align_and_sort_BAM",
    "smftools.informatics.archived.helpers.archived.aligned_BAM_to_bed",
    "smftools.informatics.archived.helpers.archived.bam_qc",
    "smftools.informatics.archived.helpers.archived.bed_to_bigwig",
    "smftools.informatics.archived.helpers.archived.canoncall",
    "smftools.informatics.archived.helpers.archived.concatenate_fastqs_to_bam",
    "smftools.informatics.archived.helpers.archived.converted_BAM_to_adata",
    "smftools.informatics.archived.helpers.archived.count_aligned_reads",
    "smftools.informatics.archived.helpers.archived.demux_and_index_BAM",
    "smftools.informatics.archived.helpers.archived.extract_base_identities",
    "smftools.informatics.archived.helpers.archived.extract_mods",
    "smftools.informatics.archived.helpers.archived.extract_read_features_from_bam",
    "smftools.informatics.archived.helpers.archived.extract_read_lengths_from_bed",
    "smftools.informatics.archived.helpers.archived.extract_readnames_from_BAM",
    "smftools.informatics.archived.helpers.archived.find_conversion_sites",
    "smftools.informatics.archived.helpers.archived.generate_converted_FASTA",
    "smftools.informatics.archived.helpers.archived.get_chromosome_lengths",
    "smftools.informatics.archived.helpers.archived.get_native_references",
    "smftools.informatics.archived.helpers.archived.index_fasta",
    "smftools.informatics.archived.helpers.archived.informatics",
    "smftools.informatics.archived.helpers.archived.load_adata",
    "smftools.informatics.archived.helpers.archived.make_modbed",
    "smftools.informatics.archived.helpers.archived.modQC",
    "smftools.informatics.archived.helpers.archived.modcall",
    "smftools.informatics.archived.helpers.archived.ohe_batching",
    "smftools.informatics.archived.helpers.archived.ohe_layers_decode",
    "smftools.informatics.archived.helpers.archived.one_hot_decode",
    "smftools.informatics.archived.helpers.archived.one_hot_encode",
    "smftools.informatics.archived.helpers.archived.plot_bed_histograms",
    "smftools.informatics.archived.helpers.archived.separate_bam_by_bc",
    "smftools.informatics.archived.helpers.archived.split_and_index_BAM",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
