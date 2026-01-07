"""Import smoke tests for smftools modules."""

from __future__ import annotations

import pytest
from tests.smoke.import_helpers import import_module_or_skip

MODULES = [
    "smftools.preprocessing.append_base_context",
    "smftools.preprocessing.append_binary_layer_by_base_context",
    "smftools.preprocessing.binarize",
    "smftools.preprocessing.binarize_on_Youden",
    "smftools.preprocessing.binary_layers_to_ohe",
    "smftools.preprocessing.calculate_complexity_II",
    "smftools.preprocessing.calculate_consensus",
    "smftools.preprocessing.calculate_coverage",
    "smftools.preprocessing.calculate_pairwise_differences",
    "smftools.preprocessing.calculate_pairwise_hamming_distances",
    "smftools.preprocessing.calculate_position_Youden",
    "smftools.preprocessing.calculate_read_length_stats",
    "smftools.preprocessing.calculate_read_modification_stats",
    "smftools.preprocessing.clean_NaN",
    "smftools.preprocessing.filter_adata_by_nan_proportion",
    "smftools.preprocessing.filter_reads_on_length_quality_mapping",
    "smftools.preprocessing.filter_reads_on_modification_thresholds",
    "smftools.preprocessing.flag_duplicate_reads",
    "smftools.preprocessing.invert_adata",
    "smftools.preprocessing.load_sample_sheet",
    "smftools.preprocessing.make_dirs",
    "smftools.preprocessing.min_non_diagonal",
    "smftools.preprocessing.recipes",
    "smftools.preprocessing.reindex_references_adata",
    "smftools.preprocessing.subsample_adata",
]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.smoke
def test_imports(module_name: str) -> None:
    import_module_or_skip(module_name)
