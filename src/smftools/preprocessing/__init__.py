from __future__ import annotations

from importlib import import_module

_LAZY_ATTRS = {
    "append_base_context": "smftools.preprocessing.append_base_context",
    "append_binary_layer_by_base_context": "smftools.preprocessing.append_binary_layer_by_base_context",
    "binarize_adata": "smftools.preprocessing.binarize",
    "binarize_on_Youden": "smftools.preprocessing.binarize_on_Youden",
    "calculate_complexity_II": "smftools.preprocessing.calculate_complexity_II",
    "calculate_coverage": "smftools.preprocessing.calculate_coverage",
    "calculate_position_Youden": "smftools.preprocessing.calculate_position_Youden",
    "calculate_read_length_stats": "smftools.preprocessing.calculate_read_length_stats",
    "calculate_read_modification_stats": "smftools.preprocessing.calculate_read_modification_stats",
    "clean_NaN": "smftools.preprocessing.clean_NaN",
    "filter_adata_by_nan_proportion": "smftools.preprocessing.filter_adata_by_nan_proportion",
    "filter_reads_on_length_quality_mapping": "smftools.preprocessing.filter_reads_on_length_quality_mapping",
    "filter_reads_on_modification_thresholds": "smftools.preprocessing.filter_reads_on_modification_thresholds",
    "flag_duplicate_reads": "smftools.preprocessing.flag_duplicate_reads",
    "invert_adata": "smftools.preprocessing.invert_adata",
    "load_sample_sheet": "smftools.preprocessing.load_sample_sheet",
    "reindex_references_adata": "smftools.preprocessing.reindex_references_adata",
    "subsample_adata": "smftools.preprocessing.subsample_adata",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_LAZY_ATTRS.keys())
