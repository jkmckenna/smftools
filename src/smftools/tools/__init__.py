from __future__ import annotations

from importlib import import_module

_LAZY_ATTRS = {
    "calculate_leiden": "smftools.tools.calculate_leiden",
    "calculate_nmf": "smftools.tools.calculate_nmf",
    "calculate_sequence_cp_decomposition": "smftools.tools.tensor_factorization",
    "calculate_umap": "smftools.tools.calculate_umap",
    "cluster_adata_on_methylation": "smftools.tools.cluster_adata_on_methylation",
    "combine_layers": "smftools.tools.general_tools",
    "create_nan_mask_from_X": "smftools.tools.general_tools",
    "create_nan_or_non_gpc_mask": "smftools.tools.general_tools",
    "calculate_relative_risk_on_activity": "smftools.tools.position_stats",
    "compute_positionwise_statistics": "smftools.tools.position_stats",
    "calculate_row_entropy": "smftools.tools.read_stats",
    "rolling_window_nn_distance": "smftools.tools.rolling_nn_distance",
    "subset_adata": "smftools.tools.subset_adata",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_LAZY_ATTRS.keys())
