from importlib import import_module

_LAZY_ATTRS = {
    "combined_hmm_raw_clustermap": "smftools.plotting.general_plotting",
    "combined_raw_clustermap": "smftools.plotting.general_plotting",
    "plot_hmm_layers_rolling_by_sample_ref": "smftools.plotting.general_plotting",
    "plot_bar_relative_risk": "smftools.plotting.position_stats",
    "plot_positionwise_matrix": "smftools.plotting.position_stats",
    "plot_positionwise_matrix_grid": "smftools.plotting.position_stats",
    "plot_volcano_relative_risk": "smftools.plotting.position_stats",
    "plot_feature_importances_or_saliency": "smftools.plotting.classifiers",
    "plot_model_curves_from_adata": "smftools.plotting.classifiers",
    "plot_model_curves_from_adata_with_frequency_grid": "smftools.plotting.classifiers",
    "plot_model_performance": "smftools.plotting.classifiers",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "combined_hmm_raw_clustermap",
    "plot_bar_relative_risk",
    "plot_positionwise_matrix",
    "plot_positionwise_matrix_grid",
    "plot_volcano_relative_risk",
    "plot_feature_importances_or_saliency",
    "plot_model_performance",
    "plot_model_curves_from_adata",
    "plot_model_curves_from_adata_with_frequency_grid",
]
