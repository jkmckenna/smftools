from __future__ import annotations

from importlib import import_module

_LAZY_ATTRS = {
    "combined_hmm_length_clustermap": "smftools.plotting.hmm_plotting",
    "combined_hmm_raw_clustermap": "smftools.plotting.hmm_plotting",
    "combined_raw_clustermap": "smftools.plotting.spatial_plotting",
    "plot_delta_hamming_summary": "smftools.plotting.chimeric_plotting",
    "plot_hamming_span_trio": "smftools.plotting.chimeric_plotting",
    "plot_rolling_nn_and_layer": "smftools.plotting.chimeric_plotting",
    "plot_rolling_nn_and_two_layers": "smftools.plotting.chimeric_plotting",
    "plot_segment_length_histogram": "smftools.plotting.chimeric_plotting",
    "plot_span_length_distributions": "smftools.plotting.chimeric_plotting",
    "plot_zero_hamming_pair_counts": "smftools.plotting.chimeric_plotting",
    "plot_zero_hamming_span_and_layer": "smftools.plotting.chimeric_plotting",
    "plot_hmm_layers_rolling_by_sample_ref": "smftools.plotting.hmm_plotting",
    "plot_nmf_components": "smftools.plotting.latent_plotting",
    "plot_pca_components": "smftools.plotting.latent_plotting",
    "plot_cp_sequence_components": "smftools.plotting.latent_plotting",
    "plot_embedding": "smftools.plotting.latent_plotting",
    "plot_embedding_grid": "smftools.plotting.latent_plotting",
    "plot_read_span_quality_clustermaps": "smftools.plotting.preprocess_plotting",
    "plot_mismatch_base_frequency_by_position": "smftools.plotting.variant_plotting",
    "plot_pca": "smftools.plotting.latent_plotting",
    "plot_pca_grid": "smftools.plotting.latent_plotting",
    "plot_pca_explained_variance": "smftools.plotting.latent_plotting",
    "plot_sequence_integer_encoding_clustermaps": "smftools.plotting.variant_plotting",
    "plot_variant_segment_clustermaps": "smftools.plotting.variant_plotting",
    "plot_variant_segment_clustermaps_multi_obs": "smftools.plotting.variant_plotting",
    "plot_umap": "smftools.plotting.latent_plotting",
    "plot_umap_grid": "smftools.plotting.latent_plotting",
    "plot_bar_relative_risk": "smftools.plotting.position_stats",
    "plot_positionwise_matrix": "smftools.plotting.position_stats",
    "plot_positionwise_matrix_grid": "smftools.plotting.position_stats",
    "plot_volcano_relative_risk": "smftools.plotting.position_stats",
    "plot_feature_importances_or_saliency": "smftools.plotting.classifiers",
    "plot_model_curves_from_adata": "smftools.plotting.classifiers",
    "plot_model_curves_from_adata_with_frequency_grid": "smftools.plotting.classifiers",
    "plot_model_performance": "smftools.plotting.classifiers",
    "plot_read_qc_histograms": "smftools.plotting.qc_plotting",
    "plot_rolling_grid": "smftools.plotting.autocorrelation_plotting",
    "plot_spatial_autocorr_grid": "smftools.plotting.autocorrelation_plotting",
    "plot_hmm_size_contours": "smftools.plotting.hmm_plotting",
    "plot_read_current_traces": "smftools.plotting.pod5_plotting",
    "plot_umi_bipartite_summary": "smftools.plotting.umi_plotting",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_LAZY_ATTRS.keys())
