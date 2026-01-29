from __future__ import annotations

from smftools.logging_utils import get_logger

from smftools.plotting.hmm_plotting import (
    combined_hmm_length_clustermap,
    combined_hmm_raw_clustermap,
    plot_hmm_layers_rolling_by_sample_ref,
)
from smftools.plotting.latent_plotting import (
    plot_cp_sequence_components,
    plot_embedding,
    plot_embedding_grid,
    plot_nmf_components,
    plot_pca,
    plot_pca_components,
    plot_pca_explained_variance,
    plot_pca_grid,
    plot_umap,
    plot_umap_grid,
)
from smftools.plotting.preprocess_plotting import (
    plot_read_span_quality_clustermaps,
    plot_sequence_integer_encoding_clustermaps,
)
from smftools.plotting.spatial_plotting import (
    combined_raw_clustermap,
    plot_rolling_nn_and_layer,
    plot_zero_hamming_pair_counts,
    plot_zero_hamming_span_and_layer,
)

logger = get_logger(__name__)

__all__ = [
    "combined_hmm_length_clustermap",
    "combined_hmm_raw_clustermap",
    "combined_raw_clustermap",
    "plot_rolling_nn_and_layer",
    "plot_zero_hamming_pair_counts",
    "plot_zero_hamming_span_and_layer",
    "plot_hmm_layers_rolling_by_sample_ref",
    "plot_nmf_components",
    "plot_pca_components",
    "plot_cp_sequence_components",
    "plot_embedding",
    "plot_embedding_grid",
    "plot_read_span_quality_clustermaps",
    "plot_pca",
    "plot_pca_grid",
    "plot_pca_explained_variance",
    "plot_sequence_integer_encoding_clustermaps",
    "plot_umap",
    "plot_umap_grid",
]
