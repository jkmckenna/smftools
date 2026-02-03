import anndata as ad
import numpy as np

from smftools.tools.rolling_nn_distance import (
    annotate_zero_hamming_segments,
    assign_per_read_segments_layer,
    segments_to_per_read_dataframe,
)


def test_annotate_zero_hamming_segments_parent_layer_maps_vars():
    parent = ad.AnnData(X=np.zeros((2, 5)))
    parent.obs_names = ["read1", "read2"]
    parent.var_names = ["v0", "v1", "v2", "v3", "v4"]

    subset = parent[:, ["v1", "v2", "v3"]].copy()
    subset.layers["nan0_0minus1"] = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    subset.uns["zero_pairs"] = [np.array([[0, 1]])]
    subset.uns["zero_pairs_starts"] = np.array([0])
    subset.uns["zero_pairs_window"] = 2
    subset.uns["zero_pairs_min_overlap"] = 1

    records = annotate_zero_hamming_segments(
        subset,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="segments",
        layer="nan0_0minus1",
        min_overlap=1,
        refine_segments=False,
    )
    per_read_segments = segments_to_per_read_dataframe(records, subset.var_names.to_numpy())
    assign_per_read_segments_layer(parent, subset, per_read_segments, layer_key="zero_span")

    assert "zero_span" in parent.layers
    target = parent.layers["zero_span"]
    assert target.shape == (2, 5)
    assert np.all(target[:, 1:3] == 1)
    assert np.all(target[:, [0, 3, 4]] == 0)


def test_annotate_zero_hamming_segments_parent_layer_fills_gaps():
    parent = ad.AnnData(X=np.zeros((2, 5)))
    parent.obs_names = ["read1", "read2"]
    parent.var_names = ["v0", "v1", "v2", "v3", "v4"]

    subset = parent[:, ["v1", "v3"]].copy()
    subset.layers["nan0_0minus1"] = np.array([[1.0, 1.0], [1.0, 1.0]])
    subset.uns["zero_pairs"] = [np.array([[0, 1]])]
    subset.uns["zero_pairs_starts"] = np.array([0])
    subset.uns["zero_pairs_window"] = 2
    subset.uns["zero_pairs_min_overlap"] = 1

    records = annotate_zero_hamming_segments(
        subset,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="segments",
        layer="nan0_0minus1",
        min_overlap=1,
        refine_segments=False,
    )
    per_read_segments = segments_to_per_read_dataframe(records, subset.var_names.to_numpy())
    assign_per_read_segments_layer(parent, subset, per_read_segments, layer_key="zero_span")

    assert "zero_span" in parent.layers
    target = parent.layers["zero_span"]
    assert target.shape == (2, 5)
    assert np.all(target[:, 1:4] == 1)
    assert np.all(target[:, [0, 4]] == 0)


def test_annotate_zero_hamming_segments_limits_overlap_and_count():
    adata = ad.AnnData(X=np.ones((2, 6)))
    adata.obs_names = ["read1", "read2"]
    adata.var_names = ["v0", "v1", "v2", "v3", "v4", "v5"]
    adata.layers["nan0_0minus1"] = np.ones((2, 6))
    adata.uns["zero_pairs"] = [np.array([[0, 1]]), np.array([[0, 1]])]
    adata.uns["zero_pairs_starts"] = np.array([0, 3])
    adata.uns["zero_pairs_window"] = 2
    adata.uns["zero_pairs_min_overlap"] = 1

    records = annotate_zero_hamming_segments(
        adata,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="segments",
        layer="nan0_0minus1",
        min_overlap=1,
        refine_segments=True,
        max_segments_per_read=2,
        max_segment_overlap=0,
    )

    assert len(records) == 1
    assert len(records[0]["segments"]) == 1


def test_annotate_zero_hamming_segments_overlap_counts_layer():
    parent = ad.AnnData(X=np.zeros((2, 6)))
    parent.obs_names = ["read1", "read2"]
    parent.var_names = ["v0", "v1", "v2", "v3", "v4", "v5"]

    subset = parent.copy()
    subset.layers["nan0_0minus1"] = np.ones((2, 6))
    subset.uns["zero_pairs"] = [np.array([[0, 1]]), np.array([[0, 1]])]
    subset.uns["zero_pairs_starts"] = np.array([0, 3])
    subset.uns["zero_pairs_window"] = 2
    subset.uns["zero_pairs_min_overlap"] = 1

    records = annotate_zero_hamming_segments(
        subset,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="segments",
        layer="nan0_0minus1",
        min_overlap=1,
        refine_segments=True,
        max_segments_per_read=2,
        max_segment_overlap=10,
    )
    per_read_segments = segments_to_per_read_dataframe(records, subset.var_names.to_numpy())
    assign_per_read_segments_layer(parent, subset, per_read_segments, layer_key="zero_span")

    target = parent.layers["zero_span"]
    assert np.all(target == 2)
