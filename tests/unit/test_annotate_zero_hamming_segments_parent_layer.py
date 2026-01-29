import anndata as ad
import numpy as np

from smftools.tools.rolling_nn_distance import annotate_zero_hamming_segments


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

    annotate_zero_hamming_segments(
        subset,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="segments",
        layer="nan0_0minus1",
        min_overlap=1,
        refine_segments=False,
        binary_layer_key="zero_span",
        parent_adata=parent,
    )

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

    annotate_zero_hamming_segments(
        subset,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="segments",
        layer="nan0_0minus1",
        min_overlap=1,
        refine_segments=False,
        binary_layer_key="zero_span",
        parent_adata=parent,
    )

    assert "zero_span" in parent.layers
    target = parent.layers["zero_span"]
    assert target.shape == (2, 5)
    assert np.all(target[:, 1:4] == 1)
    assert np.all(target[:, [0, 4]] == 0)
