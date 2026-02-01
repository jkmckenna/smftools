import anndata as ad
import numpy as np
import pandas as pd

from smftools.cli.chimeric_adata import _build_zero_hamming_span_layer_from_obs


def test_build_zero_hamming_span_layer_from_obs_counts_spans():
    adata = ad.AnnData(
        X=np.zeros((2, 4)),
        obs=pd.DataFrame(index=["read1", "read2"]),
        var=pd.DataFrame(index=["1", "2", "3", "4"]),
    )
    adata.obs["span_obs"] = [
        [(1, 2, "partner")],
        [(2, 4, "partner")],
    ]

    _build_zero_hamming_span_layer_from_obs(
        adata=adata,
        obs_key="span_obs",
        layer_key="zero_hamming_distance_spans",
    )

    expected = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(adata.layers["zero_hamming_distance_spans"], expected)


def test_build_zero_hamming_span_layer_from_obs_accumulates_counts():
    adata = ad.AnnData(
        X=np.zeros((1, 3)),
        obs=pd.DataFrame(index=["read1"]),
        var=pd.DataFrame(index=["10", "11", "12"]),
    )
    adata.obs["span_obs"] = [[(10, 11, "partner")]]

    _build_zero_hamming_span_layer_from_obs(
        adata=adata,
        obs_key="span_obs",
        layer_key="zero_hamming_distance_spans",
    )
    _build_zero_hamming_span_layer_from_obs(
        adata=adata,
        obs_key="span_obs",
        layer_key="zero_hamming_distance_spans",
    )

    expected = np.array([[2, 2, 0]], dtype=np.uint16)
    np.testing.assert_array_equal(adata.layers["zero_hamming_distance_spans"], expected)


def test_build_zero_hamming_span_layer_from_obs_skips_non_numeric_vars():
    adata = ad.AnnData(
        X=np.zeros((1, 2)),
        obs=pd.DataFrame(index=["read1"]),
        var=pd.DataFrame(index=["a", "b"]),
    )
    adata.obs["span_obs"] = [[(1, 2, "partner")]]

    _build_zero_hamming_span_layer_from_obs(
        adata=adata,
        obs_key="span_obs",
        layer_key="zero_hamming_distance_spans",
    )

    assert "zero_hamming_distance_spans" not in adata.layers
