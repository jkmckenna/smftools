import anndata as ad
import numpy as np
import pandas as pd

from smftools.tools.rolling_nn_distance import (
    annotate_zero_hamming_segments,
    assign_per_read_segments_layer,
    assign_rolling_nn_results,
    rolling_window_nn_distance,
    rolling_window_nn_distance_by_group,
    select_top_segments_per_read,
    zero_hamming_segments_to_dataframe,
    zero_pairs_to_dataframe,
)


def test_rolling_window_nn_distance_basic():
    X = np.array(
        [
            [1.0, 1.0, 0.0, np.nan],
            [1.0, 0.0, 0.0, 1.0],
            [np.nan, 1.0, 1.0, 1.0],
        ]
    )
    adata = ad.AnnData(X)

    distances, starts = rolling_window_nn_distance(
        adata,
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        store_obsm="rolling_nn_dist",
    )

    assert starts.tolist() == [0]
    assert distances.shape == (3, 1)

    expected = np.array([[1.0 / 3.0], [1.0 / 3.0], [0.5]])
    np.testing.assert_allclose(distances, expected, rtol=1e-6)

    assert "rolling_nn_dist" in adata.obsm
    assert "rolling_nn_dist_starts" in adata.uns
    assert adata.uns["rolling_nn_dist_centers"].tolist() == [1.5]
    assert adata.uns["rolling_nn_dist_window"] == 4
    assert adata.uns["rolling_nn_dist_min_overlap"] == 2


def test_rolling_window_nn_distance_by_group_matches_subset():
    X = np.array(
        [
            [1.0, 1.0, 0.0, np.nan],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [np.nan, 1.0, 1.0, 0.0],
        ]
    )
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(["s1", "s1", "s2", "s2"]),
            "reference": pd.Categorical(["r1", "r1", "r1", "r1"]),
        }
    )
    adata = ad.AnnData(X, obs=obs)

    distances, starts = rolling_window_nn_distance_by_group(
        adata,
        group_cols=["sample", "reference"],
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        store_obsm="rolling_nn_group",
    )

    assert starts.tolist() == [0]
    assert distances.shape == (4, 1)

    s1_mask = adata.obs["sample"] == "s1"
    s2_mask = adata.obs["sample"] == "s2"

    expected_s1, _ = rolling_window_nn_distance(
        adata[s1_mask],
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        store_obsm=None,
    )
    expected_s2, _ = rolling_window_nn_distance(
        adata[s2_mask],
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        store_obsm=None,
    )

    np.testing.assert_allclose(distances[s1_mask.to_numpy()], expected_s1, rtol=1e-6)
    np.testing.assert_allclose(distances[s2_mask.to_numpy()], expected_s2, rtol=1e-6)

    assert "rolling_nn_group" in adata.obsm
    assert adata.uns["rolling_nn_group_group_cols"] == ["sample", "reference"]


def test_rolling_window_nn_distance_site_types_subset():
    X = np.array(
        [
            [1.0, 0.0, 1.0, np.nan],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, np.nan, 1.0, 0.0],
        ]
    )
    var = pd.DataFrame(
        {
            "GpC_site": [True, False, True, False],
            "CpG_site": [False, True, False, True],
        }
    )
    adata = ad.AnnData(X, var=var)

    distances, starts = rolling_window_nn_distance(
        adata,
        window=2,
        step=2,
        min_overlap=1,
        return_fraction=True,
        store_obsm="rolling_nn_sites",
        site_types=["GpC"],
    )

    assert starts.tolist() == [0]
    assert distances.shape == (3, 1)
    assert adata.uns["rolling_nn_sites_var_indices"].tolist() == [0, 2]


def test_rolling_window_nn_distance_site_types_string():
    X = np.array(
        [
            [1.0, 0.0, 1.0, np.nan],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, np.nan, 1.0, 0.0],
        ]
    )
    var = pd.DataFrame(
        {
            "GpC_site": [True, False, True, False],
            "CpG_site": [False, True, False, True],
        }
    )
    adata = ad.AnnData(X, var=var)

    distances, starts = rolling_window_nn_distance(
        adata,
        window=2,
        step=2,
        min_overlap=1,
        return_fraction=True,
        store_obsm="rolling_nn_sites",
        site_types="GpC",
    )

    assert starts.tolist() == [0]
    assert distances.shape == (3, 1)
    assert adata.uns["rolling_nn_sites_var_indices"].tolist() == [0, 2]


def test_rolling_window_nn_distance_reference_site_mask():
    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    var = pd.DataFrame(
        {
            "C_site": [True, True, True],
            "ref1_C_site": [True, False, True],
            "ref2_C_site": [False, True, False],
        }
    )
    adata = ad.AnnData(X, var=var)

    distances, starts = rolling_window_nn_distance(
        adata,
        window=1,
        step=1,
        min_overlap=1,
        return_fraction=True,
        store_obsm="rolling_nn_ref",
        site_types=["C"],
        reference="ref2",
    )

    assert starts.tolist() == [0]
    assert distances.shape == (2, 1)
    assert adata.uns["rolling_nn_ref_var_indices"].tolist() == [2]


def test_assign_rolling_nn_results_to_parent():
    X = np.array(
        [
            [1.0, 1.0, 0.0, np.nan],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [np.nan, 1.0, 1.0, 0.0],
        ]
    )
    parent = ad.AnnData(X)
    subset = parent[[0, 2]].copy()

    values, starts = rolling_window_nn_distance(
        subset,
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        store_obsm=None,
    )

    assign_rolling_nn_results(
        parent,
        subset,
        values,
        starts,
        obsm_key="rolling_nn_parent",
        window=4,
        step=4,
        min_overlap=2,
        return_fraction=True,
        layer=None,
    )

    assert parent.obsm["rolling_nn_parent"].shape == (4, 1)
    np.testing.assert_allclose(parent.obsm["rolling_nn_parent"][[0, 2], :], values, rtol=1e-6)
    assert np.isnan(parent.obsm["rolling_nn_parent"][[1, 3], :]).all()
    assert parent.uns["rolling_nn_parent_starts"].tolist() == [0]
    assert parent.uns["rolling_nn_parent_centers"].tolist() == [1.5]
    assert parent.uns["rolling_nn_parent_window"] == 4


def test_zero_pairs_and_segments_to_dataframe():
    X = np.array(
        [
            [1.0, 1.0, np.nan, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    adata = ad.AnnData(X)

    rolling_window_nn_distance(
        adata,
        window=2,
        step=2,
        min_overlap=2,
        return_fraction=True,
        store_obsm="rolling_nn_dist",
        collect_zero_pairs=True,
        zero_pairs_uns_key="rolling_nn_zero_pairs",
    )

    pairs_df = zero_pairs_to_dataframe(adata, "rolling_nn_zero_pairs")
    assert not pairs_df.empty
    assert set(pairs_df.columns) == {
        "window_index",
        "window_start",
        "window_end",
        "read_i",
        "read_j",
        "read_i_name",
        "read_j_name",
    }

    records = annotate_zero_hamming_segments(
        adata,
        zero_pairs_uns_key="rolling_nn_zero_pairs",
        output_uns_key="zero_hamming_segments",
        layer=None,
        min_overlap=2,
        refine_segments=True,
    )

    segments_df = zero_hamming_segments_to_dataframe(records, adata.var_names.to_numpy())
    assert set(segments_df.columns) == {
        "read_i",
        "read_j",
        "read_i_name",
        "read_j_name",
        "segment_start",
        "segment_end_exclusive",
        "segment_end_inclusive",
        "segment_start_label",
        "segment_end_label",
    }


def test_zero_pairs_collection_and_annotation():
    X = np.array(
        [
            [1.0, 0.0, 1.0, np.nan],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    adata = ad.AnnData(X)

    rolling_window_nn_distance(
        adata,
        window=3,
        step=1,
        min_overlap=2,
        return_fraction=True,
        store_obsm="rolling_nn_dist",
        collect_zero_pairs=True,
        zero_pairs_uns_key="rolling_nn_zero_pairs",
    )

    assert "rolling_nn_zero_pairs" in adata.uns
    assert len(adata.uns["rolling_nn_zero_pairs"]) == 2
    first_window_pairs = adata.uns["rolling_nn_zero_pairs"][0]
    assert first_window_pairs.shape == (1, 2)
    assert np.array_equal(first_window_pairs[0], np.array([0, 1]))

    records = annotate_zero_hamming_segments(
        adata,
        output_uns_key="zero_hamming_segments",
        refine_segments=True,
    )

    assert records
    record = records[0]
    assert record["read_i"] == 0
    assert record["read_j"] == 1
    assert record["segments"] == [(0, 4)]


def test_zero_pairs_merge_gap_combines_adjacent_segments():
    X = np.zeros((2, 6), dtype=float)
    adata = ad.AnnData(X)
    adata.uns["zero_pairs"] = [
        np.array([[0, 1]]),
        np.array([[0, 1]]),
    ]
    adata.uns["zero_pairs_starts"] = np.array([0, 3])
    adata.uns["zero_pairs_window"] = 2
    adata.uns["zero_pairs_min_overlap"] = 1

    records = annotate_zero_hamming_segments(
        adata,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="zero_hamming_segments",
        refine_segments=False,
        merge_gap=1,
    )

    assert records
    assert records[0]["segments"] == [(0, 5)]


def test_zero_pairs_refine_stops_after_nan_run():
    X = np.array(
        [
            [1.0, 1.0, np.nan, np.nan, 1.0, 1.0],
            [1.0, 1.0, np.nan, np.nan, 1.0, 1.0],
        ]
    )
    adata = ad.AnnData(X)
    adata.uns["zero_pairs"] = [np.array([[0, 1]])]
    adata.uns["zero_pairs_starts"] = np.array([0])
    adata.uns["zero_pairs_window"] = 2
    adata.uns["zero_pairs_min_overlap"] = 1

    records = annotate_zero_hamming_segments(
        adata,
        zero_pairs_uns_key="zero_pairs",
        output_uns_key="zero_hamming_segments",
        refine_segments=True,
        max_nan_run=2,
    )

    assert records
    assert records[0]["segments"] == [(0, 3)]


def test_zero_pairs_annotation_to_parent_layer():
    X = np.array(
        [
            [1.0, 0.0, 1.0, np.nan],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    parent = ad.AnnData(X)
    subset = parent[[0, 1]].copy()

    rolling_window_nn_distance(
        subset,
        window=3,
        step=1,
        min_overlap=2,
        return_fraction=True,
        store_obsm=None,
        collect_zero_pairs=True,
        zero_pairs_uns_key="rolling_nn_zero_pairs",
    )

    annotate_zero_hamming_segments(
        subset,
        zero_pairs_uns_key="rolling_nn_zero_pairs",
        output_uns_key="zero_hamming_segments",
        binary_layer_key="zero_span",
        parent_adata=parent,
    )

    assert "zero_span" in parent.layers
    expected = np.zeros_like(X, dtype=np.uint8)
    expected[0, 0:4] = 1
    expected[1, 0:4] = 1
    np.testing.assert_array_equal(parent.layers["zero_span"], expected)


def test_select_top_segments_per_read_limits_overlap_and_partners():
    records = [
        {
            "read_i": 0,
            "read_j": 1,
            "read_i_name": "r0",
            "read_j_name": "r1",
            "segments": [(0, 5), (12, 15)],
        },
        {
            "read_i": 0,
            "read_j": 2,
            "read_i_name": "r0",
            "read_j_name": "r2",
            "segments": [(6, 11)],
        },
    ]
    var_names = np.array([str(i) for i in range(20)])
    raw_df, filtered_df = select_top_segments_per_read(
        records,
        var_names,
        max_segments_per_read=2,
        max_segment_overlap=0,
        min_span=3,
    )

    assert not raw_df.empty
    assert not filtered_df.empty
    read0 = filtered_df[filtered_df["read_id"] == 0]
    assert set(read0["partner_id"]) == {1, 2}
    assert read0["selection_rank"].max() == 2


def test_assign_per_read_segments_layer_uses_label_spans():
    parent = ad.AnnData(X=np.zeros((1, 5)))
    parent.obs_names = ["read1"]
    parent.var_names = ["10", "11", "12", "13", "14"]
    subset = parent.copy()

    per_read_segments = pd.DataFrame(
        [
            {
                "read_id": 0,
                "segment_start": 0,
                "segment_end_exclusive": 1,
                "segment_start_label": "11",
                "segment_end_label": "13",
            }
        ]
    )

    assign_per_read_segments_layer(parent, subset, per_read_segments, layer_key="top_segments")

    expected = np.zeros((1, 5), dtype=np.uint16)
    expected[0, 1:4] = 1
    np.testing.assert_array_equal(parent.layers["top_segments"], expected)
