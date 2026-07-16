from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from smftools.plotting import plot_variant_segment_clustermaps, variant_plotting


def test_plot_variant_segment_clustermaps_with_mismatch_type_annotation(tmp_path):
    matplotlib.use("Agg")

    seq1_col = "refA_top_strand_FASTA_base"
    seq2_col = "refB_top_strand_FASTA_base"
    prefix = f"{seq1_col}__{seq2_col}"
    segment_layer = f"{prefix}_variant_segments"
    call_layer = f"{prefix}_variant_call"

    adata = ad.AnnData(X=np.zeros((4, 6)))
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["refA_top", "refA_top", "refA_top", "refA_top"])
    adata.obs["chimeric_variant_sites_type"] = pd.Categorical(
        [
            "no_segment_mismatch",
            "left_segment_mismatch",
            "middle_segment_mismatch",
            "multi_segment_mismatch",
        ]
    )
    adata.var_names = [f"p{i}" for i in range(6)]
    adata.layers["read_span_mask"] = np.ones((4, 6), dtype=np.int8)
    adata.layers[segment_layer] = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 3, 1, 1, 1],
            [1, 1, 2, 2, 1, 1],
            [2, 1, 1, 2, 1, 1],
        ],
        dtype=np.int8,
    )
    adata.layers[call_layer] = np.array(
        [
            [1, 1, -1, -1, -1, -1],
            [2, 2, -1, 1, -1, -1],
            [1, -1, 2, 2, 1, -1],
            [2, 1, -1, 2, 1, -1],
        ],
        dtype=np.int8,
    )

    results = plot_variant_segment_clustermaps(
        adata,
        seq1_column=seq1_col,
        seq2_column=seq2_col,
        sample_col="Sample_Names",
        reference_col="Reference_strand",
        variant_segment_layer=segment_layer,
        read_span_layer="read_span_mask",
        save_path=tmp_path,
        mismatch_type_obs_col="chimeric_variant_sites_type",
        mismatch_type_legend_prefix="UMI content",
    )

    assert len(results) == 1
    assert results[0]["output_path"] is not None
    assert Path(results[0]["output_path"]).is_file()


def test_plot_variant_segment_clustermaps_reorders_for_inverted_reference(monkeypatch, tmp_path):
    matplotlib.use("Agg")

    seq1_col = "refA_top_strand_FASTA_base"
    seq2_col = "refB_top_strand_FASTA_base"
    prefix = f"{seq1_col}__{seq2_col}"
    segment_layer = f"{prefix}_variant_segments"

    # Column j's value equals its column index (offset by 1, since 0 is the
    # "no coverage" sentinel for the seg_matrix colormap), so column order is
    # directly observable from the captured matrix.
    adata = ad.AnnData(X=np.zeros((2, 4)))
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["refA_top", "refA_top"])
    var_coords = np.array([10, 11, 12, 13])
    adata.var_names = [str(c) for c in var_coords]
    adata.var["refA_top_reindexed"] = -var_coords  # inverted, anchored at 0
    adata.layers["read_span_mask"] = np.ones((2, 4), dtype=np.int8)
    adata.layers[segment_layer] = np.tile(np.array([1, 1, 2, 2]), (2, 1)).astype(np.int8)

    captured = {}

    def fake_heatmap(matrix, *args, **kwargs):
        if "matrix" not in captured:
            captured["matrix"] = np.asarray(matrix)
        return None

    monkeypatch.setattr(variant_plotting.sns, "heatmap", fake_heatmap)

    plot_variant_segment_clustermaps(
        adata,
        seq1_column=seq1_col,
        seq2_column=seq2_col,
        sample_col="Sample_Names",
        reference_col="Reference_strand",
        variant_segment_layer=segment_layer,
        read_span_layer="read_span_mask",
        save_path=tmp_path,
        index_col_suffix="reindexed",
    )

    assert "matrix" in captured
    np.testing.assert_array_equal(captured["matrix"][0], [2, 2, 1, 1])
