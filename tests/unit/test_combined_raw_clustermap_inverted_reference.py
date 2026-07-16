import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns

from smftools.plotting import combined_raw_clustermap


def _adata_for_invert(invert: bool, n_positions: int = 6):
    var_names = [str(i) for i in range(10, 10 + n_positions)]  # 10..15
    var_coords = np.array([int(v) for v in var_names])
    # Column j's value equals its genomic position, so column order is
    # directly observable from the rendered matrix's row values.
    matrix = np.tile(var_coords.astype(float), (2, 1))
    adata = ad.AnnData(X=np.zeros((2, n_positions)))
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = var_names
    adata.var["R1_C_site"] = [True] * n_positions
    adata.var["R1_GpC_site"] = [True] * n_positions
    adata.var["R1_CpG_site"] = [False] * n_positions
    adata.var["R1_A_site"] = [False] * n_positions
    sign = -1 if invert else 1
    adata.var["R1_reindexed"] = sign * var_coords
    return adata


def _capture_first_heatmap(monkeypatch, invert: bool):
    captured = {}

    def fake_heatmap(matrix, *args, **kwargs):
        if "matrix" not in captured:
            captured["matrix"] = np.asarray(matrix)
        return None

    monkeypatch.setattr(sns, "heatmap", fake_heatmap)
    combined_raw_clustermap(
        _adata_for_invert(invert),
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=None,
        sort_by="none",
        fill_nan_strategy="value",
        fill_nan_value=0.0,
        index_col_suffix="reindexed",
    )
    assert "matrix" in captured, "no heatmap panel was rendered"
    return captured["matrix"]


def test_non_inverted_reference_keeps_ascending_column_order(monkeypatch):
    matrix = _capture_first_heatmap(monkeypatch, invert=False)
    # Row values equal genomic position; non-inverted reindexing matches
    # physical (ascending) var order, so columns read ascending left-to-right.
    np.testing.assert_array_equal(matrix[0], [10, 11, 12, 13, 14, 15])


def test_inverted_reference_reverses_column_order(monkeypatch):
    matrix = _capture_first_heatmap(monkeypatch, invert=True)
    # Sign-flipped reindexed values are descending in physical order; sorting
    # ascending by that value reverses the rendered columns.
    np.testing.assert_array_equal(matrix[0], [15, 14, 13, 12, 11, 10])
