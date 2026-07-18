import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns

from smftools.plotting import combined_raw_clustermap


def _adata_with_span(n_positions=10):
    rng = np.random.default_rng(0)
    matrix = rng.random((2, n_positions))
    adata = ad.AnnData(X=np.zeros((2, n_positions)))
    adata.layers["nan0_0minus1"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    # read1 spans [2,5), read2 spans [6,8) -- union span is [2,8), 6 of the
    # 10 total positions.
    adata.obs["reference_start"] = [2, 6]
    adata.obs["reference_end"] = [5, 8]
    adata.var_names = [str(i) for i in range(n_positions)]
    adata.var["R1_C_site"] = [True] * n_positions
    adata.var["R1_GpC_site"] = [True] * n_positions
    adata.var["R1_CpG_site"] = [True] * n_positions
    adata.var["R1_A_site"] = [False] * n_positions
    return adata


def _capture_heatmap_shapes(monkeypatch, **kwargs):
    captured = []

    def fake_heatmap(matrix, *args, **fake_kwargs):
        captured.append(np.asarray(matrix).shape)
        return None

    monkeypatch.setattr(sns, "heatmap", fake_heatmap)
    combined_raw_clustermap(
        _adata_with_span(),
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=None,
        save_path=None,
        sort_by="gpc",
        fill_nan_strategy="value",
        fill_nan_value=0.0,
        **kwargs,
    )
    return captured


def test_restrict_to_read_span_crops_to_union_of_observed_spans(monkeypatch):
    captured = _capture_heatmap_shapes(monkeypatch, restrict_to_read_span=True)

    assert captured, "no heatmap panels were rendered"
    # Union span [2,8) covers 6 of the 10 positions, and every position is a
    # GpC/CpG/C site in this fixture, so every rendered panel should have 6
    # columns.
    for shape in captured:
        assert shape[1] == 6, f"expected 6 columns after span restriction, got {shape}"


def test_restrict_to_read_span_false_keeps_full_reference(monkeypatch):
    captured = _capture_heatmap_shapes(monkeypatch, restrict_to_read_span=False)

    assert captured, "no heatmap panels were rendered"
    for shape in captured:
        assert shape[1] == 10, f"expected the full 10 columns by default, got {shape}"
