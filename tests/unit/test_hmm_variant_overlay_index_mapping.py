import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting import hmm_plotting


def _build_test_adata(layer_name: str) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((2, 5)))
    adata.layers[layer_name] = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    adata.layers["nan0_0minus1"] = np.asarray(adata.layers[layer_name]).copy()
    adata.layers["seq1__seq2_variant_call"] = np.zeros((2, 5))
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R1"])
    adata.var_names = ["100", "101", "102", "103", "104"]
    adata.var["R1_valid_fraction"] = [0.1, 0.9, 0.2, 0.95, 0.99]
    adata.var["R1_C_site"] = [False, True, False, True, False]
    return adata


def test_combined_hmm_raw_clustermap_overlay_uses_global_var_indices(tmp_path, monkeypatch):
    adata = _build_test_adata("hmm_combined")
    captured_panels: list[tuple[object, np.ndarray]] = []
    captured_kwargs = {}

    def fake_overlay(*args, **kwargs):
        captured_panels.extend(args[3])
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(hmm_plotting, "_apply_variant_overlay_np", fake_overlay)

    hmm_plotting.combined_hmm_raw_clustermap(
        adata,
        save_path=tmp_path,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=0.8,
        sort_by="none",
        overlay_variant_calls=True,
    )

    assert len(captured_panels) == 2
    np.testing.assert_array_equal(np.asarray(captured_panels[0][1]), np.array([1, 3, 4]))
    np.testing.assert_array_equal(np.asarray(captured_panels[1][1]), np.array([1, 3]))
    assert captured_kwargs["marker_size"] == 4.0


def test_combined_hmm_length_clustermap_overlay_uses_global_var_indices(tmp_path, monkeypatch):
    adata = _build_test_adata("GpC_all_accessible_features_lengths")
    captured_panels: list[tuple[object, np.ndarray]] = []
    captured_kwargs = {}

    def fake_overlay(*args, **kwargs):
        captured_panels.extend(args[3])
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(hmm_plotting, "_apply_variant_overlay_np", fake_overlay)

    hmm_plotting.combined_hmm_length_clustermap(
        adata,
        length_layer="GpC_all_accessible_features_lengths",
        save_path=tmp_path,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        min_position_valid_fraction=0.8,
        sort_by="none",
        overlay_variant_calls=True,
    )

    assert len(captured_panels) == 2
    np.testing.assert_array_equal(np.asarray(captured_panels[0][1]), np.array([1, 3, 4]))
    np.testing.assert_array_equal(np.asarray(captured_panels[1][1]), np.array([1, 3]))
    assert captured_kwargs["marker_size"] == 4.0
