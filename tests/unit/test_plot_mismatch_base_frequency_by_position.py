from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from smftools.plotting import plot_mismatch_base_frequency_by_position


def test_plot_mismatch_base_frequency_by_position_writes_files(tmp_path):
    matplotlib.use("Agg")

    mismatch_matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 0, 4],
            [2, 3, 0, 1, 5],
            [3, 0, 1, 2, 5],
        ]
    )
    adata = ad.AnnData(X=np.zeros((4, 5)))
    adata.layers["mismatch_integer_encoding"] = mismatch_matrix
    adata.layers["read_span_mask"] = np.ones_like(mismatch_matrix)
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S2", "S2"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R2", "R1", "R2"])
    adata.var_names = [f"pos{i}" for i in range(5)]
    adata.uns["mismatch_integer_encoding_map"] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": 4,
        "PAD": 5,
    }

    results = plot_mismatch_base_frequency_by_position(adata, save_path=tmp_path)

    assert len(results) == 4
    for entry in results:
        assert entry["output_path"] is not None
        assert Path(entry["output_path"]).is_file()


def test_plot_mismatch_base_frequency_by_position_excludes_mod_sites(tmp_path):
    matplotlib.use("Agg")

    mismatch_matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 0, 4],
            [2, 3, 0, 1, 5],
            [3, 0, 1, 2, 5],
        ]
    )
    adata = ad.AnnData(X=np.zeros((4, 5)))
    adata.layers["mismatch_integer_encoding"] = mismatch_matrix
    adata.layers["read_span_mask"] = np.ones_like(mismatch_matrix)
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S2", "S2"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R2", "R1", "R2"])
    adata.var_names = [f"pos{i}" for i in range(5)]
    adata.uns["mismatch_integer_encoding_map"] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": 4,
        "PAD": 5,
    }
    adata.var["R1_GpC_site"] = np.array([True, False, True, False, True])
    adata.var["R2_GpC_site"] = np.array([False, True, False, True, True])
    adata.var["position_in_R1"] = np.ones(5, dtype=bool)
    adata.var["position_in_R2"] = np.ones(5, dtype=bool)

    results = plot_mismatch_base_frequency_by_position(
        adata,
        save_path=tmp_path,
        exclude_mod_sites=True,
        mod_site_bases=["GpC"],
    )

    assert len(results) == 4
    for entry in results:
        assert entry["n_positions"] == 2


def test_plot_mismatch_base_frequency_by_position_ignores_strand_base_mismatches(tmp_path):
    matplotlib.use("Agg")

    mismatch_matrix = np.array(
        [
            [1, 0, 2],
            [1, 3, 0],
            [0, 2, 1],
            [3, 2, 0],
        ]
    )
    adata = ad.AnnData(X=np.zeros((4, 3)))
    adata.layers["mismatch_integer_encoding"] = mismatch_matrix
    adata.layers["read_span_mask"] = np.ones_like(mismatch_matrix)
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S1", "S1"])
    adata.obs["Reference_strand"] = pd.Categorical(
        ["ref_top", "ref_top", "ref_bottom", "ref_bottom"]
    )
    adata.var_names = [f"pos{i}" for i in range(3)]
    adata.var["Reference_strand_FASTA_sequence_base"] = pd.Series(["C", "G", "A"])
    adata.uns["mismatch_integer_encoding_map"] = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": 4,
        "PAD": 5,
    }

    results = plot_mismatch_base_frequency_by_position(adata, save_path=tmp_path)

    assert len(results) == 2
    for entry in results:
        assert entry["n_positions"] == 2
