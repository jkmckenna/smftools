from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from smftools.plotting import plot_sequence_integer_encoding_clustermaps


def test_plot_sequence_integer_encoding_clustermaps_writes_files(tmp_path):
    matplotlib.use("Agg")

    matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 0, 4],
            [2, 3, 0, 1, 4],
            [3, 0, 1, 2, 4],
        ]
    )
    adata = ad.AnnData(X=np.zeros((4, 5)))
    adata.layers["sequence_integer_encoding"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S2", "S2"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R2", "R1", "R2"])
    adata.var_names = [f"pos{i}" for i in range(5)]
    adata.uns["sequence_integer_decoding_map"] = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

    results = plot_sequence_integer_encoding_clustermaps(
        adata,
        save_path=tmp_path,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
    )

    assert len(results) == 4
    for entry in results:
        assert entry["output_path"] is not None
        assert Path(entry["output_path"]).is_file()


def test_plot_sequence_integer_encoding_clustermaps_excludes_mod_sites(tmp_path):
    matplotlib.use("Agg")

    matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 0, 4],
            [2, 3, 0, 1, 4],
            [3, 0, 1, 2, 4],
        ]
    )
    adata = ad.AnnData(X=np.zeros((4, 5)))
    adata.layers["sequence_integer_encoding"] = matrix
    adata.layers["mismatch_integer_encoding"] = matrix
    adata.obs["Sample_Names"] = pd.Categorical(["S1", "S1", "S2", "S2"])
    adata.obs["Reference_strand"] = pd.Categorical(["R1", "R2", "R1", "R2"])
    adata.var_names = [f"pos{i}" for i in range(5)]
    adata.uns["sequence_integer_decoding_map"] = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

    adata.var["R1_GpC_site"] = np.array([True, False, True, False, False])
    adata.var["R2_GpC_site"] = np.array([False, True, False, True, False])
    adata.var["R1_ambiguous_GpC_CpG_site"] = np.array([False, False, False, False, True])
    adata.var["R2_ambiguous_GpC_CpG_site"] = np.array([False, False, False, False, True])
    adata.var["position_in_R1"] = np.ones(5, dtype=bool)
    adata.var["position_in_R2"] = np.ones(5, dtype=bool)

    results = plot_sequence_integer_encoding_clustermaps(
        adata,
        save_path=tmp_path,
        min_quality=None,
        min_length=None,
        min_mapped_length_to_reference_length_ratio=None,
        exclude_mod_sites=True,
        mod_site_bases=["GpC"],
    )

    assert len(results) == 4
    for entry in results:
        assert entry["output_path"] is not None
        assert entry["n_positions"] == 2
