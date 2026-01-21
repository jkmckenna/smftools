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
