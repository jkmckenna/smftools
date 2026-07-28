import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.append_variant_call_layer import append_variant_call_layer


def test_append_variant_call_layer_preserves_shifted_legacy_coordinates() -> None:
    seq1_column = "refA_top_strand_FASTA_base"
    seq2_column = "refB_top_strand_FASTA_base"
    prefix = f"{seq1_column}__{seq2_column}"
    obs = pd.DataFrame({"Reference_strand": pd.Categorical(["refA_top", "refB_top"])})
    adata = ad.AnnData(
        X=np.zeros((2, 4)),
        obs=obs,
        var=pd.DataFrame(index=["0", "1", "2", "3"]),
    )
    adata.layers["sequence_integer_encoding"] = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 4, 4],
        ],
        dtype=np.int8,
    )
    adata.layers["read_span_mask"] = np.ones((2, 4), dtype=np.int8)
    adata.uns["mismatch_integer_encoding_map"] = {
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "N": 0,
        "PAD": -1,
    }
    adata.uns[f"{prefix}_substitution_map"] = pd.DataFrame(
        {
            "seq1_var_idx": [1],
            "seq2_var_idx": [2],
            "seq1_base": ["C"],
            "seq2_base": ["T"],
        }
    )

    append_variant_call_layer(adata, seq1_column, seq2_column)

    assert adata.layers[f"{prefix}_variant_call"].tolist() == [
        [-1, 1, -1, -1],
        [-1, -1, 2, -1],
    ]
