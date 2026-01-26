import numpy as np
import pandas as pd
import anndata as ad

from smftools.preprocessing.append_mismatch_frequency_sites import append_mismatch_frequency_sites


def test_append_mismatch_frequency_sites_adds_expected_vars():
    mismatch_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "PAD": 5}
    mismatch_matrix = np.array(
        [
            [0, 4, 4],  # ref1 read1: A mismatch at pos0
            [4, 1, 4],  # ref1 read2: C mismatch at pos1
            [4, 4, 3],  # ref2 read1: T mismatch at pos2
            [4, 4, 4],  # ref2 read2: no mismatches
        ],
        dtype=np.int16,
    )
    read_span_mask = np.ones_like(mismatch_matrix, dtype=np.int16)

    obs = pd.DataFrame({"Reference_strand": ["ref1", "ref1", "ref2", "ref2"]})
    obs["Reference_strand"] = obs["Reference_strand"].astype("category")
    var = pd.DataFrame(index=["p0", "p1", "p2"])
    var["position_in_ref1"] = [True, True, False]
    var["position_in_ref2"] = [True, True, True]

    adata = ad.AnnData(X=np.zeros(mismatch_matrix.shape), obs=obs, var=var)
    adata.layers["mismatch_integer_encoding"] = mismatch_matrix
    adata.layers["read_span_mask"] = read_span_mask
    adata.uns["mismatch_integer_encoding_map"] = mismatch_map

    append_mismatch_frequency_sites(
        adata,
        ref_column="Reference_strand",
        mismatch_layer="mismatch_integer_encoding",
        read_span_layer="read_span_mask",
        mismatch_frequency_range=(0.4, 0.6),
    )

    assert "ref1_mismatch_frequency" in adata.var
    assert "ref1_variable_sequence_site" in adata.var
    assert "ref1_mismatch_base_frequencies" in adata.var
    assert "ref2_mismatch_frequency" in adata.var

    np.testing.assert_allclose(
        adata.var["ref1_mismatch_frequency"].values,
        np.array([0.5, 0.5, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        adata.var["ref2_mismatch_frequency"].values,
        np.array([0.0, 0.0, 0.5]),
        equal_nan=True,
    )

    assert adata.var["ref1_variable_sequence_site"].tolist() == [True, True, False]
    assert adata.var["ref2_variable_sequence_site"].tolist() == [False, False, True]

    assert adata.var["ref1_mismatch_base_frequencies"].iloc[0] == [("A", 0.5)]
    assert adata.var["ref1_mismatch_base_frequencies"].iloc[1] == [("C", 0.5)]
    assert adata.var["ref1_mismatch_base_frequencies"].iloc[2] == []
    assert adata.var["ref2_mismatch_base_frequencies"].iloc[2] == [("T", 0.5)]
