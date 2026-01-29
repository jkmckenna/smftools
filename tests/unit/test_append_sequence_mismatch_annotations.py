import anndata as ad
import pandas as pd

from smftools.preprocessing.append_sequence_mismatch_annotations import (
    append_sequence_mismatch_annotations,
)


def test_append_sequence_mismatch_annotations_adds_expected_vars() -> None:
    var = pd.DataFrame(
        {
            "seq_a": ["ACGT", "ACGT"],
            "seq_b": ["ACCT", "ACGTT"],
        }
    )
    adata = ad.AnnData(X=[[0], [0]], var=var)

    append_sequence_mismatch_annotations(adata, seq1_column="seq_a", seq2_column="seq_b")

    positions = adata.var["seq_a__seq_b_mismatch_positions"].tolist()
    types = adata.var["seq_a__seq_b_mismatch_types"].tolist()
    identities = adata.var["seq_a__seq_b_mismatch_identities"].tolist()

    assert positions == [[(2, 2)], [(None, 4)]]
    assert types == [["substitution"], ["insertion"]]
    assert identities == [["G->C"], ["ins:T"]]
