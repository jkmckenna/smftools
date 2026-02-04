import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.append_sequence_mismatch_annotations import (
    append_sequence_mismatch_annotations,
)


def test_append_sequence_mismatch_annotations_adds_expected_vars() -> None:
    var = pd.DataFrame(
        {
            "seq_a": ["A", "C", "N", "G", "T"],
            "seq_b": ["A", "C", "C", "G", "T"],
        }
    )
    adata = ad.AnnData(X=np.zeros((1, 5)), var=var)

    append_sequence_mismatch_annotations(adata, seq1_column="seq_a", seq2_column="seq_b")

    types = adata.var["seq_a__seq_b_mismatch_type"].tolist()
    identities = adata.var["seq_a__seq_b_mismatch_identity"].tolist()
    flags = adata.var["seq_a__seq_b_is_mismatch"].tolist()

    assert types.count("insertion") == 1
    assert identities.count("ins:C") == 1
    assert sum(flags) == 1
