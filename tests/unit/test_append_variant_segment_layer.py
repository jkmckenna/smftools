import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.append_variant_call_layer import append_variant_segment_layer


def test_append_variant_segment_layer_adds_chimeric_variant_site_labels() -> None:
    seq1_col = "refA_top_strand_FASTA_base"
    seq2_col = "refB_top_strand_FASTA_base"
    output_prefix = f"{seq1_col}__{seq2_col}"
    layer_name = f"{output_prefix}_variant_call"

    n_obs, n_vars = 7, 10
    variant_call = np.full((n_obs, n_vars), -1, dtype=np.int8)
    read_span = np.ones((n_obs, n_vars), dtype=np.int8)

    # r0: non-chimeric (all informative calls match own seq1 reference)
    variant_call[0, 2] = 1
    variant_call[0, 7] = 1

    # r1: full-span mismatch to own seq1 reference -> left segment mismatch
    variant_call[1, 4] = 2

    # r2: mismatch on right side for seq1 reference
    variant_call[2, 2] = 1
    variant_call[2, 7] = 2

    # r3: mismatch in the middle for seq1 reference
    variant_call[3, 1] = 1
    variant_call[3, 4] = 2
    variant_call[3, 8] = 1

    # r4: multiple mismatch segments for seq1 reference
    variant_call[4, 1] = 1
    variant_call[4, 3] = 2
    variant_call[4, 5] = 1
    variant_call[4, 7] = 2
    variant_call[4, 9] = 1

    # r5: left mismatch for seq2 reference (mismatch value is 1 for seq2-aligned reads)
    variant_call[5, 2] = 1
    variant_call[5, 7] = 2

    # r6: non-chimeric for seq2 reference
    variant_call[6, 2] = 2
    variant_call[6, 7] = 2

    obs = pd.DataFrame(
        {
            "Reference_strand": [
                "refA_top",
                "refA_top",
                "refA_top",
                "refA_top",
                "refA_top",
                "refB_top",
                "refB_top",
            ]
        }
    )
    obs["Reference_strand"] = obs["Reference_strand"].astype("category")
    var = pd.DataFrame(index=[str(i) for i in range(n_vars)])
    adata = ad.AnnData(X=np.zeros((n_obs, n_vars)), obs=obs, var=var)
    adata.layers[layer_name] = variant_call
    adata.layers["read_span_mask"] = read_span

    append_variant_segment_layer(
        adata,
        seq1_column=seq1_col,
        seq2_column=seq2_col,
        variant_call_layer=layer_name,
        read_span_layer="read_span_mask",
        reference_col="Reference_strand",
    )

    assert adata.obs["chimeric_variant_sites"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    observed_types = adata.obs["chimeric_variant_sites_type"].tolist()
    assert observed_types[0] == "no_segment_mismatch"
    assert observed_types[1] == "left_segment_mismatch"
    assert observed_types[2] == "right_segment_mismatch"
    assert observed_types[3] == "middle_segment_mismatch"
    assert observed_types[4] == "multi_segment_mismatch"
    assert observed_types[5] == "left_segment_mismatch"
    assert observed_types[6] == "no_segment_mismatch"
