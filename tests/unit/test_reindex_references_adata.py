import anndata as ad
import numpy as np
import pandas as pd

from smftools.preprocessing.reindex_references_adata import reindex_references_adata


def _adata(references=("refA", "refB")):
    n_vars = 6
    var_names = [str(i) for i in range(10, 10 + n_vars)]  # 10..15
    n_obs = len(references)
    X = np.zeros((n_obs, n_vars))
    obs = pd.DataFrame({"Reference_strand": list(references)})
    var = pd.DataFrame(index=var_names)
    return ad.AnnData(X=X, obs=obs, var=var)


def test_reindex_no_invert_matches_offset_only():
    adata = _adata()
    reindex_references_adata(adata, offsets={"refA": -12, "refB": 0})
    var_coords = adata.var_names.astype(int).to_numpy()
    np.testing.assert_array_equal(adata.var["refA_reindexed"].to_numpy(), var_coords - 12)
    np.testing.assert_array_equal(adata.var["refB_reindexed"].to_numpy(), var_coords)


def test_reindex_invert_true_flips_sign_for_single_ref():
    adata = _adata(references=("refA",))
    reindex_references_adata(adata, offsets={"refA": -12}, invert=True)
    var_coords = adata.var_names.astype(int).to_numpy()
    np.testing.assert_array_equal(adata.var["refA_reindexed"].to_numpy(), -(var_coords - 12))


def test_reindex_invert_true_applies_to_every_reference_without_listing_them():
    adata = _adata(references=("refA", "refB"))
    reindex_references_adata(adata, offsets={"refA": -12, "refB": 3}, invert=True)
    var_coords = adata.var_names.astype(int).to_numpy()
    np.testing.assert_array_equal(adata.var["refA_reindexed"].to_numpy(), -(var_coords - 12))
    np.testing.assert_array_equal(adata.var["refB_reindexed"].to_numpy(), -(var_coords + 3))


def test_reindex_invert_mixed_dict_flips_only_selected_ref():
    adata = _adata(references=("refA", "refB"))
    reindex_references_adata(
        adata,
        offsets={"refA": -12, "refB": -12},
        invert={"refA": True, "refB": False},
    )
    var_coords = adata.var_names.astype(int).to_numpy()
    np.testing.assert_array_equal(adata.var["refA_reindexed"].to_numpy(), -(var_coords - 12))
    np.testing.assert_array_equal(adata.var["refB_reindexed"].to_numpy(), var_coords - 12)


def test_reindex_invert_applies_to_identity_mapping_when_no_offset():
    adata = _adata(references=("refA",))
    reindex_references_adata(adata, offsets=None, invert=True)
    var_coords = adata.var_names.astype(int).to_numpy()
    np.testing.assert_array_equal(adata.var["refA_reindexed"].to_numpy(), -var_coords)


def test_reindex_invert_default_false_when_omitted():
    adata = _adata(references=("refA",))
    reindex_references_adata(adata, offsets={"refA": 5})
    var_coords = adata.var_names.astype(int).to_numpy()
    np.testing.assert_array_equal(adata.var["refA_reindexed"].to_numpy(), var_coords + 5)


def test_reindex_invert_bad_type_raises():
    adata = _adata(references=("refA",))
    try:
        reindex_references_adata(adata, offsets={"refA": 5}, invert="yes")
    except TypeError as exc:
        assert "invert" in str(exc)
    else:
        raise AssertionError("expected TypeError for invalid invert type")
