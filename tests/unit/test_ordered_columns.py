import anndata as ad
import numpy as np
import pandas as pd

from smftools.plotting.plotting_utils import _ordered_columns, _select_labels


def _adata_with_reindexed(ref="refA", n_vars=6, invert=False):
    var_names = [str(i) for i in range(10, 10 + n_vars)]  # 10..15
    var_coords = np.arange(10, 10 + n_vars)
    reindexed = -var_coords if invert else var_coords
    var = pd.DataFrame({f"{ref}_reindexed": reindexed}, index=var_names)
    obs = pd.DataFrame({"Reference_strand": [ref]})
    X = np.zeros((1, n_vars))
    return ad.AnnData(X=X, obs=obs, var=var)


def test_ordered_columns_identity_when_no_index_col_suffix():
    adata = _adata_with_reindexed()
    sites = np.array([0, 1, 2, 3, 4, 5])
    sites_out, labels_out = _ordered_columns(adata, sites, "refA", None)
    expected_labels = _select_labels(adata, sites, "refA", None)
    np.testing.assert_array_equal(sites_out, sites)
    np.testing.assert_array_equal(labels_out, expected_labels)


def test_ordered_columns_identity_when_not_inverted():
    adata = _adata_with_reindexed(invert=False)
    sites = np.array([0, 1, 2, 3, 4, 5])
    sites_out, labels_out = _ordered_columns(adata, sites, "refA", "reindexed")
    # Ascending reindexed values already match physical order -> no reordering.
    np.testing.assert_array_equal(sites_out, sites)
    np.testing.assert_array_equal(labels_out, ["10", "11", "12", "13", "14", "15"])


def test_ordered_columns_reverses_when_inverted():
    adata = _adata_with_reindexed(invert=True)
    sites = np.array([0, 1, 2, 3, 4, 5])
    sites_out, labels_out = _ordered_columns(adata, sites, "refA", "reindexed")
    # reindexed values are -10..-15 (descending); ascending sort reverses sites.
    np.testing.assert_array_equal(sites_out, sites[::-1])
    np.testing.assert_array_equal(labels_out, ["-15", "-14", "-13", "-12", "-11", "-10"])


def test_ordered_columns_empty_sites():
    adata = _adata_with_reindexed()
    sites = np.array([], dtype=int)
    sites_out, labels_out = _ordered_columns(adata, sites, "refA", "reindexed")
    assert sites_out.size == 0
    assert labels_out.size == 0


def test_ordered_columns_subset_of_sites_reorders_correctly():
    adata = _adata_with_reindexed(invert=True, n_vars=6)
    # Only a subset of columns selected, in physical (ascending var_names) order.
    sites = np.array([1, 3, 5])
    sites_out, labels_out = _ordered_columns(adata, sites, "refA", "reindexed")
    np.testing.assert_array_equal(sites_out, sites[::-1])
    np.testing.assert_array_equal(labels_out, ["-15", "-13", "-11"])
