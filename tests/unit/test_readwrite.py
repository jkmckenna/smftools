import anndata as ad
import numpy as np

from smftools.readwrite import (
    normalize_uns_string_lists,
    safe_read_h5ad,
    safe_write_h5ad,
    uns_string_list,
)


def test_safe_readwrite_restores_varm_backups(tmp_path):
    adata = ad.AnnData(X=np.zeros((2, 2)))
    varm_data = np.array([[{"a": 1}], [{"b": 2}]], dtype=object)
    adata.varm["complex"] = varm_data

    path = tmp_path / "varm_test.h5ad"
    write_report = safe_write_h5ad(adata, path, backup=True, verbose=False)

    assert "complex" in write_report["varm_skipped"]

    read_adata, read_report = safe_read_h5ad(path, restore_backups=True, verbose=False)

    assert "complex" in read_adata.varm
    assert np.array_equal(read_adata.varm["complex"], varm_data)
    assert any(entry[0] == "complex" for entry in read_report["restored_varm"])


def test_safe_write_h5ad_keys_csv_includes_varm(tmp_path):
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.varm["components"] = np.ones((3, 2))

    path = tmp_path / "keys_varm.h5ad"
    safe_write_h5ad(adata, path, backup=False, verbose=False)

    keys_csv = tmp_path / "csvs" / "keys_varm.keys.csv"
    keys_df = np.loadtxt(keys_csv, delimiter=",", dtype=str, skiprows=1)
    if keys_df.ndim == 1:
        keys_df = np.array([keys_df])

    assert any(row[0] == "varm" and row[1] == "components" for row in keys_df)


def test_uns_string_list_accepts_every_degraded_form():
    """Already-published artifacts carry several encodings of the same list."""
    assert uns_string_list(None) == []
    assert uns_string_list(["a", "b"]) == ["a", "b"]
    assert uns_string_list(np.array(["a", "b"], dtype=object)) == ["a", "b"]
    # Numpy's repr omits commas; a plain str repr keeps them. Both must parse.
    assert uns_string_list("['a' 'b']") == ["a", "b"]
    assert uns_string_list("['a', 'b']") == ["a", "b"]
    assert uns_string_list("[]") == []
    # A bare string that is not a list repr stays a single item.
    assert uns_string_list("mod_a") == ["mod_a"]


def test_normalize_uns_string_lists_survives_a_write_read_cycle(tmp_path):
    adata = ad.AnnData(X=np.zeros((1, 1)))
    adata.uns["signal_columns"] = ["mod_a", "mod_b"]
    path = tmp_path / "spine.h5ad"
    safe_write_h5ad(adata, path, backup=False, verbose=False)

    # Reading returns an array; rewriting it unnormalized is what degraded it.
    reread, _ = safe_read_h5ad(path, verbose=False)
    normalize_uns_string_lists(reread)
    safe_write_h5ad(reread, path, backup=False, verbose=False)

    final, _ = safe_read_h5ad(path, verbose=False)
    assert not isinstance(final.uns["signal_columns"], str)
    assert list(final.uns["signal_columns"]) == ["mod_a", "mod_b"]
