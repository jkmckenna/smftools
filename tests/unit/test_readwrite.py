import numpy as np
import anndata as ad

from smftools.readwrite import safe_read_h5ad, safe_write_h5ad


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
