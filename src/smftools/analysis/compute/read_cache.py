"""
Reader utilities for the per-read modification matrix cache.

The cache stores pre-materialised, obs-filtered modification matrices as parquet
files so that analysis scripts can load data without re-reading the full HMM h5ad.

Cache directory layout (relative to ``cache_root``)::

    var_info/
        <ref_strand>_var_info.parquet
    <barcode>_<ref_strand>/
        obs_metadata.parquet
        <layer_name>.parquet

Parquet columns are ``str(int(TSS_coord))`` (e.g. ``"-1690"``, ``"0"``).
Cast back to int with ``np.array(df.columns, dtype=int)``.

Example::

    from smftools.analysis.compute.read_cache import load_layer, load_var_info

    df, coords = load_layer(cache_root, "NB01", "6B6_top", "C_site_binary")
    var_info   = load_var_info(cache_root, "6B6_top")
    keep_cols  = [str(c) for c in var_info.index[var_info["C_site"].to_numpy()]]
    mat        = df[keep_cols].to_numpy()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cache_key(barcode: str, ref_strand: str) -> str:
    return f"{barcode}_{ref_strand}"


def cache_dir(cache_root: Path, barcode: str, ref_strand: str) -> Path:
    return Path(cache_root) / cache_key(barcode, ref_strand)


def is_cached(cache_root: Path, barcode: str, ref_strand: str, layer_name: str) -> bool:
    return (cache_dir(cache_root, barcode, ref_strand) / f"{layer_name}.parquet").exists()


def load_var_info(cache_root: Path, ref_strand: str) -> pd.DataFrame:
    """
    Load var_info for a reference strand.

    Returns DataFrame with int TSS-coord index and bool columns C_site, GpC_site.
    """
    path = Path(cache_root) / "var_info" / f"{ref_strand}_var_info.parquet"
    if not path.exists():
        raise FileNotFoundError(f"var_info not found for {ref_strand!r}: {path}")
    return pd.read_parquet(path)


def load_obs_metadata(cache_root: Path, barcode: str, ref_strand: str) -> pd.DataFrame:
    """
    Load per-read metadata for a barcode × ref_strand pair.

    Returns DataFrame indexed by obs_name with all adata.obs columns
    plus precomputed max_cigar_del (int).
    """
    path = cache_dir(cache_root, barcode, ref_strand) / "obs_metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"obs_metadata not found for {barcode}/{ref_strand}: {path}")
    return pd.read_parquet(path)


def load_layer(
    cache_root: Path,
    barcode: str,
    ref_strand: str,
    layer_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load a modification matrix layer from the parquet cache.

    Returns
    -------
    tuple of (pd.DataFrame, np.ndarray)
        DataFrame of shape (n_reads × n_positions) — index is obs_name, columns are
        ``str(int(TSS_coord))``, values are float (NaN = no coverage) — and an
        int array of TSS-centred coordinates matching the DataFrame columns.
    """
    path = cache_dir(cache_root, barcode, ref_strand) / f"{layer_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found for {barcode}/{ref_strand}/{layer_name}: {path}")
    df = pd.read_parquet(path)
    coords = np.array(df.columns, dtype=int)
    return df, coords
