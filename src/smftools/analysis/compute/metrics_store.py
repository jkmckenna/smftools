"""
metrics_store.py — Per-run Zarr store for computed per-read analysis metrics.

Keeps raw signal caches pristine by storing derived metrics separately.
Row ordering matches the companion <run>.zarr barcode-sorted obs.

Structure::

    <run>_metrics.zarr/
        obs/
            ls_nrl_bp_<mask>        float (n_obs,)  — NaN where not computed
            ls_snr_<mask>           float (n_obs,)
            ls_peak_power_<mask>    float (n_obs,)
            ls_fwhm_bp_<mask>       float (n_obs,)
        obsm/
            ls_power_<mask>         float (n_obs, n_freqs)
            ls_freqs_<mask>         float (n_freqs,)  — shared frequency grid

Usage::

    from smftools.analysis.compute import metrics_store

    # Writing (analyses driver):
    store = metrics_store.open_or_create(path, n_obs)
    metrics_store.write_obs_array(store, "ls_nrl_bp_full_locus", arr)
    metrics_store.write_obsm_array(store, "ls_power_full_locus", power_mat)
    metrics_store.write_obsm_array(store, "ls_freqs_full_locus", freqs)
    metrics_store.consolidate(path)

    # Reading:
    store = metrics_store.open_metrics_store(path)
    nrl = store["obs"]["ls_nrl_bp_full_locus"][:]
    power = store["obsm"]["ls_power_full_locus"][:]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def open_or_create(path: Path, n_obs: int) -> object:
    """Open an existing metrics store in append mode, or create a new one.

    Parameters
    ----------
    path : Path to the ``<run>_metrics.zarr`` directory.
    n_obs : Total number of obs rows (must match the companion signal Zarr).

    Returns
    -------
    zarr.Group
    """
    import zarr

    path = Path(path)
    if path.exists():
        z = zarr.open(str(path), mode="r+")
    else:
        z = zarr.open(str(path), mode="w")
        z.require_group("obs")
        z.require_group("obsm")
        z.attrs["n_obs"] = int(n_obs)
    return z


def write_obs_array(store, col: str, values: np.ndarray) -> None:
    """Write or overwrite a full-length (n_obs,) obs metric array."""
    store["obs"].create_array(col, data=np.asarray(values, dtype=float), overwrite=True)


def write_obsm_array(store, key: str, values: np.ndarray) -> None:
    """Write or overwrite an obsm array (2D for per-read power, 1D for shared freqs)."""
    store["obsm"].create_array(key, data=np.asarray(values, dtype=float), overwrite=True)


def consolidate(path: Path) -> None:
    """Consolidate Zarr metadata so arrays are visible in read mode."""
    import zarr

    zarr.consolidate_metadata(str(Path(path)))


def open_metrics_store(path: Path):
    """Open an existing metrics store read-only."""
    import zarr

    return zarr.open(str(Path(path)), mode="r")
