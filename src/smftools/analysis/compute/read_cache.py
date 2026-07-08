"""
Reader/writer utilities for the per-read modification matrix cache.

Two cache backends are supported:

**Zarr cache** (recommended)
    One Zarr store per run containing all layers.  Obs is sorted by
    ``(Barcode, Reference_strand)`` so any barcode × strand combination can be
    read as a contiguous slice in ~138 ms regardless of which layer is needed.
    A companion ``barcode_index.json`` maps ``barcode → ref_strand → [start, end]``
    for O(1) slice lookup.

    Layout::

        <run>.zarr/
            obs/       # all obs columns
            var/       # all var columns (reindexed coords, C_site flags, …)
            layers/
                C_site_binary/
                C_nucleosome_depleted_region_merged/
                …

    Example::

        from smftools.analysis.compute.read_cache import (
            open_zarr_cache, load_barcode_index, load_zarr_layer,
        )

        z     = open_zarr_cache("cache/260406_run.zarr")
        index = load_barcode_index("cache/260406_run_index.json")
        mat   = load_zarr_layer(z, index, "NB01", "6B6_top", "C_site_binary")

**Parquet cache** (legacy)
    One parquet file per barcode × ref_strand × layer.

    Layout::

        var_info/<ref_strand>_var_info.parquet
        <barcode>_<ref_strand>/obs_metadata.parquet
        <barcode>_<ref_strand>/<layer_name>.parquet

    Parquet columns are ``str(int(TSS_coord))``, e.g. ``"-1690"``.
    Cast back to int with ``np.array(df.columns, dtype=int)``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Zarr cache — write
# ---------------------------------------------------------------------------


def write_zarr_cache(
    adata,
    zarr_path: Path,
    sort_cols: list[str] | None = None,
    layers: list[str] | None = None,
    obs_chunk: int = 512,
) -> None:
    """
    Write an AnnData to a Zarr cache, sorting obs and optionally subsetting layers.

    Parameters
    ----------
    adata : AnnData
        In-memory AnnData to write (backed mode not supported — materialise first).
    zarr_path : Path
        Destination Zarr store path.  Created if it does not exist.
    sort_cols : list of str, optional
        obs column names to sort by before writing, e.g.
        ``["Barcode", "Reference_strand"]``.  Sorting ensures contiguous
        slices for fast barcode-level access.  Defaults to no sorting.
    layers : list of str, optional
        Layer names to include.  ``None`` writes all layers.
        Non-existent layer names are silently skipped.
    obs_chunk : int
        Number of obs (reads) per Zarr chunk along the obs axis.
    """
    import anndata as ad

    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort obs if requested
    if sort_cols:
        keys = [adata.obs[c].values for c in reversed(sort_cols)]
        order = np.lexsort(keys)
        adata = adata[order]

    # Subset layers
    if layers is not None:
        available = [layer for layer in layers if layer in adata.layers]
        adata = ad.AnnData(
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            layers={l: np.asarray(adata.layers[layer]) for layer in available},
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress zarr v2/v3 deprecation notice
        adata.write_zarr(zarr_path, chunks=[obs_chunk, adata.n_vars])


# ---------------------------------------------------------------------------
# Zarr cache — index
# ---------------------------------------------------------------------------


def build_barcode_index(
    zarr_path: Path,
    barcodes: np.ndarray,
    ref_strands: np.ndarray,
    index_path: Path | None = None,
) -> dict:
    """
    Build a barcode × ref_strand → [start, end] slice index.

    Call this immediately after :func:`write_zarr_cache`, passing the
    sorted obs arrays from the in-memory AnnData (before it goes out of scope).
    This avoids decoding the obs from the Zarr store, which requires navigating
    AnnData's categorical encoding.

    Parameters
    ----------
    zarr_path : Path
        Path to the Zarr store (used only to determine the default index path).
    barcodes : np.ndarray
        Barcode values in sorted order, matching the Zarr obs rows.
    ref_strands : np.ndarray
        Reference strand values in sorted order, matching the Zarr obs rows.
    index_path : Path, optional
        Where to save the JSON index.  Defaults to
        ``zarr_path.parent / (zarr_path.stem + "_index.json")``.

    Returns
    -------
    dict
        ``{barcode: {ref_strand: [start, end]}}``
    """
    barcodes = np.asarray(barcodes, dtype=str)
    ref_strands = np.asarray(ref_strands, dtype=str)

    combined = np.array([f"{b}\x00{r}" for b, r in zip(barcodes, ref_strands)])
    boundaries = np.concatenate(
        [[0], np.where(combined[:-1] != combined[1:])[0] + 1, [len(combined)]]
    )

    index: dict[str, dict[str, list[int]]] = {}
    for i in range(len(boundaries) - 1):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        barcode, ref = combined[start].split("\x00")
        index.setdefault(barcode, {})[ref] = [start, end]

    if index_path is None:
        index_path = zarr_index_path(Path(zarr_path))
    with open(index_path, "w") as f:
        json.dump(index, f)

    return index


def load_barcode_index(index_path: Path) -> dict:
    """Load a barcode index previously saved by :func:`build_barcode_index`."""
    with open(index_path) as f:
        return json.load(f)


def zarr_index_path(zarr_path: Path) -> Path:
    """Return the conventional index path for a given Zarr store path."""
    zarr_path = Path(zarr_path)
    return zarr_path.parent / f"{zarr_path.stem}_index.json"


def zarr_cache_exists(zarr_path: Path) -> bool:
    """Return True if both the Zarr store and its index file exist."""
    zp = Path(zarr_path)
    return zp.exists() and zarr_index_path(zp).exists()


# ---------------------------------------------------------------------------
# Zarr cache — read
# ---------------------------------------------------------------------------


def open_zarr_cache(zarr_path: Path):
    """Open a Zarr store in read-only mode and return the root group."""
    import zarr

    return zarr.open(Path(zarr_path), mode="r")


def load_zarr_layer(
    z,
    index: dict,
    barcode: str,
    ref_strand: str,
    layer: str,
    pos_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Load one layer for a barcode × ref_strand pair from a Zarr cache.

    Parameters
    ----------
    z : zarr.Group
        Open Zarr root group (from :func:`open_zarr_cache`).
    index : dict
        Barcode index from :func:`load_barcode_index`.
    barcode : str
        e.g. ``"NB01"``.
    ref_strand : str
        e.g. ``"6B6_top"``.
    layer : str
        Layer name, e.g. ``"C_site_binary"``.
    pos_mask : np.ndarray of bool, optional
        Boolean position mask to apply after slicing (column filter).
        Must have length equal to ``n_var``.

    Returns
    -------
    np.ndarray
        Float array of shape (n_reads, n_positions) or (n_reads, n_masked_positions).
    """
    start, end = _get_slice(index, barcode, ref_strand)
    arr = z["layers"][layer][start:end, :].astype(float)
    if pos_mask is not None:
        arr = arr[:, pos_mask]
    return arr


def load_zarr_obs(
    zarr_path: Path,
    index: dict,
    barcode: str,
    ref_strand: str,
) -> pd.DataFrame:
    """
    Load obs metadata for a barcode × ref_strand pair.

    Uses anndata to decode the Zarr obs (handles categorical encoding).
    This is not on the critical performance path — use :func:`load_zarr_layer`
    for fast repeated layer access.

    Parameters
    ----------
    zarr_path : Path
        Path to the Zarr store.
    index : dict
        Barcode index from :func:`load_barcode_index`.

    Returns
    -------
    pd.DataFrame
        obs rows for the selected barcode × ref_strand.
    """
    import warnings

    import anndata as ad

    start, end = _get_slice(index, barcode, ref_strand)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = ad.read_zarr(zarr_path)
    return adata.obs.iloc[start:end].copy()


def load_zarr_var(zarr_path: Path) -> pd.DataFrame:
    """Load the full var DataFrame from a Zarr cache using anndata."""
    import warnings

    import anndata as ad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = ad.read_zarr(zarr_path)
    return adata.var.copy()


def load_zarr_var_fast(z, zarr_path: Path) -> pd.DataFrame:
    """Load var columns directly from the Zarr store — avoids full anndata load.

    Reads each column array via ``zarr.open_array``.  ~10x faster than
    :func:`load_zarr_var` for large stores because it bypasses anndata's
    categorical / sparse decoding overhead.
    """
    import zarr as _zarr

    var_path = Path(zarr_path) / "var"
    cols = {}
    for key in z["var"].keys():
        try:
            arr = _zarr.open_array(str(var_path / key), mode="r")
            cols[key] = arr[...]
        except Exception:
            pass
    return pd.DataFrame(cols)


def resolve_barcode_str(barcode_int: int | str, index: dict) -> str:
    """Resolve an integer barcode number to its string key in a Zarr barcode index.

    Tries ``NB{n:02d}`` first (the standard format for most runs), then scans for
    keys ending in ``_barcode{n:02d}`` to handle SQK-style barcodes used by some
    earlier runs (e.g. ``SQK-NBD114-24_barcode07``).

    Raises ``KeyError`` if no match is found.
    """
    n = int(barcode_int)
    nb_key = f"NB{n:02d}"
    if nb_key in index:
        return nb_key
    suffix = f"_barcode{n:02d}"
    for k in index:
        if k.endswith(suffix):
            return k
    raise KeyError(f"Barcode {n} not found in index (tried {nb_key!r})")


def _get_slice(index: dict, barcode: str, ref_strand: str) -> tuple[int, int]:
    """Look up [start, end) from the barcode index; raises KeyError if not found."""
    try:
        start, end = index[barcode][ref_strand]
    except KeyError:
        raise KeyError(f"{barcode!r} / {ref_strand!r} not found in index.")
    return int(start), int(end)


# ---------------------------------------------------------------------------
# Parquet cache (legacy)
# ---------------------------------------------------------------------------


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
