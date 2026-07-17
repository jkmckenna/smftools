"""Append one array to an already-written zarr AnnData store.

Complements ``readwrite.safe_write_zarr`` (whole-object writes: sanitize, then
one ``adata.write_zarr()`` call) for the opposite case -- a caller that has
already written a store's skeleton (``obs``/``var``/``X``) and wants to add
``layers`` entries one at a time as they're computed, instead of holding every
derived layer in memory until a single combined write. See
``preprocessing/partitioned_executor.py::execute_preprocess_task`` for the
motivating caller.

Uses ``anndata.io.write_elem`` (not ``anndata.experimental.write_elem``, which
is deprecated as of anndata 0.12) to write a single element into an existing
zarr group without rewriting the rest of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def append_zarr_element(
    path: str | Path,
    group_path: str,
    array: np.ndarray,
    *,
    chunks: tuple[int, ...] | None = None,
    consolidate: bool = True,
) -> None:
    """Write one array into an already-written zarr AnnData store.

    Shared implementation behind ``append_zarr_layer`` (``group_path``
    ``"layers/<name>"``) and ``append_zarr_obsm`` (``"obsm/<name>"``) -- both
    are thin wrappers over this, so the write/consolidate logic lives in one
    place regardless of which AnnData slot the caller is streaming into.

    Args:
        path: Path to a zarr store previously written by
            ``readwrite.safe_write_zarr`` (or plain ``AnnData.write_zarr``).
        group_path: Zarr group path to write to, e.g. ``"layers/foo"`` or
            ``"obsm/bar"``.
        array: The array to write. Written as-is (caller is responsible for
            dtype/shape matching the store's ``X``/``obs``).
        chunks: Optional explicit chunk shape, passed through to zarr.
        consolidate: Re-consolidate the store's metadata after this write, so
            it's immediately readable via ``anndata.read_zarr``/
            ``safe_read_zarr``. Set ``False`` when writing several arrays in a
            row and consolidate once after the last one -- consolidating after
            every single array is wasted work at that point.
    """
    import zarr
    from anndata.io import write_elem

    path = Path(path)
    dataset_kwargs: Mapping[str, object] = {} if chunks is None else {"chunks": chunks}
    # use_consolidated=False: the store was written by write_zarr(), which
    # consolidates metadata by default -- reopening for edits with the default
    # (consolidated) read raises ValueError.
    group = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    write_elem(group, group_path, np.asarray(array), dataset_kwargs=dataset_kwargs)
    if consolidate:
        zarr.consolidate_metadata(str(path))


def append_zarr_layer(
    path: str | Path,
    name: str,
    array: np.ndarray,
    *,
    chunks: tuple[int, ...] | None = None,
    consolidate: bool = True,
) -> None:
    """Write one layer into an already-written zarr AnnData store.

    Args:
        path: Path to a zarr store previously written by
            ``readwrite.safe_write_zarr`` (or plain ``AnnData.write_zarr``).
        name: Layer name -- written to ``layers/<name>``.
        array: The layer's array. Written as-is (caller is responsible for
            dtype/shape matching the store's ``X``).
        chunks: Optional explicit chunk shape, passed through to zarr.
        consolidate: Re-consolidate the store's metadata after this write, so
            it's immediately readable via ``anndata.read_zarr``/
            ``safe_read_zarr``. Set ``False`` when writing several layers in a
            row and consolidate once after the last one -- consolidating after
            every single layer is wasted work at that point.
    """
    append_zarr_element(path, f"layers/{name}", array, chunks=chunks, consolidate=consolidate)


def append_zarr_obsm(
    path: str | Path,
    name: str,
    array: np.ndarray,
    *,
    chunks: tuple[int, ...] | None = None,
    consolidate: bool = True,
) -> None:
    """Write one ``obsm`` entry into an already-written zarr AnnData store.

    Same streaming-write-then-free contract as ``append_zarr_layer``, for
    per-read matrices (e.g. spatial autocorrelation, Lomb-Scargle power) that
    belong in ``obsm`` rather than ``layers``. See ``tools/partitioned_
    spatial.py::execute_spatial_task`` for the motivating caller.

    Args:
        path: Path to a zarr store previously written by
            ``readwrite.safe_write_zarr`` (or plain ``AnnData.write_zarr``).
        name: Obsm key -- written to ``obsm/<name>``.
        array: The obsm array. Written as-is (caller is responsible for
            its leading dimension matching the store's ``obs``).
        chunks: Optional explicit chunk shape, passed through to zarr.
        consolidate: Re-consolidate the store's metadata after this write --
            see ``append_zarr_layer``'s docstring for the same tradeoff.
    """
    append_zarr_element(path, f"obsm/{name}", array, chunks=chunks, consolidate=consolidate)


def consolidate_zarr_store(path: str | Path) -> None:
    """Consolidate a store's metadata after one or more ``consolidate=False`` writes."""
    import zarr

    zarr.consolidate_metadata(str(Path(path)))
