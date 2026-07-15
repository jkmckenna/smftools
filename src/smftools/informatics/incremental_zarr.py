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
    import zarr
    from anndata.io import write_elem

    path = Path(path)
    dataset_kwargs: Mapping[str, object] = {} if chunks is None else {"chunks": chunks}
    # use_consolidated=False: the store was written by write_zarr(), which
    # consolidates metadata by default -- reopening for edits with the default
    # (consolidated) read raises ValueError.
    group = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    write_elem(group, f"layers/{name}", np.asarray(array), dataset_kwargs=dataset_kwargs)
    if consolidate:
        zarr.consolidate_metadata(str(path))


def consolidate_zarr_store(path: str | Path) -> None:
    """Consolidate a store's metadata after one or more ``consolidate=False`` writes."""
    import zarr

    zarr.consolidate_metadata(str(Path(path)))
