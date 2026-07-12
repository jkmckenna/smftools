"""Read side of the partitioned store: assemble a concrete AnnData for a molecule
selection from only the relevant zarr partitions, via the thin spine index.

This is the Phase 2 (1.1.x) reader that lets downstream stages stop materializing
the whole monolithic AnnData. A selection (by reference / sample / read_id / obs
mask) is resolved against the spine's pointer columns, the referenced partitions
are loaded (only those), subset to the selected reads, and concatenated.

Two backends:
- **eager** (default, no extra deps): ``safe_read_zarr`` each needed partition.
- **lazy** (``lazy=True``, needs ``xarray`` for ``anndata.experimental.read_lazy``):
  dask-backed row selection then ``to_memory``; falls back to eager if unavailable.

Either way, peak memory scales with the *selected partitions*, not the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)

PARTITION_COL = "partition"


def read_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Read a partition ``catalog.parquet`` into a DataFrame."""
    return pd.read_parquet(catalog_path)


def load_spine(spine_path: str | Path) -> "ad.AnnData":
    """Load the thin molecule-index spine written by ``write_experiment_store``."""
    from ..readwrite import safe_read_h5ad

    spine, _ = safe_read_h5ad(spine_path)
    return spine


def _resolve_spine(spine, base_dir):
    """Return ``(spine_adata, base_dir)`` resolving partition path resolution root."""
    if isinstance(spine, (str, Path)):
        spine_path = Path(spine)
        obj = load_spine(spine_path)
        base = Path(base_dir) if base_dir is not None else spine_path.parent
        return obj, base
    # in-memory spine
    if base_dir is not None:
        return spine, Path(base_dir)
    store_root = spine.uns.get("store_root")
    if store_root:
        # partition group_paths are relative to the experiment output dir (store's parent)
        return spine, Path(store_root).parent
    raise ValueError(
        "materialize: base_dir is required when spine is in-memory and lacks uns['store_root']"
    )


def _as_set(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def _select_spine_obs(
    spine: "ad.AnnData",
    references,
    samples,
    read_ids,
    obs_mask,
) -> pd.DataFrame:
    """Filter the spine obs to the selected molecules, preserving spine order."""
    obs = spine.obs
    mask = np.ones(obs.shape[0], dtype=bool)
    if obs_mask is not None:
        mask &= np.asarray(obs_mask, dtype=bool)
    refs = _as_set(references)
    if refs is not None:
        mask &= obs["Reference_strand"].astype(str).isin(refs).to_numpy()
    samps = _as_set(samples)
    if samps is not None:
        mask &= obs["Sample"].astype(str).isin(samps).to_numpy()
    sel = obs.loc[mask]
    rids = _as_set(read_ids)
    if rids is not None:
        sel = sel.loc[sel.index.isin(rids)]
    return sel


def _load_partition(
    path: Path,
    read_ids: list[str],
    layers: Iterable[str] | None,
    lazy: bool,
) -> "ad.AnnData":
    """Load one partition, subset to ``read_ids`` (in order), keep ``layers``."""
    keep = None if layers is None else set(layers)

    if lazy:
        try:
            from anndata.experimental import read_lazy

            sub = read_lazy(path)[read_ids].to_memory()
        except Exception as e:  # xarray/read_lazy unavailable, or lazy read failed
            logger.debug("read_lazy failed for %s (%s); using eager read", path, e)
            sub = None
        if sub is not None:
            if keep is not None:
                for k in list(sub.layers):
                    if k not in keep:
                        del sub.layers[k]
            return sub

    from ..readwrite import safe_read_zarr

    obj, _ = safe_read_zarr(path)
    sub = obj[list(read_ids)].copy()
    if keep is not None:
        for k in list(sub.layers):
            if k not in keep:
                del sub.layers[k]
    return sub


def materialize(
    spine,
    *,
    base_dir: str | Path | None = None,
    references=None,
    samples=None,
    read_ids=None,
    obs_mask=None,
    layers: Iterable[str] | None = None,
    lazy: bool = False,
) -> "ad.AnnData":
    """Assemble a concrete AnnData for a molecule selection from the partitions.

    Only the partitions referenced by the selection are read, so peak memory scales
    with the selection, not the whole dataset.

    Args:
        spine: The spine AnnData or a path to ``spine.h5ad``.
        base_dir: Root for resolving partition ``group_path`` values. Defaults to the
            spine file's directory (when a path is given) or ``uns['store_root']``'s
            parent (in-memory spine).
        references: Optional ``Reference_strand`` value(s) to include.
        samples: Optional ``Sample`` value(s) to include.
        read_ids: Optional explicit read id(s) to include.
        obs_mask: Optional boolean array aligned to ``spine.obs`` rows.
        layers: Optional subset of layers to load (default: all).
        lazy: Use ``anndata.experimental.read_lazy`` (needs ``xarray``); falls back
            to eager ``safe_read_zarr`` if unavailable.

    Returns:
        anndata.AnnData: The materialized selection, ordered as in the spine, with
        decoder/reference maps copied from the spine ``uns``.

    Raises:
        ValueError: If the selection matches no molecules or the spine lacks the
            ``partition`` pointer column.
    """
    import anndata as ad

    spine_obj, base = _resolve_spine(spine, base_dir)
    sel = _select_spine_obs(spine_obj, references, samples, read_ids, obs_mask)
    if sel.shape[0] == 0:
        raise ValueError("materialize: selection matched no molecules")
    if PARTITION_COL not in sel.columns:
        raise ValueError("materialize: spine.obs lacks the 'partition' pointer column")

    parts: list[ad.AnnData] = []
    for partition_rel, grp in sel.groupby(PARTITION_COL, sort=False, observed=True):
        parts.append(_load_partition(base / str(partition_rel), list(grp.index), layers, lazy))

    if len(parts) == 1:
        result = parts[0]
    else:
        result = ad.concat(parts, join="outer", merge="first", uns_merge="first")

    # Restore full selection (spine) order.
    result = result[list(sel.index)].copy()

    # Carry decoder / reference maps so the result is self-describing.
    for key, value in spine_obj.uns.items():
        if key == "References" or key.endswith("_map"):
            result.uns.setdefault(key, value)

    logger.debug(
        "materialized %d molecules from %d partition(s)", result.n_obs, len(parts)
    )
    return result
