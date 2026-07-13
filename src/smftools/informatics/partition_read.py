"""Assemble a concrete AnnData selection through the thin molecule spine.

This is the Phase 2 (1.1.x) reader that lets downstream stages stop materializing
the whole monolithic AnnData. A selection (by reference / sample / read_id / obs
mask) is resolved against the spine's pointer columns, the referenced partitions
are loaded (only those), subset to the selected reads, and concatenated.

Two backends:
- **eager** (default, no extra deps): ``safe_read_zarr`` each needed partition.
- **lazy** (``lazy=True``, needs ``xarray`` for ``anndata.experimental.read_lazy``):
  dask-backed row selection then ``to_memory``; falls back to eager if unavailable.

When the selected dense cache partitions do not exist, the reader falls back to
the spine's read-relative parquet shards and performs CIGAR placement on demand.
Either way, peak memory scales with the selection rather than the full dataset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)

PARTITION_COL = "partition"
RAGGED_STORE_KEY = "ragged_store"
REFERENCE_LENGTHS_KEY = "reference_lengths"


def read_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Read a partition ``catalog.parquet`` into a DataFrame."""
    return pd.read_parquet(catalog_path)


def load_spine(spine_path: str | Path, *, verbose: bool = True) -> "ad.AnnData":
    """Load the thin molecule-index spine (e.g. written by ``write_raw_store``,
    ``execute_partitioned_preprocessing``, or another stage's spine writer)."""
    from ..readwrite import safe_read_h5ad

    spine, _ = safe_read_h5ad(spine_path, verbose=verbose)
    return spine


def _resolve_spine(spine, base_dir):
    """Return ``(spine_adata, base_dir)`` resolving partition path resolution root."""
    if isinstance(spine, (str, Path)):
        spine_path = Path(spine)
        obj = load_spine(spine_path)
        base = (
            Path(base_dir)
            if base_dir is not None
            else Path(obj.uns.get("source_base_dir", spine_path.parent))
        )
        return obj, base
    # in-memory spine
    if base_dir is not None:
        return spine, Path(base_dir)
    source_base_dir = spine.uns.get("source_base_dir")
    if source_base_dir:
        return spine, Path(source_base_dir)
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
    start,
    end,
) -> pd.DataFrame:
    """Filter the spine obs to the selected molecules, preserving spine order."""
    obs = spine.obs
    mask = np.ones(obs.shape[0], dtype=bool)
    if obs_mask is not None:
        mask &= np.asarray(obs_mask, dtype=bool)
    refs = _as_set(references)
    if refs is not None:
        mask &= obs["Reference_strand"].astype(str).isin(refs).to_numpy()
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    if start is not None:
        if int(start) < 0 or int(end) <= int(start):
            raise ValueError("materialization interval must satisfy 0 <= start < end")
        if "reference_start" not in obs or "reference_end" not in obs:
            raise ValueError("spine lacks reference_start/reference_end interval metadata")
        mask &= (obs["reference_start"].to_numpy(dtype="int64") < int(end)) & (
            obs["reference_end"].to_numpy(dtype="int64") > int(start)
        )
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
    start: int | None,
    end: int | None,
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
            if start is not None:
                positions = np.asarray(sub.var_names, dtype="int64")
                sub = sub[:, (positions >= start) & (positions < end)].copy()
            return sub

    from ..readwrite import safe_read_zarr

    obj, _ = safe_read_zarr(path)
    sub = obj[list(read_ids)].copy()
    if start is not None:
        positions = np.asarray(sub.var_names, dtype="int64")
        sub = sub[:, (positions >= start) & (positions < end)].copy()
    if keep is not None:
        for k in list(sub.layers):
            if k not in keep:
                del sub.layers[k]
    return sub


def _dense_cache_available(selection: pd.DataFrame, base: Path) -> bool:
    """Return whether every dense partition referenced by a selection exists."""
    if PARTITION_COL not in selection.columns:
        return False
    if selection[PARTITION_COL].isna().any():
        return False
    if (selection[PARTITION_COL].astype(str) == "").any():
        return False
    return all((base / str(path)).exists() for path in selection[PARTITION_COL].unique())


def _resolve_ragged_paths(spine: "ad.AnnData", base: Path) -> list[Path]:
    """Resolve ragged parquet shard paths stored on the spine."""
    value = spine.uns.get(RAGGED_STORE_KEY)
    if value is None:
        return []
    if isinstance(value, str):
        # Older safe-write round trips could stringify a numpy string array.
        if value.startswith("["):
            # Parse quoted items directly: numpy repr omits commas, and Python
            # would otherwise concatenate adjacent string literals.
            quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", value)
            raw_paths = quoted_items if quoted_items else [value]
        else:
            raw_paths = [value]
    else:
        raw_paths = list(value)
    return [path if path.is_absolute() else base / path for path in map(Path, raw_paths)]


def _reference_lengths(spine: "ad.AnnData", references: Iterable[str]) -> dict[str, int]:
    """Resolve exact Reference_strand lengths, with legacy References fallback."""
    explicit = spine.uns.get(REFERENCE_LENGTHS_KEY, {})
    result = {str(key): int(value) for key, value in dict(explicit).items()}
    legacy = spine.uns.get("References", {})
    for reference_strand in map(str, references):
        if reference_strand in result:
            continue
        base_reference = reference_strand.removesuffix("_top").removesuffix("_bottom")
        sequence = legacy.get(f"{base_reference}_FASTA_sequence")
        if sequence is None:
            sequence = spine.uns.get(f"{base_reference}_FASTA_sequence")
        if sequence is not None:
            result[reference_strand] = len(sequence)
    return result


def _load_ragged_selection(
    spine: "ad.AnnData",
    selection: pd.DataFrame,
    base: Path,
    layers: Iterable[str] | None,
    start: int | None,
    end: int | None,
) -> "ad.AnnData":
    """Materialize a spine selection from its read-relative parquet shards."""
    from .ragged_store import materialize_ragged, read_ragged_parquet

    if "ragged_shard" in selection:
        paths = [
            path if path.is_absolute() else base / path
            for path in map(Path, selection["ragged_shard"].astype(str).unique())
        ]
    else:
        paths = _resolve_ragged_paths(spine, base)
    if not paths:
        raise ValueError(
            "materialize: dense cache is unavailable and spine.uns lacks 'ragged_store'"
        )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ragged parquet shard(s) not found: {missing}")
    frame = read_ragged_parquet(paths, read_ids=selection.index)
    lengths = _reference_lengths(spine, selection["Reference_strand"].unique())
    return materialize_ragged(
        frame,
        obs=selection,
        reference_lengths=lengths,
        layers=layers,
        uns={
            key: value
            for key, value in spine.uns.items()
            if key == "References" or key.endswith("_map")
        },
        start=start,
        end=end,
    )


def _reference_plan(spine: "ad.AnnData", reference: str) -> dict[str, object]:
    """Return one persisted reference plan, tolerating legacy spines."""
    plans = spine.uns.get("reference_plans", {})
    if not hasattr(plans, "get"):
        return {}
    plan = plans.get(str(reference), {})
    return dict(plan) if hasattr(plan, "items") else {}


def _load_tiled_cache_selection(
    spine: "ad.AnnData",
    selection: pd.DataFrame,
    base: Path,
    layers: Iterable[str] | None,
    lazy: bool,
    start: int | None,
    end: int | None,
) -> "ad.AnnData | None":
    """Load one cached genome tile when it fully covers the requested interval."""
    if start is None or end is None:
        return None
    references = selection["Reference_strand"].astype(str).unique()
    if len(references) != 1:
        return None
    catalog_value = spine.uns.get("catalog")
    if not catalog_value:
        return None
    catalog_path = Path(str(catalog_value))
    if not catalog_path.is_absolute():
        catalog_path = base / catalog_path
    if not catalog_path.exists():
        return None
    catalog = pd.read_parquet(catalog_path)
    required = {
        "cache_kind",
        "reference",
        "core_start",
        "core_end",
        "load_start",
        "load_end",
        "group_path",
    }
    if not required.issubset(catalog.columns):
        return None
    candidates = catalog.loc[
        (catalog["cache_kind"] == "tiled")
        & (catalog["reference"].astype(str) == references[0])
        & (catalog["core_start"] <= int(start))
        & (catalog["core_end"] >= int(end))
        & (catalog["load_start"] <= int(start))
        & (catalog["load_end"] >= int(end))
    ].sort_values(["load_start", "load_end"])
    if candidates.empty:
        return None
    path = Path(str(candidates.iloc[0]["group_path"]))
    if not path.is_absolute():
        path = base / path
    if not path.exists():
        return None
    return _load_partition(path, list(selection.index), layers, lazy, start, end)


def _derived_layer_names(spine: "ad.AnnData") -> set[str]:
    """Return layer names published by partitioned derived-stage stores."""
    names: set[str] = set()
    for catalog_key in ("preprocess_catalog", "hmm_catalog"):
        catalog_value = spine.uns.get(catalog_key)
        if not catalog_value:
            continue
        catalog_path = Path(str(catalog_value))
        if not catalog_path.exists():
            continue
        catalog = pd.read_parquet(catalog_path, columns=["layers"])
        for value in catalog["layers"]:
            if isinstance(value, str):
                names.add(value)
            else:
                names.update(map(str, value))
    return names


def _overlay_preprocess_layers(
    spine: "ad.AnnData",
    result: "ad.AnnData",
    requested_layers: set[str],
) -> None:
    """Stitch task-partitioned derived layers onto a materialized slice."""
    from ..readwrite import safe_read_zarr

    if not requested_layers:
        return
    references = result.obs["Reference_strand"].astype(str).unique()
    positions = np.asarray(result.var_names, dtype=np.int64)
    if len(references) != 1 or positions.size == 0:
        raise ValueError("derived layers require one non-empty reference slice")
    absent_fill = {
        **dict(spine.uns.get("preprocess_layer_absent_fill", {})),
        **dict(spine.uns.get("hmm_layer_absent_fill", {})),
    }
    assembled = {
        layer: np.full(result.shape, float(absent_fill.get(layer, np.nan)), dtype=np.float32)
        for layer in requested_layers
    }
    source_dtypes: dict[str, np.dtype] = {}
    result_rows = {str(read_id): row for row, read_id in enumerate(result.obs_names)}
    result_positions = {int(position): column for column, position in enumerate(positions)}
    for catalog_key in ("preprocess_catalog", "hmm_catalog"):
        catalog_value = spine.uns.get(catalog_key)
        if not catalog_value:
            continue
        catalog_path = Path(str(catalog_value))
        catalog = pd.read_parquet(catalog_path)
        catalog = catalog.loc[
            (catalog["reference"].astype(str) == references[0])
            & (catalog["core_start"] < positions.max() + 1)
            & (catalog["core_end"] > positions.min())
        ]
        for record in catalog.to_dict("records"):
            path = catalog_path.parent / str(record["group_path"])
            part, _ = safe_read_zarr(path, verbose=False)
            shared_reads = [
                read_id for read_id in map(str, part.obs_names) if read_id in result_rows
            ]
            shared_positions = [
                position for position in map(int, part.var_names) if position in result_positions
            ]
            if not shared_reads or not shared_positions:
                continue
            source_rows = [part.obs_names.get_loc(read_id) for read_id in shared_reads]
            source_columns = [
                part.var_names.get_loc(str(position)) for position in shared_positions
            ]
            target_rows = [result_rows[read_id] for read_id in shared_reads]
            target_columns = [result_positions[position] for position in shared_positions]
            for layer in requested_layers.intersection(part.layers.keys()):
                values = np.asarray(part.layers[layer])[np.ix_(source_rows, source_columns)]
                assembled[layer][np.ix_(target_rows, target_columns)] = values
                source_dtypes.setdefault(layer, values.dtype)

    for layer in requested_layers:
        values = assembled[layer]
        source_dtype = source_dtypes.get(layer)
        if source_dtype is None:
            raise KeyError(f"derived layer {layer!r} was not found in selected partitions")
        if np.issubdtype(source_dtype, np.integer) and not np.isnan(values).any():
            values = values.astype(source_dtype)
        result.layers[layer] = values


def _overlay_preprocess_var(spine: "ad.AnnData", result: "ad.AnnData") -> None:
    """Attach reduced coverage and reference-context columns to a slice."""
    value = spine.uns.get("preprocess_var")
    if not value or result.n_vars == 0:
        return
    path = Path(str(value))
    if not path.exists():
        return
    references = result.obs["Reference_strand"].astype(str).unique()
    if len(references) != 1:
        return
    reference = references[0]
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["reference"].astype(str) == reference].set_index("position")
    positions = pd.Index(np.asarray(result.var_names, dtype=np.int64), name="position")
    selected = frame.reindex(positions)
    for column in selected.columns.difference(["reference"]):
        result.var[column] = selected[column].to_numpy()
    if "valid_count" in selected:
        result.var[f"{reference}_valid_count"] = selected["valid_count"].to_numpy()
    if "valid_fraction" in selected:
        result.var[f"{reference}_valid_fraction"] = selected["valid_fraction"].to_numpy()
    if "position_valid" in selected:
        valid = selected["position_valid"].fillna(False).to_numpy(dtype=bool)
        result.var[f"position_in_{reference}"] = valid
        result.var["N_Reference_strand_with_position"] = valid.astype(np.int64)


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
    start: int | None = None,
    end: int | None = None,
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
        start: Optional zero-based inclusive genomic start.
        end: Optional zero-based exclusive genomic end. Must accompany ``start``.

    Returns:
        anndata.AnnData: The materialized selection, ordered as in the spine, with
        decoder/reference maps copied from the spine ``uns``.

    Raises:
        ValueError: If the selection matches no molecules or neither a complete
            dense cache nor ragged-store pointers are available.
    """
    import anndata as ad

    spine_obj, base = _resolve_spine(spine, base_dir)
    derived_available = _derived_layer_names(spine_obj)
    requested_layer_set = None if layers is None else set(layers)
    requested_derived = (
        derived_available
        if requested_layer_set is None
        else requested_layer_set.intersection(derived_available)
    )
    source_layers = (
        None if requested_layer_set is None else requested_layer_set.difference(derived_available)
    )
    sel = _select_spine_obs(spine_obj, references, samples, read_ids, obs_mask, start, end)
    if sel.shape[0] == 0:
        raise ValueError("materialize: selection matched no molecules")
    if start is None:
        genome_references = [
            reference
            for reference in sel["Reference_strand"].astype(str).unique()
            if _reference_plan(spine_obj, reference).get("analysis_mode") == "genome"
        ]
        if genome_references:
            raise ValueError(
                "materialize: genome-mode references require start/end intervals: "
                f"{sorted(genome_references)}"
            )
    if not _dense_cache_available(sel, base):
        tiled = _load_tiled_cache_selection(spine_obj, sel, base, source_layers, lazy, start, end)
        if tiled is not None:
            _overlay_preprocess_layers(spine_obj, tiled, requested_derived)
            _overlay_preprocess_var(spine_obj, tiled)
            logger.debug("materialized %d molecules from tiled dense cache", tiled.n_obs)
            return tiled
        result = _load_ragged_selection(spine_obj, sel, base, source_layers, start, end)
        _overlay_preprocess_layers(spine_obj, result, requested_derived)
        _overlay_preprocess_var(spine_obj, result)
        logger.debug("materialized %d molecules from ragged parquet", result.n_obs)
        return result

    parts: list[ad.AnnData] = []
    for partition_rel, grp in sel.groupby(PARTITION_COL, sort=False, observed=True):
        parts.append(
            _load_partition(
                base / str(partition_rel),
                list(grp.index),
                source_layers,
                lazy,
                start,
                end,
            )
        )

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

    _overlay_preprocess_layers(spine_obj, result, requested_derived)
    _overlay_preprocess_var(spine_obj, result)

    logger.debug("materialized %d molecules from %d partition(s)", result.n_obs, len(parts))
    return result
