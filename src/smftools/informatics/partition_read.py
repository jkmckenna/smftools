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

from .artifact_paths import resolve_artifact_path, serialize_artifact_path
from .partition_query import (
    DEFAULT_QUERY_MEMORY_MB,
    query_derived_index,
    query_molecule_index,
    read_zarr_subset,
)

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


def relative_uns_path(path: str | Path, anchor: str | Path) -> str:
    """Return ``path`` as a POSIX-style string relative to ``anchor``, for storing on a spine.

    Used instead of an absolute (``.resolve()``'d) string so cross-artifact pointers
    (e.g. a preprocess spine's pointer back to its source raw spine, or a raw
    spine's ``obs["bam_path"]``) survive the containing directory tree being
    copied to a different machine or mount point. General-purpose despite the
    name -- applies equally to a ``uns`` value or a per-read ``obs`` column,
    anything that stores a path string on the spine. Pair with
    :func:`resolve_relative_path` on the read side.

    For any value that might be inherited unchanged into a later stage's spine via
    ``spine.copy()`` (e.g. ``source_base_dir``, ``preprocess_catalog``,
    ``obs["bam_path"]``), pass a stage-independent anchor -- see
    :func:`_run_root_from_spine_path` -- since a later stage's spine file lives in
    a different directory than the one that originally wrote the value, so an
    anchor tied to "wherever this spine currently lives" would resolve correctly
    there but not once copied elsewhere.
    """
    return serialize_artifact_path(path, anchor)


def resolve_relative_path(value: object, anchor: Path | None) -> Path | None:
    """Resolve a stored spine path value written by :func:`relative_uns_path`.

    Accepts both the new relative encoding (resolved against ``anchor``) and the
    historical absolute-string encoding (used as-is, for old spines written before
    this fix), so already-written spines keep working without being regenerated.
    Returns ``None`` if ``value`` is absent, or relative with no ``anchor`` available.
    """
    return resolve_artifact_path(value, anchor)


def _run_root_from_spine_path(spine_path: Path) -> Path:
    """Recover the run's ``output_directory`` from a canonical stage spine path.

    Every stage spine lives at ``output_directory/<STAGE_DIR>/spine.h5ad`` (see
    ``cli.helpers.AdataPaths``), so two parents up from the spine file is always
    the run's output_directory -- a stable anchor shared by every sibling stage
    directory (``raw_outputs``, ``preprocess_adata_outputs``, ...), regardless of
    which stage's spine is currently open. Used to resolve/write cross-stage uns
    pointers so they stay correct after being copied into a later stage's spine.
    """
    return spine_path.parent.parent


def _resolve_spine(spine, base_dir):
    """Return ``(spine_adata, base_dir)`` resolving partition path resolution root.

    A legacy (pre-partitioned-store) monolithic AnnData has no ``uns["is_spine"]``
    flag and no partitions to resolve at all, so `base` is a best-effort/unused
    value for it -- never an error, since ``materialize()`` dispatches legacy
    spines to :func:`_materialize_legacy` immediately after this call, before
    `base` is used for anything.
    """
    if isinstance(spine, (str, Path)):
        spine_path = Path(spine)
        obj = load_spine(spine_path)
        if not bool(obj.uns.get("is_spine", False)):
            return obj, Path(base_dir) if base_dir is not None else spine_path.parent
        if base_dir is not None:
            base = Path(base_dir)
        else:
            resolved = resolve_relative_path(
                obj.uns.get("source_base_dir"), _run_root_from_spine_path(spine_path)
            )
            base = resolved if resolved is not None else spine_path.parent
        return obj, base
    # in-memory spine
    if not bool(spine.uns.get("is_spine", False)):
        return spine, Path(base_dir) if base_dir is not None else None
    if base_dir is not None:
        return spine, Path(base_dir)
    source_base_dir = spine.uns.get("source_base_dir")
    if source_base_dir:
        source_path = Path(str(source_base_dir))
        if source_path.is_absolute():
            return spine, source_path
        raise ValueError(
            "materialize: spine.uns['source_base_dir'] is relative "
            f"({source_base_dir!r}) but the spine is in-memory with no on-disk "
            "location to resolve it against -- pass base_dir explicitly."
        )
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
    barcodes,
    read_ids,
    molecule_uids,
    obs_mask,
    start,
    end,
    candidate_read_ids: Iterable[str] | None = None,
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
    barcode_values = _as_set(barcodes)
    if barcode_values is not None:
        mask &= obs["Barcode"].astype(str).isin(barcode_values).to_numpy()
    molecule_values = _as_set(molecule_uids)
    if molecule_values is not None:
        mask &= obs["molecule_uid"].astype(str).isin(molecule_values).to_numpy()
    candidate_values = _as_set(candidate_read_ids)
    if candidate_values is not None:
        mask &= obs.index.astype(str).isin(candidate_values)
    sel = obs.loc[mask]
    rids = _as_set(read_ids)
    if rids is not None:
        sel = sel.loc[sel.index.isin(rids)]
    return sel


def _load_partition(
    path: Path,
    read_ids: list[str],
    layers: Iterable[str] | None,
    lazy: bool | None,
    start: int | None,
    end: int | None,
    query_memory_mb: int = DEFAULT_QUERY_MEMORY_MB,
) -> "ad.AnnData":
    """Load one partition, subset to ``read_ids`` (in order), keep ``layers``."""
    return read_zarr_subset(
        path,
        read_ids=read_ids,
        start=start,
        end=end,
        layers=layers,
        lazy=lazy,
        memory_mb=query_memory_mb,
    )


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
    """Materialize a spine selection from its read-relative parquet shards.

    Streams one shard at a time into a single preallocated dense grid
    (``materialize_ragged_streaming``) rather than reading every selected read's
    ragged arrays into one pandas frame first: those int64/float64 list-columns
    are ~25-50x their dense footprint, so the whole-frame read drove a nominally-
    0.5GB preprocess task to a 16-44GB peak (see dev/pipeline_scaling_audit.md).
    Peak is now ~one shard's frame + the dense output.
    """
    from .ragged_store import materialize_ragged_streaming, read_ragged_parquet

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
    lengths = _reference_lengths(spine, selection["Reference_strand"].unique())
    selected_read_ids = selection.index

    def _shard_chunks():
        # One shard per iteration, filtered to the selection's reads (a shard
        # holds many barcodes' reads); the streaming materializer scatters and
        # frees each before the next is read.
        for path in paths:
            yield read_ragged_parquet([path], read_ids=selected_read_ids)

    return materialize_ragged_streaming(
        _shard_chunks(),
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
    lazy: bool | None,
    start: int | None,
    end: int | None,
    query_memory_mb: int = DEFAULT_QUERY_MEMORY_MB,
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
    return _load_partition(
        path,
        list(selection.index),
        layers,
        lazy,
        start,
        end,
        query_memory_mb,
    )


def _load_preprocess_x_selection(
    spine: "ad.AnnData",
    selection: pd.DataFrame,
    run_root: Path | None,
    layers: Iterable[str] | None,
    start: int | None,
    end: int | None,
    lazy: bool | None,
    query_memory_mb: int,
) -> "ad.AnnData | None":
    """Read ``X`` directly from one preprocess task shard that fully covers
    ``selection``, skipping raw ragged/tiled CIGAR reconstruction entirely.

    Deliberately conservative: returns ``None`` (safe to fall back to the existing
    dispatch, never a partial/wrong answer) on any partial coverage, a selection
    spanning more than one ``Reference_strand``, or a preprocess catalog with no
    ``has_x`` column (a run written before this fast path existed). Only a *single*
    shard fully covering every requested read is considered -- stitching ``X``
    across multiple shards (e.g. a whole-reference query spanning several barcode-
    partitioned tasks) is not attempted here; that selection just falls through to
    today's raw-store path unchanged. This still covers the common case: preprocess
    tasks are barcode-partitioned by default (``dispatch_plan.plan_preprocess_
    tasks``), so any single-sample-scoped ``materialize()`` call -- e.g. the
    project layer's per-sample streaming -- hits this path.

    ``layers`` must be an explicit, *empty* collection -- never ``None`` and never
    non-empty. A preprocess shard only ever carries its own newly-computed derived
    layers, never raw-sourced ones (``read_span_mask``, sequence/quality/mismatch
    encodings, ...). ``layers=None`` means "give me everything, including raw
    layers"; a non-empty ``layers`` here means the caller wants specific
    *raw-sourced* layers (anything derived was already filtered out of this
    argument by the caller -- see ``source_layers`` in ``materialize()``). Neither
    case is satisfiable from this shard alone, so both fall back to the existing
    raw-store dispatch instead of silently dropping layers.
    """
    if run_root is None or layers is None or len(list(layers)) > 0:
        return None
    catalog_path = resolve_relative_path(spine.uns.get("preprocess_catalog"), run_root)
    if catalog_path is None or not catalog_path.exists():
        return None
    catalog = pd.read_parquet(catalog_path)
    if "has_x" not in catalog.columns:
        return None
    catalog = catalog.loc[catalog["has_x"].astype(bool)]
    if catalog.empty:
        return None

    references = selection["Reference_strand"].astype(str).unique()
    if len(references) != 1:
        return None
    candidates = catalog.loc[catalog["reference"].astype(str) == references[0]]
    if start is not None:
        candidates = candidates.loc[
            (candidates["core_start"] <= int(start)) & (candidates["core_end"] >= int(end))
        ]
    if candidates.empty:
        return None

    required = set(selection.index.astype(str))
    index_path = resolve_relative_path(spine.uns.get("preprocess_read_index"), run_root)
    indexed_by_path: dict[str, pd.DataFrame] = {}
    if index_path is not None and index_path.exists():
        indexed = query_derived_index(
            index_path,
            references=references,
            read_ids=required,
            start=start,
            end=end,
        )
        indexed_by_path = {
            str(group_path): group
            for group_path, group in indexed.groupby("group_path", sort=False, observed=True)
        }
    for record in candidates.to_dict("records"):
        path = catalog_path.parent / str(record["group_path"])
        if not path.exists():
            continue
        index_rows = indexed_by_path.get(str(record["group_path"]))
        if index_rows is not None:
            if set(index_rows["read_id"].astype(str)) != required:
                continue
            part = read_zarr_subset(
                path,
                row_indices=index_rows["group_row"].astype(int),
                start=start,
                end=end,
                layers=layers,
                lazy=lazy,
                memory_mb=query_memory_mb,
            )
        else:
            try:
                part = read_zarr_subset(
                    path,
                    read_ids=required,
                    start=start,
                    end=end,
                    layers=layers,
                    lazy=lazy,
                    memory_mb=query_memory_mb,
                )
            except KeyError:
                continue
        if part.X is None:
            continue
        if not required.issubset(set(map(str, part.obs_names))):
            continue
        return part[list(selection.index)].copy()
    return None


def _derived_layer_names(spine: "ad.AnnData", run_root: Path | None) -> set[str]:
    """Return layer names published by partitioned derived-stage stores."""
    names: set[str] = set()
    for catalog_key in ("preprocess_catalog", "hmm_catalog"):
        catalog_path = resolve_relative_path(spine.uns.get(catalog_key), run_root)
        if catalog_path is None or not catalog_path.exists():
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
    run_root: Path | None,
    lazy: bool | None,
    query_memory_mb: int,
) -> None:
    """Stitch task-partitioned derived layers onto a materialized slice.

    Looped per distinct ``Reference_strand`` present in ``result``: the derived-layer
    task catalog is keyed per reference (``core_start``/``core_end`` are only
    meaningful within one reference's own coordinate system), but a canonical-
    reference selection can legitimately span multiple strands of the same locus at
    once (e.g. ``project.catalog.project_adata`` pooling a "top" and "bottom"
    ``Reference_strand`` that share one sequence identity and therefore one shared
    position axis) -- that's the common multi-reference case this needs to support,
    not a set of unrelated loci with incompatible coordinates.
    """
    if not requested_layers:
        return
    references = result.obs["Reference_strand"].astype(str).unique()
    positions = np.asarray(result.var_names, dtype=np.int64)
    if len(references) == 0 or positions.size == 0:
        raise ValueError("derived layers require at least one non-empty reference slice")
    absent_fill = {
        **dict(spine.uns.get("preprocess_layer_absent_fill", {})),
        **dict(spine.uns.get("hmm_layer_absent_fill", {})),
    }
    assembled = {
        layer: np.full(result.shape, float(absent_fill.get(layer, np.nan)), dtype=np.float32)
        for layer in requested_layers
    }
    source_dtypes: dict[str, np.dtype] = {}
    result_positions = {int(position): column for column, position in enumerate(positions)}
    result_reference = result.obs["Reference_strand"].astype(str).to_numpy()

    for reference in references:
        result_rows = {
            str(read_id): row
            for row, (read_id, this_reference) in enumerate(zip(result.obs_names, result_reference))
            if this_reference == reference
        }
        if not result_rows:
            continue
        for stage in ("preprocess", "hmm"):
            catalog_key = f"{stage}_catalog"
            catalog_path = resolve_relative_path(spine.uns.get(catalog_key), run_root)
            if catalog_path is None:
                continue
            catalog = pd.read_parquet(catalog_path)
            catalog = catalog.loc[
                (catalog["reference"].astype(str) == reference)
                & (catalog["core_start"] < positions.max() + 1)
                & (catalog["core_end"] > positions.min())
            ]
            index_path = resolve_relative_path(spine.uns.get(f"{stage}_read_index"), run_root)
            index_active = index_path is not None and index_path.exists()
            indexed_groups: dict[str, pd.DataFrame] = {}
            if index_active:
                indexed = query_derived_index(
                    index_path,
                    references=reference,
                    read_ids=result_rows,
                    start=int(positions.min()),
                    end=int(positions.max()) + 1,
                )
                indexed_groups = {
                    str(group_path): group
                    for group_path, group in indexed.groupby(
                        "group_path", sort=False, observed=True
                    )
                }
            for record in catalog.to_dict("records"):
                group_path = str(record["group_path"])
                index_rows = indexed_groups.get(group_path)
                if index_active and index_rows is None:
                    continue
                path = catalog_path.parent / group_path
                part = read_zarr_subset(
                    path,
                    row_indices=(
                        index_rows["group_row"].astype(int) if index_rows is not None else None
                    ),
                    read_ids=None,
                    start=int(positions.min()),
                    end=int(positions.max()) + 1,
                    layers=requested_layers,
                    lazy=lazy,
                    memory_mb=query_memory_mb,
                )
                shared_reads = [
                    read_id for read_id in map(str, part.obs_names) if read_id in result_rows
                ]
                shared_positions = [
                    position
                    for position in map(int, part.var_names)
                    if position in result_positions
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


def _overlay_preprocess_var(
    spine: "ad.AnnData", result: "ad.AnnData", run_root: Path | None
) -> None:
    """Attach reduced coverage and reference-context columns to a slice."""
    if result.n_vars == 0:
        return
    path = resolve_relative_path(spine.uns.get("preprocess_var"), run_root)
    if path is None or not path.exists():
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


def _overlay_spatial_read_metrics(
    spine: "ad.AnnData",
    result: "ad.AnnData",
    requested: bool | set[str],
    run_root: Path | None,
    lazy: bool | None,
    query_memory_mb: int,
) -> None:
    """Attach spatial-stage per-read outputs (autocorrelation, Lomb-Scargle, ...).

    Unlike preprocess/hmm derived layers (read x position, same shape as ``result``),
    spatial's per-task read outputs are read x lag or read x frequency arrays -- a
    different axis than genomic position -- so they're attached to ``obs``/``obsm``
    instead of ``layers``. Gated behind the opt-in ``read_metrics`` argument (not
    loaded whenever available, unlike layers): opening every matching task's
    ``read_metrics.zarr`` isn't free, and most callers don't need it.
    """
    if not requested:
        return
    catalog_path = resolve_relative_path(spine.uns.get("spatial_task_catalog"), run_root)
    if catalog_path is None or not catalog_path.exists():
        return
    references = result.obs["Reference_strand"].astype(str).unique()
    catalog = pd.read_parquet(catalog_path)
    # group_path/obsm_keys: current naming, shared with preprocess/hmm's catalogs.
    # read_metrics_path: pre-normalization naming, kept readable for catalogs
    # written before this change -- no forced backfill.
    path_column = "group_path" if "group_path" in catalog.columns else "read_metrics_path"
    catalog = catalog.loc[
        catalog["reference"].astype(str).isin(references) & catalog[path_column].notna()
    ]
    if catalog.empty:
        return

    result_rows = {str(read_id): row for row, read_id in enumerate(result.obs_names)}
    assembled_obs: dict[str, np.ndarray] = {}
    assembled_obsm: dict[str, np.ndarray] = {}
    task_specific_uns = {"task_id", "reference", "barcode", "core_start", "core_end"}
    catalog_dir = catalog_path.parent
    want_all = requested is True
    index_path = resolve_relative_path(spine.uns.get("spatial_read_index"), run_root)
    index_active = index_path is not None and index_path.exists()
    indexed_groups: dict[str, pd.DataFrame] = {}
    if index_active:
        indexed = query_derived_index(
            index_path,
            references=references,
            read_ids=result_rows,
        )
        indexed_groups = {
            str(group_path): group
            for group_path, group in indexed.groupby("group_path", sort=False, observed=True)
        }

    for record in catalog.to_dict("records"):
        group_path = str(record[path_column])
        index_rows = indexed_groups.get(group_path)
        if index_active and index_rows is None:
            continue
        part = read_zarr_subset(
            catalog_dir / group_path,
            row_indices=(index_rows["group_row"].astype(int) if index_rows is not None else None),
            lazy=lazy,
            memory_mb=query_memory_mb,
        )
        shared_reads = [read_id for read_id in map(str, part.obs_names) if read_id in result_rows]
        if not shared_reads:
            continue
        source_rows = [part.obs_names.get_loc(read_id) for read_id in shared_reads]
        target_rows = [result_rows[read_id] for read_id in shared_reads]

        for column in part.obs.columns:
            if not want_all and column not in requested:
                continue
            values = part.obs[column].to_numpy()
            if column not in assembled_obs:
                # Absent reads (not covered by any spatial task) are filled with
                # NaN, so numeric columns are always float -- an int source dtype
                # can't hold NaN, and silently produces garbage via unsafe casting.
                numeric = np.issubdtype(values.dtype, np.number)
                assembled_obs[column] = np.full(
                    result.n_obs,
                    np.nan if numeric else None,
                    dtype=np.float64 if numeric else object,
                )
            assembled_obs[column][target_rows] = values[source_rows]

        for key in part.obsm.keys():
            if not want_all and key not in requested:
                continue
            values = np.asarray(part.obsm[key])
            if key not in assembled_obsm:
                assembled_obsm[key] = np.full(
                    (result.n_obs, values.shape[1]), np.nan, dtype=np.float32
                )
            assembled_obsm[key][target_rows, :] = values[source_rows, :]

        for uns_key, uns_value in part.uns.items():
            if uns_key in task_specific_uns:
                continue
            result.uns.setdefault(uns_key, uns_value)

    for column, values in assembled_obs.items():
        result.obs[column] = values
    for key, values in assembled_obsm.items():
        result.obsm[key] = values


def _materialize_legacy(
    spine: "ad.AnnData",
    *,
    references,
    samples,
    barcodes,
    read_ids,
    molecule_uids,
    obs_mask,
    layers,
    start,
    end,
) -> "ad.AnnData":
    """Materialize a selection directly from an already-loaded legacy AnnData.

    A legacy monolithic AnnData (pre-partitioned-store pipeline: no
    ``uns["is_spine"]``) already holds ``X`` and every layer fully in memory --
    there's no partition/ragged-store index to resolve, so this is a plain
    boolean-mask slice rather than the dense/tiled/ragged dispatch the rest of
    ``materialize()`` does. Derived-layer and read-metric overlays don't apply
    (a legacy file has no task catalog to pull them from), so callers just get
    back whatever was already in the source file.
    """
    sel = _select_spine_obs(
        spine,
        references,
        samples,
        barcodes,
        read_ids,
        molecule_uids,
        obs_mask,
        start,
        end,
    )
    if sel.shape[0] == 0:
        raise ValueError("materialize: selection matched no molecules")
    result = spine[list(sel.index)].copy()
    if start is not None:
        positions = np.asarray(result.var_names, dtype=np.int64)
        core_mask = (positions >= int(start)) & (positions < int(end))
        result = result[:, core_mask].copy()
    if layers is not None:
        keep = set(layers)
        for layer_name in list(result.layers.keys()):
            if layer_name not in keep:
                del result.layers[layer_name]
    logger.debug("materialized %d molecules from legacy monolithic spine", result.n_obs)
    return result


def materialize(
    spine,
    *,
    base_dir: str | Path | None = None,
    references=None,
    samples=None,
    barcodes=None,
    read_ids=None,
    molecule_uids=None,
    obs_mask=None,
    layers: Iterable[str] | None = None,
    read_metrics: bool | Iterable[str] = False,
    lazy: bool | None = None,
    start: int | None = None,
    end: int | None = None,
    query_memory_mb: int = DEFAULT_QUERY_MEMORY_MB,
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
        barcodes: Optional ``Barcode`` value(s) to include.
        read_ids: Optional explicit read id(s) to include.
        molecule_uids: Optional experiment-global molecule identity value(s).
        obs_mask: Optional boolean array aligned to ``spine.obs`` rows.
        layers: Optional subset of layers to load (default: all).
        read_metrics: Attach spatial-stage per-read outputs (autocorrelation,
            Lomb-Scargle periodograms, ...) if the spine has a spatial stage
            available -- ``True`` for everything found, a name subset, or ``False``
            (default) to skip. Unlike ``layers``, off by default: it's a different
            axis than genomic position (obs/obsm, not layers) and opening every
            matching task's ``read_metrics.zarr`` isn't free.
        lazy: Use ``anndata.experimental.read_lazy`` and slice before loading.
            ``None`` (default) or ``True`` prefers it with eager fallback;
            ``False`` explicitly uses the eager reader.
        start: Optional zero-based inclusive genomic start.
        end: Optional zero-based exclusive genomic end. Must accompany ``start``.
        query_memory_mb: Approximate memory budget for each projected Zarr batch.

    Returns:
        anndata.AnnData: The materialized selection, ordered as in the spine, with
        decoder/reference maps copied from the spine ``uns``.

    Raises:
        ValueError: If the selection matches no molecules or neither a complete
            dense cache nor ragged-store pointers are available.
    """
    import anndata as ad

    candidate_read_ids = None
    if (
        isinstance(spine, (str, Path))
        and Path(spine).parent.name.endswith("_outputs")
        and obs_mask is None
        and any(
            value is not None
            for value in (references, samples, barcodes, read_ids, molecule_uids, start, end)
        )
    ):
        spine_path = Path(spine)
        index_path = _run_root_from_spine_path(spine_path) / "molecule_index"
        if index_path.exists():
            try:
                indexed = query_molecule_index(
                    index_path,
                    references=references,
                    samples=samples,
                    barcodes=barcodes,
                    read_ids=read_ids,
                    molecule_uids=molecule_uids,
                    start=start,
                    end=end,
                    columns=["read_id"],
                )
            except (ImportError, KeyError) as error:
                logger.debug("molecule-index query unavailable (%s); filtering spine", error)
            else:
                candidate_read_ids = indexed["read_id"].astype(str).tolist()
                if not candidate_read_ids:
                    raise ValueError("materialize: selection matched no molecules")

    spine_obj, base = _resolve_spine(spine, base_dir)
    if not bool(spine_obj.uns.get("is_spine", False)):
        # Legacy monolithic AnnData (pre-partitioned-store pipeline): already
        # fully loaded, nothing to resolve against partitions/ragged shards.
        return _materialize_legacy(
            spine_obj,
            references=references,
            samples=samples,
            barcodes=barcodes,
            read_ids=read_ids,
            molecule_uids=molecule_uids,
            obs_mask=obs_mask,
            layers=layers,
            start=start,
            end=end,
        )
    # Cross-stage uns pointers (preprocess_catalog, hmm_catalog, preprocess_var, ...)
    # are stored relative to the run's output_directory, not `base` (which is the
    # raw store's own directory) -- see _run_root_from_spine_path. `base` is always
    # a direct child of output_directory by construction, so its parent recovers it
    # without needing the original spine_path here.
    run_root = base.parent
    requested_read_metrics: bool | set[str] = (
        read_metrics if isinstance(read_metrics, bool) else set(read_metrics)
    )
    derived_available = _derived_layer_names(spine_obj, run_root)
    requested_layer_set = None if layers is None else set(layers)
    requested_derived = (
        derived_available
        if requested_layer_set is None
        else requested_layer_set.intersection(derived_available)
    )
    source_layers = (
        None if requested_layer_set is None else requested_layer_set.difference(derived_available)
    )
    sel = _select_spine_obs(
        spine_obj,
        references,
        samples,
        barcodes,
        read_ids,
        molecule_uids,
        obs_mask,
        start,
        end,
        candidate_read_ids,
    )
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
    fast = _load_preprocess_x_selection(
        spine_obj,
        sel,
        run_root,
        source_layers,
        start,
        end,
        lazy,
        query_memory_mb,
    )
    if fast is not None:
        _overlay_preprocess_layers(
            spine_obj, fast, requested_derived, run_root, lazy, query_memory_mb
        )
        _overlay_preprocess_var(spine_obj, fast, run_root)
        _overlay_spatial_read_metrics(
            spine_obj, fast, requested_read_metrics, run_root, lazy, query_memory_mb
        )
        logger.debug("materialized %d molecules from preprocess-owned X", fast.n_obs)
        return fast
    if not _dense_cache_available(sel, base):
        tiled = _load_tiled_cache_selection(
            spine_obj,
            sel,
            base,
            source_layers,
            lazy,
            start,
            end,
            query_memory_mb,
        )
        if tiled is not None:
            _overlay_preprocess_layers(
                spine_obj, tiled, requested_derived, run_root, lazy, query_memory_mb
            )
            _overlay_preprocess_var(spine_obj, tiled, run_root)
            _overlay_spatial_read_metrics(
                spine_obj, tiled, requested_read_metrics, run_root, lazy, query_memory_mb
            )
            logger.debug("materialized %d molecules from tiled dense cache", tiled.n_obs)
            return tiled
        result = _load_ragged_selection(spine_obj, sel, base, source_layers, start, end)
        _overlay_preprocess_layers(
            spine_obj, result, requested_derived, run_root, lazy, query_memory_mb
        )
        _overlay_preprocess_var(spine_obj, result, run_root)
        _overlay_spatial_read_metrics(
            spine_obj, result, requested_read_metrics, run_root, lazy, query_memory_mb
        )
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
                query_memory_mb,
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

    _overlay_preprocess_layers(
        spine_obj, result, requested_derived, run_root, lazy, query_memory_mb
    )
    _overlay_preprocess_var(spine_obj, result, run_root)
    _overlay_spatial_read_metrics(
        spine_obj, result, requested_read_metrics, run_root, lazy, query_memory_mb
    )

    logger.debug("materialized %d molecules from %d partition(s)", result.n_obs, len(parts))
    return result
