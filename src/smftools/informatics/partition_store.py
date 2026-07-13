"""Partitioned zarr store + thin molecule-index spine + partition catalog.

This is the storage layer for the 1.0.0 output re-architecture. Instead of one
monolithic ``.h5ad`` per experiment, an experiment is written as:

- ``store/<reference>/<sample>/``: one anndata-native **zarr partition** per
  ``(Reference_strand, Sample)`` group, holding that group's ``X`` + heavy layers
  (chunked along ``obs``). Each partition keeps its own ``var``/``uns`` and is
  self-describing.
- ``spine.h5ad``: a **thin molecule-index AnnData** -- one ``obs`` row per read,
  carrying identity columns plus pointers (``partition``, ``partition_row``,
  ``bam_path``) that link each molecule to its data in the partitions / BAM. It
  holds no ``X`` or layers.
- ``catalog.parquet``: one row per partition, for fast indexed selection without
  scanning the store tree.

The partition + spine writes go through :func:`smftools.readwrite.safe_write_zarr`
/ :func:`safe_write_h5ad`, so the same sanitization + pickle-backup behavior used
by the monolithic path applies here too. Partitions are written zarr v3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from smftools.constants import (
    BARCODE,
    DATASET,
    READ_MAPPING_DIRECTION,
    REFERENCE_STRAND,
    SAMPLE,
    STRAND,
)
from smftools.logging_utils import get_logger

from .sidecar_manifest import register_sidecar, sidecar_manifest_path

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)

# Default obs-axis chunk size for partition arrays. Partitions are typically far
# smaller than this, in which case each array is a single chunk.
DEFAULT_OBS_CHUNK = 10_000

# obs columns copied onto the spine as molecule identity (only those present are used).
DEFAULT_IDENTITY_COLS = (
    REFERENCE_STRAND,
    SAMPLE,
    BARCODE,
    STRAND,
    DATASET,
    READ_MAPPING_DIRECTION,
)

STORE_SUBDIR = "store"
SPINE_FILENAME = "spine.h5ad"
CATALOG_FILENAME = "catalog.parquet"


@dataclass
class PartitionInfo:
    """Metadata for one written ``(reference, sample)`` partition.

    Attributes:
        partition_id: Stable identifier (``"<slug_ref>/<slug_sample>"``).
        reference: Exact ``Reference_strand`` value for this partition.
        sample: Exact ``Sample`` value for this partition.
        group_path: Partition zarr path, relative to the experiment output dir.
        n_reads: Number of molecules (obs rows) in the partition.
        n_positions: Number of positions (var) in the partition.
        read_ids: Read names in stored order (index i -> ``partition_row`` i).
    """

    partition_id: str
    reference: str
    sample: str
    group_path: str
    n_reads: int
    n_positions: int
    read_ids: list[str] = field(default_factory=list)


def _slug(value: str) -> str:
    """Return a filesystem-safe slug for a reference/sample name."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return slug or "x"


def write_partitioned_store(
    adata: "ad.AnnData",
    output_dir: str | Path,
    *,
    reference_col: str = REFERENCE_STRAND,
    sample_col: str = SAMPLE,
    obs_chunk: int = DEFAULT_OBS_CHUNK,
    zarr_format: int = 3,
    backup: bool = True,
    verbose: bool = False,
) -> list[PartitionInfo]:
    """Split *adata* by ``(reference, sample)`` and write one zarr partition each.

    Args:
        adata: In-memory AnnData to partition (X + layers + obs + var + uns).
        output_dir: Experiment output directory; partitions go under
            ``output_dir/store/<slug_ref>/<slug_sample>``.
        reference_col: obs column identifying the reference (default Reference_strand).
        sample_col: obs column identifying the sample (default Sample).
        obs_chunk: zarr chunk size along the obs axis.
        zarr_format: zarr on-disk format to write (default v3).
        backup: Passed to safe_write_zarr (pickle non-serializable uns/obs).
        verbose: Verbose sanitization logging.

    Returns:
        list[PartitionInfo]: One entry per written partition, with read order.

    Raises:
        KeyError: If ``reference_col`` or ``sample_col`` is missing from ``adata.obs``.
    """
    from ..readwrite import safe_write_zarr

    output_dir = Path(output_dir)
    for col in (reference_col, sample_col):
        if col not in adata.obs.columns:
            raise KeyError(f"partitioning column '{col}' not found in adata.obs")

    store_root = output_dir / STORE_SUBDIR
    store_root.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()

    refs = adata.obs[reference_col].astype(str)
    samples = adata.obs[sample_col].astype(str)
    # Deterministic ordering of partitions.
    pairs = sorted(set(zip(refs.tolist(), samples.tolist())))

    partitions: list[PartitionInfo] = []
    used_dirs: set[str] = set()
    for ref, sample in pairs:
        part_t0 = perf_counter()
        mask = (refs == ref).values & (samples == sample).values
        part = adata[mask].copy()
        n_reads = part.n_obs
        if n_reads == 0:
            continue

        rel_dir = Path(STORE_SUBDIR) / _slug(ref) / _slug(sample)
        # Guard against distinct (ref, sample) that slugify to the same path.
        base = rel_dir
        n = 1
        while str(rel_dir) in used_dirs:
            rel_dir = base.with_name(f"{base.name}_{n}")
            n += 1
        used_dirs.add(str(rel_dir))

        part_path = output_dir / rel_dir
        part_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = (min(obs_chunk, n_reads), part.n_vars)
        safe_write_zarr(
            part,
            part_path,
            backup=backup,
            verbose=verbose,
            zarr_format=zarr_format,
            chunks=chunks,
        )
        partitions.append(
            PartitionInfo(
                partition_id=f"{_slug(ref)}/{_slug(sample)}",
                reference=ref,
                sample=sample,
                group_path=rel_dir.as_posix(),
                n_reads=n_reads,
                n_positions=part.n_vars,
                read_ids=list(map(str, part.obs_names)),
            )
        )
        logger.info(
            "Wrote partition %s (%d reads x %d pos) in %.2fs",
            rel_dir,
            n_reads,
            part.n_vars,
            perf_counter() - part_t0,
        )

    logger.info(
        "Wrote %d partitions under %s in %.2fs",
        len(partitions),
        store_root,
        perf_counter() - t0,
    )
    return partitions


def build_spine(
    adata: "ad.AnnData",
    partitions: Sequence[PartitionInfo],
    *,
    identity_cols: Sequence[str] | None = None,
    bam_path: str | None = None,
    extra_uns: dict | None = None,
) -> "ad.AnnData":
    """Build the thin molecule-index spine AnnData from written partitions.

    One obs row per molecule, indexed by read name, with identity columns plus
    pointer columns (``partition`` group path, ``partition_row`` position within
    that partition, and ``bam_path``). Carries no ``X`` or layers.

    Args:
        adata: The source AnnData (for identity columns and uns pointers).
        partitions: Partition metadata from :func:`write_partitioned_store`.
        identity_cols: obs columns to copy as identity; defaults to those in
            :data:`DEFAULT_IDENTITY_COLS` that exist on ``adata``.
        bam_path: Optional aligned-BAM path recorded for every molecule.
        extra_uns: Extra key/values to store in the spine ``uns``.

    Returns:
        anndata.AnnData: The spine (obs-only) object.
    """
    import anndata as ad

    if identity_cols is None:
        identity_cols = [c for c in DEFAULT_IDENTITY_COLS if c in adata.obs.columns]

    # read_id -> pointer record, in stored order.
    records: dict[str, dict] = {}
    for p in partitions:
        for row_idx, read_id in enumerate(p.read_ids):
            records[read_id] = {
                "partition": p.group_path,
                "partition_row": row_idx,
                "Reference_strand": p.reference,
                "Sample": p.sample,
            }

    spine_obs = pd.DataFrame.from_dict(records, orient="index")
    # Attach any additional identity columns from the source obs (aligned on read_id).
    extra_ident = [c for c in identity_cols if c not in ("Reference_strand", "Sample")]
    if extra_ident:
        spine_obs = spine_obs.join(adata.obs[extra_ident])
    if bam_path is not None:
        spine_obs["bam_path"] = bam_path

    # Preserve original molecule order where possible.
    order = [rid for rid in map(str, adata.obs_names) if rid in records]
    spine_obs = spine_obs.reindex(order)

    spine = ad.AnnData(obs=spine_obs)
    spine.uns["is_spine"] = True
    spine.uns["n_partitions"] = len(partitions)
    # Carry pointers/decoders so a materialized selection is self-describing.
    for key in ("bam_paths", "References"):
        if key in adata.uns:
            spine.uns[key] = adata.uns[key]
    for key in list(adata.uns.keys()):
        if key.endswith("_map"):  # integer sequence encode/decode maps
            spine.uns[key] = adata.uns[key]
    if extra_uns:
        spine.uns.update(extra_uns)
    return spine


def write_catalog(
    catalog_path: str | Path,
    partitions: Sequence[PartitionInfo],
    *,
    experiment: str | None = None,
    modality: str | None = None,
    config_hash: str | None = None,
) -> Path:
    """Write the partition catalog parquet (one row per partition).

    Args:
        catalog_path: Output parquet path.
        partitions: Partition metadata from :func:`write_partitioned_store`.
        experiment: Optional experiment name recorded on every row.
        modality: Optional smf modality recorded on every row.
        config_hash: Optional config hash recorded on every row (staleness check).

    Returns:
        Path: The written catalog path.
    """
    catalog_path = Path(catalog_path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = [
        {
            "experiment": experiment,
            "partition_id": p.partition_id,
            "reference": p.reference,
            "sample": p.sample,
            "n_reads": p.n_reads,
            "n_positions": p.n_positions,
            "group_path": p.group_path,
            "modality": modality,
            "config_hash": config_hash,
            "created_at": now,
        }
        for p in partitions
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "experiment",
            "partition_id",
            "reference",
            "sample",
            "n_reads",
            "n_positions",
            "group_path",
            "modality",
            "config_hash",
            "created_at",
        ],
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(catalog_path, index=False)
    logger.info("Wrote catalog with %d partitions -> %s", len(df), catalog_path)
    return catalog_path


def write_experiment_store(
    adata: "ad.AnnData",
    output_dir: str | Path,
    *,
    experiment: str | None = None,
    modality: str | None = None,
    config_hash: str | None = None,
    bam_path: str | None = None,
    reference_col: str = REFERENCE_STRAND,
    sample_col: str = SAMPLE,
    obs_chunk: int = DEFAULT_OBS_CHUNK,
    zarr_format: int = 3,
    verbose: bool = False,
) -> dict[str, Path]:
    """Write a full experiment store: partitions + spine + catalog, and register them.

    Args:
        adata: The in-memory experiment AnnData to convert.
        output_dir: Experiment output directory.
        experiment: Experiment name (recorded in catalog and spine uns).
        modality: smf modality (recorded in catalog).
        config_hash: Config hash (recorded in catalog for staleness detection).
        bam_path: Aligned-BAM path recorded per molecule in the spine.
        reference_col: obs column identifying the reference.
        sample_col: obs column identifying the sample.
        obs_chunk: zarr obs-axis chunk size.
        zarr_format: zarr on-disk format (default v3).
        verbose: Verbose sanitization logging.

    Returns:
        dict[str, Path]: Paths for ``store``, ``spine``, ``catalog``, ``manifest``.
    """
    from ..readwrite import safe_write_h5ad

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()

    partitions = write_partitioned_store(
        adata,
        output_dir,
        reference_col=reference_col,
        sample_col=sample_col,
        obs_chunk=obs_chunk,
        zarr_format=zarr_format,
        verbose=verbose,
    )

    catalog_path = output_dir / CATALOG_FILENAME
    write_catalog(
        catalog_path,
        partitions,
        experiment=experiment,
        modality=modality,
        config_hash=config_hash,
    )

    spine = build_spine(
        adata,
        partitions,
        bam_path=bam_path,
        extra_uns={
            "store_root": (output_dir / STORE_SUBDIR).as_posix(),
            "catalog": catalog_path.name,
            "experiment": experiment,
            "modality": modality,
        },
    )
    spine_path = output_dir / SPINE_FILENAME
    # backup=False: the spine is fully rebuildable from partitions, so there is no
    # need to pickle-back-up its (coerced) identity/pointer columns.
    safe_write_h5ad(spine, spine_path, backup=False, verbose=verbose)

    manifest_path = sidecar_manifest_path(output_dir)
    register_sidecar(
        manifest_path,
        "partition_store",
        output_dir / STORE_SUBDIR,
        metadata={"n_partitions": len(partitions)},
    )
    register_sidecar(manifest_path, "spine", spine_path)
    register_sidecar(manifest_path, "catalog", catalog_path)

    logger.info(
        "Experiment store written: %d partitions, spine=%s, catalog=%s in %.2fs",
        len(partitions),
        spine_path.name,
        catalog_path.name,
        perf_counter() - t0,
    )
    return {
        "store": output_dir / STORE_SUBDIR,
        "spine": spine_path,
        "catalog": catalog_path,
        "manifest": manifest_path,
    }


def write_dense_cache_from_spine(
    spine_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    obs_chunk: int = DEFAULT_OBS_CHUNK,
    zarr_format: int = 3,
    verbose: bool = False,
) -> dict[str, Path]:
    """Materialize and persist a reference-partitioned dense cache.

    Each reference is densified independently from the ragged store, ordered by
    barcode, and written before the next reference is loaded. This bounds peak
    memory by the largest reference partition and preserves one shared row layout.
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad, safe_write_zarr
    from .partition_read import _resolve_ragged_paths, materialize

    spine_path = Path(spine_path)
    output_dir = Path(output_dir) if output_dir is not None else spine_path.parent
    output_spine_path = output_dir / SPINE_FILENAME
    spine, _ = safe_read_h5ad(spine_path)
    if REFERENCE_STRAND not in spine.obs:
        raise KeyError(f"spine.obs lacks required column {REFERENCE_STRAND!r}")
    ragged_paths: list[Path] = []
    if output_dir != spine_path.parent and "ragged_store" in spine.uns:
        ragged_paths = _resolve_ragged_paths(spine, spine_path.parent)
        spine.uns["ragged_store"] = [str(path.resolve()) for path in ragged_paths]
        if "ragged_shard" in spine.obs:
            spine.obs["ragged_shard"] = [
                str((spine_path.parent / str(path)).resolve())
                if not Path(str(path)).is_absolute()
                else str(path)
                for path in spine.obs["ragged_shard"]
            ]
        interval_catalog = spine.uns.get("interval_catalog")
        if interval_catalog and not Path(str(interval_catalog)).is_absolute():
            spine.uns["interval_catalog"] = str(
                (spine_path.parent / str(interval_catalog)).resolve()
            )

    store_root = output_dir / STORE_SUBDIR
    store_root.mkdir(parents=True, exist_ok=True)
    references = sorted(spine.obs[REFERENCE_STRAND].astype(str).unique())
    raw_plans = spine.uns.get("reference_plans", {})
    reference_plans = dict(raw_plans) if hasattr(raw_plans, "items") else {}
    partitions: list[PartitionInfo] = []
    catalog_rows: list[dict[str, object]] = []
    used_paths: set[str] = set()

    for reference in references:
        raw_plan = reference_plans.get(reference, {})
        plan = dict(raw_plan) if hasattr(raw_plan, "items") else {}
        cache_mode = str(plan.get("cache_mode", "full"))
        if cache_mode == "none":
            logger.info("Skipping dense cache for %s (cache_mode=none)", reference)
            continue
        if cache_mode == "tiled":
            reference_length = int(plan["reference_length"])
            tile_size = int(plan["tile_size"])
            tile_halo = int(plan["tile_halo"])
            for core_start in range(0, reference_length, tile_size):
                core_end = min(core_start + tile_size, reference_length)
                load_start = max(0, core_start - tile_halo)
                load_end = min(reference_length, core_end + tile_halo)
                reference_obs = spine.obs.loc[spine.obs[REFERENCE_STRAND].astype(str) == reference]
                core_obs = reference_obs.loc[
                    (reference_obs["reference_start"].astype("int64") < core_end)
                    & (reference_obs["reference_end"].astype("int64") > core_start)
                ]
                if core_obs.empty:
                    continue
                try:
                    dense = materialize(
                        spine_path,
                        references=reference,
                        read_ids=core_obs.index,
                        start=load_start,
                        end=load_end,
                    )
                except ValueError as exc:
                    if "selection matched no molecules" in str(exc):
                        continue
                    raise
                rel_path = (
                    Path(STORE_SUBDIR)
                    / _slug(reference)
                    / "tiles"
                    / f"{core_start:012d}-{core_end:012d}"
                )
                partition_path = output_dir / rel_path
                chunks = (min(obs_chunk, dense.n_obs), dense.n_vars)
                safe_write_zarr(
                    dense,
                    partition_path,
                    backup=True,
                    verbose=verbose,
                    zarr_format=zarr_format,
                    chunks=chunks,
                )
                catalog_rows.append(
                    {
                        "partition_id": f"{_slug(reference)}:{core_start}-{core_end}",
                        "cache_kind": "tiled",
                        "reference": reference,
                        "barcode": "*",
                        "row_start": 0,
                        "row_end": dense.n_obs,
                        "n_reads": dense.n_obs,
                        "n_positions": dense.n_vars,
                        "core_start": core_start,
                        "core_end": core_end,
                        "load_start": load_start,
                        "load_end": load_end,
                        "group_path": rel_path.as_posix(),
                    }
                )
            continue
        if cache_mode != "full":
            raise ValueError(f"unsupported cache mode {cache_mode!r} for {reference}")

        dense = materialize(spine_path, references=reference)
        order_columns = [column for column in (BARCODE, SAMPLE) if column in dense.obs]
        order_frame = dense.obs.assign(_read_id=dense.obs_names.astype(str))
        order = order_frame.sort_values(order_columns + ["_read_id"], kind="stable").index
        dense = dense[list(order)].copy()

        base_rel = Path(STORE_SUBDIR) / _slug(reference)
        rel_path = base_rel
        suffix = 1
        while rel_path.as_posix() in used_paths:
            rel_path = base_rel.with_name(f"{base_rel.name}_{suffix}")
            suffix += 1
        used_paths.add(rel_path.as_posix())
        partition_path = output_dir / rel_path
        chunks = (min(obs_chunk, dense.n_obs), dense.n_vars)
        safe_write_zarr(
            dense,
            partition_path,
            backup=True,
            verbose=verbose,
            zarr_format=zarr_format,
            chunks=chunks,
        )
        partitions.append(
            PartitionInfo(
                partition_id=_slug(reference),
                reference=reference,
                sample="*",
                group_path=rel_path.as_posix(),
                n_reads=dense.n_obs,
                n_positions=dense.n_vars,
                read_ids=list(map(str, dense.obs_names)),
            )
        )

        barcode_values = (
            dense.obs[BARCODE].astype(str)
            if BARCODE in dense.obs
            else pd.Series("*", index=dense.obs_names)
        )
        start = 0
        for barcode, group in barcode_values.groupby(barcode_values, sort=False):
            end = start + len(group)
            catalog_rows.append(
                {
                    "partition_id": _slug(reference),
                    "cache_kind": "full",
                    "reference": reference,
                    "barcode": str(barcode),
                    "row_start": start,
                    "row_end": end,
                    "n_reads": len(group),
                    "n_positions": dense.n_vars,
                    "group_path": rel_path.as_posix(),
                }
            )
            start = end

    pointers: dict[str, tuple[str, int]] = {}
    for partition in partitions:
        for row, read_id in enumerate(partition.read_ids):
            pointers[read_id] = (partition.group_path, row)
    spine.obs["partition"] = [
        pointers.get(str(read_id), ("", -1))[0] for read_id in spine.obs_names
    ]
    spine.obs["partition_row"] = [
        pointers.get(str(read_id), ("", -1))[1] for read_id in spine.obs_names
    ]
    spine.uns["store_root"] = store_root.as_posix()
    # AnnData reads string lists from HDF5 as numpy arrays. Normalize them back
    # to plain lists before safe_write_h5ad so the sanitizer preserves JSON list
    # semantics rather than storing a string representation of the array.
    for key in ("ragged_store", "signal_columns"):
        value = spine.uns.get(key)
        if isinstance(value, str) and value.startswith("["):
            quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", value)
            if quoted_items or value.strip() == "[]":
                value = quoted_items
        if value is not None and not isinstance(value, str):
            spine.uns[key] = [str(item) for item in value]

    catalog_path = output_dir / CATALOG_FILENAME
    pd.DataFrame(catalog_rows).to_parquet(catalog_path, index=False)
    spine.uns["catalog"] = catalog_path.name
    safe_write_h5ad(spine, output_spine_path, backup=False, verbose=verbose)

    manifest_path = sidecar_manifest_path(output_dir)
    register_sidecar(manifest_path, "dense_store", store_root)
    register_sidecar(manifest_path, "catalog", catalog_path)
    register_sidecar(manifest_path, "spine", output_spine_path)
    if ragged_paths:
        register_sidecar(
            manifest_path,
            "ragged_store",
            ragged_paths[0].parent,
            metadata={"shards": len(ragged_paths)},
        )
    logger.info("Wrote dense cache with %d reference partition(s)", len(partitions))
    return {
        "store": store_root,
        "spine": output_spine_path,
        "catalog": catalog_path,
        "manifest": manifest_path,
    }
