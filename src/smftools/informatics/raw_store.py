"""Write the ragged source-of-truth store and its thin molecule spine."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd

from smftools.constants import (
    BARCODE,
    DATASET,
    READ_MAPPING_DIRECTION,
    REFERENCE,
    REFERENCE_STRAND,
    SAMPLE,
    STRAND,
)
from smftools.logging_utils import get_logger

from ..readwrite import safe_write_h5ad
from .experiment_spine import write_experiment_spine
from .partition_read import relative_uns_path
from .ragged_store import (
    RAGGED_ARRAY_COLUMNS,
    READ_ID,
    validate_ragged_frame,
    write_ragged_parquet,
)
from .sidecar_manifest import register_sidecar, sidecar_manifest_path
from .stage_obs import write_stage_obs
from .storage_planner import plan_references

logger = get_logger(__name__)

RAW_SUBDIR = "raw"
RAW_SHARD_TEMPLATE = "part-{index:05d}.parquet"
INTERVAL_CATALOG_FILENAME = "interval_catalog.parquet"
MOLECULES_FILENAME = "molecules.parquet"
BARCODE_INDEX_FILENAME = "barcode_index.parquet"
SPINE_FILENAME = "spine.h5ad"
RAW_SCHEMA_VERSION = 2

_OBS_COLUMN_ALIASES = {
    REFERENCE: (REFERENCE, "reference"),
    REFERENCE_STRAND: (REFERENCE_STRAND, "reference_strand"),
    SAMPLE: (SAMPLE, "sample"),
    BARCODE: (BARCODE, "barcode"),
    STRAND: (STRAND, "strand"),
    DATASET: (DATASET, "dataset"),
    READ_MAPPING_DIRECTION: (READ_MAPPING_DIRECTION, "mapping_direction"),
}

_RAW_SCALAR_OBS_COLUMNS = (
    "BC",
    "U1",
    "U2",
    "Experiment_name",
    "Experiment_name_and_barcode",
    "Read_mismatch_trend",
    "ct_event_count",
    "ga_event_count",
    "strand_segment_purity",
    "strand_switch_position",
    "read_length",
    "read_quality",
    "reference_length",
    "mapped_length",
    "mapping_quality",
    "read_length_to_reference_length_ratio",
    "mapped_length_to_reference_length_ratio",
    "mapped_length_to_read_length_ratio",
    "max_insertion_length",
    "max_deletion_length",
)


def _build_raw_spine_obs(
    frame: pd.DataFrame,
    shard_by_read: Mapping[str, str],
    row_by_read: Mapping[str, int],
) -> pd.DataFrame:
    """Build canonical spine obs columns and ragged row pointers."""
    obs = pd.DataFrame(index=pd.Index(frame[READ_ID].astype(str), name=None))
    for canonical, candidates in _OBS_COLUMN_ALIASES.items():
        source = next((column for column in candidates if column in frame.columns), None)
        if source is not None:
            obs[canonical] = frame[source].to_numpy()

    # Scalar BAM/sidecar metrics belong on the molecule spine. Variable-length
    # arrays remain exclusively in parquet.
    excluded = set(RAGGED_ARRAY_COLUMNS) | {READ_ID}
    for column in _RAW_SCALAR_OBS_COLUMNS:
        if column not in frame or column in excluded or column in obs.columns:
            continue
        non_null = frame[column].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            continue
        obs[column] = frame[column].to_numpy()

    # Carry POD5 sequencing/signal metadata (scalar ``pod5_*`` columns). Ragged
    # arrays such as ``pod5_current_pa`` are skipped here and stay in the parquet.
    for column in frame.columns:
        column = str(column)
        if not column.startswith("pod5_") or column in obs.columns:
            continue
        non_null = frame[column].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (list, tuple)):
            continue
        obs[column] = frame[column].to_numpy()

    obs["reference_start"] = frame["reference_start"].to_numpy(dtype="int64")
    obs["reference_end"] = (frame["reference_start"] + frame["aligned_length"]).to_numpy(
        dtype="int64"
    )
    obs["aligned_length"] = frame["aligned_length"].to_numpy(dtype="int64")
    obs["ragged_shard"] = [shard_by_read[read_id] for read_id in obs.index]
    obs["ragged_row"] = [row_by_read[read_id] for read_id in obs.index]
    return obs


def _reference_path_component(reference: str) -> str:
    """Return a reversible filesystem-safe reference component."""
    return quote(str(reference), safe="._-")


def _contiguous_value_ranges(values: pd.Series) -> list[tuple[str, int, int]]:
    """Return ``(value, start_row, end_row)`` (``end_row`` exclusive) for each
    contiguous run of identical values in an already-sorted series."""
    array = values.astype(str).to_numpy()
    if array.size == 0:
        return []
    change_points = np.flatnonzero(array[1:] != array[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [array.size]))
    return [(str(array[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _write_raw_shards_streaming(
    reference_groups: Iterable[pd.DataFrame],
    output_dir: Path,
    raw_dir: Path,
    *,
    shard_size: int,
    start_bin_size: int,
    on_reference_written=None,
) -> dict[str, object]:
    """Write ragged parquet shards one reference-group at a time.

    Never holds more than one reference's ragged array data (``sequence``/
    ``quality``/``mismatch``/``modification_signal``) in memory at once --
    each group is sorted, bucketed by ``start_bin``, shard-written, and
    dropped before the next group arrives. Catalog/molecules/barcode-index
    rows and the ``shard_by_read``/``row_by_read`` pointers (all small,
    scalar bookkeeping, not ragged arrays) accumulate across groups and are
    returned once every group has been consumed.

    Groups must not repeat a ``Reference_strand`` (each reference's rows
    should arrive as a single, contiguous group) -- callers that already have
    a full frame get this for free from ``DataFrame.groupby(REFERENCE_STRAND,
    sort=True)``; a true streaming producer (e.g. one reference fully
    extracted from a coordinate-sorted BAM before the next begins) gets it
    naturally from the BAM's own reference ordering.

    Sorting a reference's own rows by ``[start_bin, sample_or_reference_start,
    read_id]`` and writing shards group-by-group like this is mathematically
    equivalent to a single global stable sort by ``[Reference_strand,
    start_bin, sample_or_reference_start, read_id]`` followed by one grouped
    shard-write pass (``Reference_strand`` is the first, outermost sort key),
    so callers that feed groups in the same reference order as today's global
    sort reproduce byte-identical shard contents and ``canonical_row``
    assignment.

    ``on_reference_written``, when given, is called as ``on_reference_written
    (reference_name, group, shard_by_read, row_by_read)`` immediately after
    that reference's shards are written -- ``group`` is the same (sorted,
    ragged-array-bearing) frame just written, ``shard_by_read``/``row_by_read``
    are the accumulated-so-far pointer dicts (already containing entries for
    this reference's own reads). Lets a streaming caller (e.g. a spine-obs
    builder) consume a reference's data immediately and let it be freed,
    instead of every reference's frame needing to stay alive until the end.
    """
    shard_paths: list[Path] = []
    catalog_rows: list[dict[str, object]] = []
    barcode_index_rows: list[dict[str, object]] = []
    molecules_rows: list[dict[str, object]] = []
    shard_by_read: dict[str, str] = {}
    row_by_read: dict[str, int] = {}
    sample_column: str | None = None
    canonical_row = 0
    n_reads_total = 0
    n_references = 0

    for group in reference_groups:
        if group is None or group.empty:
            continue
        n_references += 1
        group_t0 = perf_counter()
        group = group.reset_index(drop=True)
        group["reference_end"] = (group["reference_start"] + group["aligned_length"]).astype(
            "int64"
        )
        group["start_bin"] = (group["reference_start"] // start_bin_size).astype("int64")

        if sample_column is None:
            sample_column = next(
                (column for column in _OBS_COLUMN_ALIASES[SAMPLE] if column in group.columns),
                None,
            )
        sort_keys = ["start_bin"]
        if sample_column is not None and sample_column in group.columns:
            sort_keys.append(sample_column)
        else:
            sort_keys.append("reference_start")
        sort_keys.append(READ_ID)
        sorted_group = group.sort_values(sort_keys, kind="stable")
        reference_name = str(sorted_group[REFERENCE_STRAND].iloc[0])
        n_reads_total += len(sorted_group)

        for start_bin, bucket in sorted_group.groupby("start_bin", sort=True, observed=True):
            reference_dir = raw_dir / f"reference={_reference_path_component(reference_name)}"
            bin_dir = reference_dir / f"start_bin={int(start_bin):012d}"
            for shard_index, start in enumerate(range(0, len(bucket), shard_size)):
                shard = bucket.iloc[start : start + shard_size]
                path = bin_dir / RAW_SHARD_TEMPLATE.format(index=shard_index)
                relative_path = path.relative_to(output_dir).as_posix()
                write_ragged_parquet(shard.drop(columns=["reference_end", "start_bin"]), path)
                shard_paths.append(path)
                shard_read_ids = shard[READ_ID].astype(str).tolist()
                for row_number, read_id in enumerate(shard_read_ids):
                    shard_by_read[read_id] = relative_path
                    row_by_read[read_id] = row_number
                    molecules_rows.append(
                        {
                            "read_id": read_id,
                            "canonical_row": canonical_row,
                            REFERENCE_STRAND: reference_name,
                            **(
                                {SAMPLE: str(shard[sample_column].iloc[row_number])}
                                if sample_column is not None and sample_column in shard.columns
                                else {}
                            ),
                        }
                    )
                    canonical_row += 1
                catalog_rows.append(
                    {
                        "reference": reference_name,
                        "start_bin": int(start_bin),
                        "min_start": int(shard["reference_start"].min()),
                        "max_end": int(shard["reference_end"].max()),
                        "n_reads": len(shard),
                        "aligned_bases": int(shard["aligned_length"].sum()),
                        "group_path": relative_path,
                    }
                )
                if sample_column is not None and sample_column in shard.columns:
                    # Rows within `shard` are already Sample-sorted (see
                    # sort_keys above), so a barcode's reads are contiguous
                    # *within this shard* -- though a barcode can still
                    # straddle two shards if its reads happen to fall across a
                    # shard_size boundary.
                    for sample_value, start_row, end_row in _contiguous_value_ranges(
                        shard[sample_column]
                    ):
                        barcode_index_rows.append(
                            {
                                "reference": reference_name,
                                "start_bin": int(start_bin),
                                "group_path": relative_path,
                                "sample": sample_value,
                                "start_row": start_row,
                                "end_row": end_row,
                            }
                        )
        logger.debug(
            "Finished raw shard group reference=%s (%d reads) in %.2fs",
            reference_name,
            len(sorted_group),
            perf_counter() - group_t0,
        )
        if on_reference_written is not None:
            on_reference_written(reference_name, sorted_group, shard_by_read, row_by_read)

    if n_references == 0:
        raise ValueError("cannot write an empty raw store")
    logger.info(
        "Streamed raw shards for %d reference(s), %d reads total", n_references, n_reads_total
    )
    return {
        "shard_paths": shard_paths,
        "catalog_rows": catalog_rows,
        "molecules_rows": molecules_rows,
        "barcode_index_rows": barcode_index_rows,
        "shard_by_read": shard_by_read,
        "row_by_read": row_by_read,
        "sample_column": sample_column,
    }


def write_raw_store(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    reference_lengths: Mapping[str, int],
    shard_size: int = 100_000,
    start_bin_size: int = 1_000_000,
    analysis_mode: str = "auto",
    load_cache_mode: str = "auto",
    max_full_matrix_gb: float = 8.0,
    genome_tile_size: int = 10_000,
    genome_tile_halo: int = 1_000,
    bam_path: str | Path | None = None,
    extra_uns: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write ragged parquet shards, a thin spine, and manifest registration.

    Args:
        frame: One row per physical read using the ragged schema.
        output_dir: Experiment load output directory.
        reference_lengths: Exact lengths keyed by ``Reference_strand``.
        shard_size: Maximum reads per parquet shard.
        start_bin_size: Width used to group reads by alignment start.
        bam_path: Optional aligned BAM pointer copied to every spine row.
        extra_uns: Additional self-describing metadata for the spine.

    Returns:
        Dictionary containing ``spine``, ``ragged_store``, and ``manifest`` paths.

    Whole-frame convenience entry point: the caller already has every read in
    memory (tests, or any other caller assembling a frame directly), so this
    provides no memory-scaling benefit on its own -- ``cli/raw_adata.py::
    build_ragged_records`` is the streaming producer that gets the actual win,
    by never materializing more than one reference's frame at a time before
    calling ``_write_raw_shards_streaming`` directly. This function groups by
    ``Reference_strand`` and feeds the same streaming core, so its output
    (shard contents, catalog, molecules, barcode index) is unchanged from
    before this split -- see that function's docstring for why.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if start_bin_size <= 0:
        raise ValueError("start_bin_size must be positive")
    t0 = perf_counter()
    normalized = validate_ragged_frame(frame).reset_index(drop=True)
    if normalized.empty:
        raise ValueError("cannot write an empty raw store")

    output_dir = Path(output_dir)
    raw_dir = output_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Writing raw store: %d reads, %d reference strand(s), shard_size=%d, start_bin_size=%d",
        len(normalized),
        normalized[REFERENCE_STRAND].astype(str).nunique(),
        shard_size,
        start_bin_size,
    )

    reference_groups = (
        group
        for _, group in normalized.groupby(REFERENCE_STRAND, sort=True, observed=True)
    )
    streamed = _write_raw_shards_streaming(
        reference_groups,
        output_dir,
        raw_dir,
        shard_size=shard_size,
        start_bin_size=start_bin_size,
    )
    shard_paths = streamed["shard_paths"]
    catalog_rows = streamed["catalog_rows"]
    molecules_rows = streamed["molecules_rows"]
    barcode_index_rows = streamed["barcode_index_rows"]
    shard_by_read = streamed["shard_by_read"]
    row_by_read = streamed["row_by_read"]
    sample_column = streamed["sample_column"]

    catalog_path = output_dir / INTERVAL_CATALOG_FILENAME
    pd.DataFrame(catalog_rows).to_parquet(catalog_path, index=False)

    # Canonical molecule-ID catalog: one row per read, in the exact physical write
    # order every shard was cut from -- a standalone, cheap-to-scan statement of
    # "what molecules exist and in what order," separate from spine.h5ad (which is
    # built from `normalized`'s original, pre-sort order and stays that way -- this
    # is purely additive, no existing consumer changes).
    molecules_path = output_dir.parent / MOLECULES_FILENAME
    pd.DataFrame(molecules_rows).to_parquet(molecules_path, index=False)

    barcode_index_path: Path | None = None
    if barcode_index_rows:
        barcode_index_path = output_dir / BARCODE_INDEX_FILENAME
        pd.DataFrame(barcode_index_rows).to_parquet(barcode_index_path, index=False)

    obs = _build_raw_spine_obs(normalized, shard_by_read, row_by_read)
    if bam_path is not None:
        # Relative to the run's output_directory (not output_dir/raw_dir), not
        # absolute -- the aligned BAM lives under this same raw store (bam_outputs/),
        # and this column is copied unchanged into every downstream stage's spine
        # via spine.copy(), so it needs the same stage-independent anchor as the
        # uns cross-stage pointers. See _run_root_from_spine_path / relative_uns_path.
        obs["bam_path"] = relative_uns_path(bam_path, output_dir.parent)
    # Formal obs.parquet analog (dev/experiment_storage_schema.md, Phase 3, partial):
    # raw has no earlier stage to normalize against, so this is the full obs, written
    # alongside (not instead of) spine.h5ad -- purely additive, no consumer changes.
    obs_path = write_stage_obs(output_dir, obs)
    spine = ad.AnnData(obs=obs)
    plans = plan_references(
        normalized,
        reference_lengths,
        analysis_mode=analysis_mode,
        load_cache_mode=load_cache_mode,
        max_full_matrix_gb=max_full_matrix_gb,
        tile_size=genome_tile_size,
        tile_halo=genome_tile_halo,
    )
    spine.uns.update(
        {
            "is_spine": True,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "ragged_store": [path.relative_to(output_dir).as_posix() for path in shard_paths],
            "interval_catalog": catalog_path.relative_to(output_dir).as_posix(),
            # Relative to the run root (output_dir.parent), same anchor as bam_path
            # above, since molecules.parquet lives alongside output_dir, not inside it.
            "molecules_catalog": relative_uns_path(molecules_path, output_dir.parent),
            "reference_plans": {plan.reference: plan.to_dict() for plan in plans},
            "reference_lengths": {
                str(reference): int(length) for reference, length in reference_lengths.items()
            },
        }
    )
    if barcode_index_path is not None:
        spine.uns["barcode_index"] = barcode_index_path.relative_to(output_dir).as_posix()
    if extra_uns:
        spine.uns.update(dict(extra_uns))

    spine_path = output_dir / SPINE_FILENAME
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)
    manifest_path = sidecar_manifest_path(output_dir)
    register_sidecar(
        manifest_path,
        "ragged_store",
        raw_dir,
        metadata={"schema_version": RAW_SCHEMA_VERSION, "shards": len(shard_paths)},
    )
    register_sidecar(manifest_path, "spine", spine_path)
    register_sidecar(manifest_path, "interval_catalog", catalog_path)
    register_sidecar(manifest_path, "molecules", molecules_path)
    if barcode_index_path is not None:
        register_sidecar(manifest_path, "barcode_index", barcode_index_path)
    register_sidecar(manifest_path, "obs", obs_path)
    write_experiment_spine(output_dir.parent)
    logger.info(
        "Wrote raw store with %d reads in %d shard(s) in %.2fs",
        len(normalized),
        len(shard_paths),
        perf_counter() - t0,
    )
    return {
        "spine": spine_path,
        "ragged_store": shard_paths,
        "interval_catalog": catalog_path,
        "molecules": molecules_path,
        "barcode_index": barcode_index_path,
        "obs": obs_path,
        "manifest": manifest_path,
    }


def write_raw_store_streaming(
    reference_groups: Iterable[pd.DataFrame],
    output_dir: str | Path,
    *,
    reference_lengths: Mapping[str, int],
    shard_size: int = 100_000,
    start_bin_size: int = 1_000_000,
    analysis_mode: str = "auto",
    load_cache_mode: str = "auto",
    max_full_matrix_gb: float = 8.0,
    genome_tile_size: int = 10_000,
    genome_tile_halo: int = 1_000,
    bam_path: str | Path | None = None,
    extra_uns: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Streaming entry point: consumes one already-validated, already-attached
    frame per reference, never holding more than one reference's ragged data
    (``sequence``/``quality``/``mismatch``/``modification_signal``) in memory
    at once. ``cli/raw_adata.py::build_ragged_records`` is the intended
    producer -- it extracts and attaches one reference's rows, yields that
    frame, and only then extracts the next reference's.

    Every other artifact this writes (catalog, molecules, barcode index,
    spine obs/uns, manifest) is identical in shape to ``write_raw_store``'s --
    this function differs only in how much of the experiment it needs
    resident in memory to produce them, via ``_write_raw_shards_streaming``'s
    ``on_reference_written`` callback: each reference's ``obs`` contribution
    and ``ReferencePlan`` are built immediately after that reference's shards
    are written (while its `shard_by_read`/`row_by_read` entries are already
    available), so the reference's own frame can be dropped before the next
    one arrives -- obs/plan data is small (scalar columns only), so
    accumulating *that* across the whole run is not the memory concern this
    function exists to avoid.

    Each yielded frame must already be validated (``validate_ragged_frame``)
    by the caller -- unlike ``write_raw_store``, this function never sees the
    whole experiment at once, so it cannot run that check across references
    itself; only within-reference invariants are enforced downstream.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if start_bin_size <= 0:
        raise ValueError("start_bin_size must be positive")
    t0 = perf_counter()
    output_dir = Path(output_dir)
    raw_dir = output_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    obs_frames: list[pd.DataFrame] = []
    plans: list = []

    def _on_reference_written(reference_name, group, shard_by_read, row_by_read):
        obs_frames.append(_build_raw_spine_obs(group, shard_by_read, row_by_read))
        plans.extend(
            plan_references(
                group,
                reference_lengths,
                analysis_mode=analysis_mode,
                load_cache_mode=load_cache_mode,
                max_full_matrix_gb=max_full_matrix_gb,
                tile_size=genome_tile_size,
                tile_halo=genome_tile_halo,
            )
        )

    streamed = _write_raw_shards_streaming(
        reference_groups,
        output_dir,
        raw_dir,
        shard_size=shard_size,
        start_bin_size=start_bin_size,
        on_reference_written=_on_reference_written,
    )
    shard_paths = streamed["shard_paths"]
    catalog_rows = streamed["catalog_rows"]
    molecules_rows = streamed["molecules_rows"]
    barcode_index_rows = streamed["barcode_index_rows"]

    catalog_path = output_dir / INTERVAL_CATALOG_FILENAME
    pd.DataFrame(catalog_rows).to_parquet(catalog_path, index=False)

    molecules_path = output_dir.parent / MOLECULES_FILENAME
    pd.DataFrame(molecules_rows).to_parquet(molecules_path, index=False)

    barcode_index_path: Path | None = None
    if barcode_index_rows:
        barcode_index_path = output_dir / BARCODE_INDEX_FILENAME
        pd.DataFrame(barcode_index_rows).to_parquet(barcode_index_path, index=False)

    # Concatenating per-reference obs frames reproduces write_raw_store's own
    # obs shape (same columns, built by the same _build_raw_spine_obs), but in
    # reference-arrival order rather than the caller's original row order --
    # spine.obs order was never a documented/relied-upon contract (read_id is
    # always the lookup key), so this is a safe, deliberate difference, not a
    # regression. See raw_store.py's module docstring / write_raw_store's own
    # docstring for the equivalence argument this relies on for everything else.
    obs = pd.concat(obs_frames) if obs_frames else pd.DataFrame()
    if bam_path is not None:
        obs["bam_path"] = relative_uns_path(bam_path, output_dir.parent)
    obs_path = write_stage_obs(output_dir, obs)
    spine = ad.AnnData(obs=obs)
    spine.uns.update(
        {
            "is_spine": True,
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "ragged_store": [path.relative_to(output_dir).as_posix() for path in shard_paths],
            "interval_catalog": catalog_path.relative_to(output_dir).as_posix(),
            "molecules_catalog": relative_uns_path(molecules_path, output_dir.parent),
            "reference_plans": {plan.reference: plan.to_dict() for plan in plans},
            "reference_lengths": {
                str(reference): int(length) for reference, length in reference_lengths.items()
            },
        }
    )
    if barcode_index_path is not None:
        spine.uns["barcode_index"] = barcode_index_path.relative_to(output_dir).as_posix()
    if extra_uns:
        spine.uns.update(dict(extra_uns))

    spine_path = output_dir / SPINE_FILENAME
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)
    manifest_path = sidecar_manifest_path(output_dir)
    register_sidecar(
        manifest_path,
        "ragged_store",
        raw_dir,
        metadata={"schema_version": RAW_SCHEMA_VERSION, "shards": len(shard_paths)},
    )
    register_sidecar(manifest_path, "spine", spine_path)
    register_sidecar(manifest_path, "interval_catalog", catalog_path)
    register_sidecar(manifest_path, "molecules", molecules_path)
    if barcode_index_path is not None:
        register_sidecar(manifest_path, "barcode_index", barcode_index_path)
    register_sidecar(manifest_path, "obs", obs_path)
    write_experiment_spine(output_dir.parent)
    logger.info(
        "Streamed raw store with %d reads in %d shard(s) in %.2fs",
        len(obs),
        len(shard_paths),
        perf_counter() - t0,
    )
    return {
        "spine": spine_path,
        "ragged_store": shard_paths,
        "interval_catalog": catalog_path,
        "molecules": molecules_path,
        "barcode_index": barcode_index_path,
        "obs": obs_path,
        "manifest": manifest_path,
    }
