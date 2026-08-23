"""Write the ragged source-of-truth store and its thin molecule spine."""

from __future__ import annotations

import shutil
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
from .molecule_identity import (
    BASECALL_PARENT_READ_ID_COLUMN,
    BASECALL_READ_ID_COLUMN,
    EXPERIMENT_UID_COLUMN,
    IDENTITY_SCHEMA_VERSION,
    MOLECULE_UID_COLUMN,
    SEGMENT_ID_COLUMN,
    SEGMENT_UID_COLUMN,
    TEMPLATE_ID_COLUMN,
    molecule_uid,
    new_experiment_uid,
    segment_uid,
    validate_experiment_uid,
)
from .partition_read import relative_uns_path
from .physical_layout import portable_parquet_row_group_rows
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
SEGMENTS_FILENAME = "segments.parquet"
MOLECULE_INDEX_DIRNAME = "molecule_index"
SEGMENT_INDEX_DIRNAME = "segment_index"
BARCODE_INDEX_FILENAME = "barcode_index.parquet"
SPINE_FILENAME = "spine.h5ad"
RAW_SCHEMA_VERSION = 5

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
    "BM",
    # `F31`: without these names the columns are computed on the read frame and
    # then silently dropped here -- `_RAW_SCALAR_OBS_COLUMNS` is an allowlist,
    # so anything unnamed never reaches obs however correctly it was derived.
    "demux_type",
    "demux_type_source",
    "demux_type_confidence",
    "barcode_agreement",
    "barcode_source",
    "barcode_confidence",
    "sample_source",
    "sample_confidence",
    "read_group",
    "namespace",
    "source_read_id",
    BASECALL_READ_ID_COLUMN,
    BASECALL_PARENT_READ_ID_COLUMN,
    "identity_status",
    "identity_conflicts",
    "identity_schema_version",
    "template_id",
    "mate",
    "paired",
    "proper_pair",
    "mate_unmapped",
    "mate_reverse",
    "mate_reference",
    "mate_reference_start",
    "template_length",
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


def _normalize_segment_identities(frame: pd.DataFrame, experiment_uid: str) -> pd.DataFrame:
    """Attach versioned template, molecule, and segment identities to raw rows."""
    result = frame.copy()
    read_ids = result[READ_ID].astype(str)
    if TEMPLATE_ID_COLUMN in result:
        template_ids = result[TEMPLATE_ID_COLUMN].fillna("").astype(str)
        template_ids = template_ids.where(template_ids.str.len() > 0, read_ids)
    else:
        template_ids = read_ids

    if SEGMENT_ID_COLUMN in result:
        segment_ids = result[SEGMENT_ID_COLUMN].fillna("").astype(str)
    else:
        segment_ids = pd.Series("", index=result.index, dtype="string")
    mates = result.get("mate", pd.Series("unpaired", index=result.index)).astype(str)
    inferred_segments = mates.map({"R1": "R1", "R2": "R2"}).fillna("single")
    segment_ids = segment_ids.where(segment_ids.str.len() > 0, inferred_segments)

    result[TEMPLATE_ID_COLUMN] = template_ids
    result[SEGMENT_ID_COLUMN] = segment_ids
    result[EXPERIMENT_UID_COLUMN] = experiment_uid
    result[MOLECULE_UID_COLUMN] = [
        molecule_uid(experiment_uid, template_id) for template_id in template_ids
    ]
    result[SEGMENT_UID_COLUMN] = [
        segment_uid(experiment_uid, template_id, segment_id)
        for template_id, segment_id in zip(template_ids, segment_ids, strict=True)
    ]
    return result


def _build_segment_obs(
    frame: pd.DataFrame,
    shard_by_read: Mapping[str, str],
    row_by_read: Mapping[str, int],
) -> pd.DataFrame:
    """Build the scalar segment catalog contribution for one written group."""
    obs = _build_raw_spine_obs(frame, shard_by_read, row_by_read)
    for position, column in enumerate(
        (
            EXPERIMENT_UID_COLUMN,
            MOLECULE_UID_COLUMN,
            TEMPLATE_ID_COLUMN,
            SEGMENT_ID_COLUMN,
            SEGMENT_UID_COLUMN,
        )
    ):
        if column in obs:
            obs.drop(columns=column, inplace=True)
        obs.insert(position, column, frame[column].to_numpy())
    obs.insert(0, "segment_read_id", obs.index.astype(str))
    return obs


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """A boolean view of ``column``, or all-False when it is absent."""
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(bool)


def _collapse_single_segment_obs(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized collapse for molecules with exactly one segment (`F24`).

    Every long read is its own molecule, so on nanopore data *every* group has
    size one and the general path below spends ~1,060 us per molecule running
    group-aggregation machinery over a single row -- 22.6 minutes for the 1.28M
    molecules of one real run. Each per-group computation degenerates here:
    the minimum and maximum over one value are that value, `iloc[0]` is the row
    itself, and `segment_count` is 1.

    The conflict checks the general path performs -- multiple templates per
    molecule, conflicting sample/barcode/read_group, mixed references -- are
    *structurally impossible* within a single row, which is why skipping them
    is safe rather than merely faster. Anything with more than one segment
    still goes through the general path.
    """
    obs = frame.copy()
    obs["read_id"] = obs[TEMPLATE_ID_COLUMN].astype(str)
    obs["segment_count"] = 1
    # With one segment, `paired` alone decides: a paired read whose mate is
    # absent from this molecule is a singleton, and an unpaired one is single.
    obs["pair_state"] = np.where(_bool_column(obs, "paired"), "singleton", "single")
    obs["outer_fragment_start"] = obs["reference_start"].astype(int)
    obs["outer_fragment_end"] = obs["reference_end"].astype(int)
    obs["group_path"] = obs["ragged_shard"]
    obs["group_row"] = obs["ragged_row"].astype(int)
    return obs


def _collapse_segment_obs(segment_obs: pd.DataFrame) -> pd.DataFrame:
    """Collapse lossless segment metadata into one molecule-spine row per template."""
    if segment_obs.empty:
        return segment_obs

    # Split by segment count so the common single-segment case avoids the
    # per-group Python loop entirely (`F24`).
    counts = segment_obs.groupby(MOLECULE_UID_COLUMN, sort=False, observed=True)[
        MOLECULE_UID_COLUMN
    ].transform("size")
    singles = segment_obs.loc[counts == 1]
    multi = segment_obs.loc[counts > 1]

    frames: list[pd.DataFrame] = []
    if not singles.empty:
        frames.append(_collapse_single_segment_obs(singles))

    rows: list[pd.Series] = []
    for this_molecule_uid, group in multi.groupby(MOLECULE_UID_COLUMN, sort=False, observed=True):
        template_values = group[TEMPLATE_ID_COLUMN].astype(str).unique()
        if len(template_values) != 1:
            raise ValueError(f"molecule_uid {this_molecule_uid!r} maps to multiple templates")
        for column in (SAMPLE, BARCODE, "read_group", "namespace"):
            if column in group:
                values = group[column].dropna().astype(str).unique()
                if len(values) > 1:
                    raise ValueError(
                        f"molecule {template_values[0]!r} has conflicting {column} metadata"
                    )
        row = group.iloc[0].copy()
        segment_count = len(group)
        paired = bool(group.get("paired", pd.Series(False, index=group.index)).astype(bool).any())
        proper = bool(
            group.get("proper_pair", pd.Series(False, index=group.index)).astype(bool).all()
        )
        mate_unmapped = bool(
            group.get("mate_unmapped", pd.Series(False, index=group.index)).astype(bool).any()
        )
        pair_state = (
            "single"
            if not paired
            else "singleton"
            if mate_unmapped or segment_count == 1
            else "proper_pair"
            if proper
            else "discordant"
        )
        row["read_id"] = template_values[0]
        row["segment_count"] = segment_count
        row["pair_state"] = pair_state
        row["outer_fragment_start"] = int(group["reference_start"].min())
        row["outer_fragment_end"] = int(group["reference_end"].max())
        references = group[REFERENCE_STRAND].astype(str).unique()
        if len(references) > 1:
            row[REFERENCE_STRAND] = "mixed"
        if segment_count > 1:
            row["ragged_shard"] = ""
            row["ragged_row"] = -1
        row["group_path"] = row["ragged_shard"]
        row["group_row"] = int(row["ragged_row"])
        rows.append(row)

    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return segment_obs.iloc[0:0]
    obs = pd.concat(frames) if len(frames) > 1 else frames[0]
    # Restore the input's molecule order, which the split above disturbed.
    if len(frames) > 1:
        order = {
            uid: position
            for position, uid in enumerate(
                segment_obs[MOLECULE_UID_COLUMN].drop_duplicates().tolist()
            )
        }
        obs = obs.assign(_molecule_order=obs[MOLECULE_UID_COLUMN].map(order)).sort_values(
            "_molecule_order", kind="stable"
        )
        obs = obs.drop(columns="_molecule_order")
    obs.index = pd.Index(obs["read_id"].astype(str), name=None)
    obs.drop(
        columns=[
            column
            for column in ("segment_read_id", SEGMENT_ID_COLUMN, SEGMENT_UID_COLUMN)
            if column in obs
        ],
        inplace=True,
    )
    if obs.index.has_duplicates:
        duplicated = obs.index[obs.index.duplicated()].unique().tolist()
        raise ValueError(f"molecule template identities must be unique; repeated: {duplicated[:5]}")
    if obs[MOLECULE_UID_COLUMN].duplicated().any():
        raise ValueError("molecule_uid values must be unique on the molecule spine")
    return obs


def _resolve_experiment_uid(output_dir: Path, extra_uns: Mapping[str, object] | None) -> str:
    """Reuse the run's persisted experiment identity or create it exactly once."""
    from .experiment_manifest import read_experiment_manifest, update_experiment_manifest

    proposed = (extra_uns or {}).get(EXPERIMENT_UID_COLUMN)
    persisted = read_experiment_manifest(output_dir.parent).get(EXPERIMENT_UID_COLUMN)
    existing = None
    existing_spine = output_dir / SPINE_FILENAME
    if existing_spine.exists():
        from ..readwrite import safe_read_h5ad

        existing = safe_read_h5ad(existing_spine, verbose=False)[0].uns.get(EXPERIMENT_UID_COLUMN)
    candidates = {
        validate_experiment_uid(value)
        for value in (proposed, persisted, existing)
        if value is not None
    }
    if len(candidates) > 1:
        raise ValueError("configured experiment_uid conflicts with the persisted run identity")
    resolved = next(iter(candidates), new_experiment_uid())
    update_experiment_manifest(
        output_dir.parent,
        experiment_uid=resolved,
        identity_schema_version=IDENTITY_SCHEMA_VERSION,
    )
    return resolved


def _write_pointer_index(rows: list[dict[str, object]], output_dir: Path, *, dirname: str) -> Path:
    """Write bounded Parquet pointer-index pieces mirroring raw shard boundaries."""
    index_dir = output_dir.parent / dirname
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for group_path, group in frame.groupby("group_path", sort=True, observed=True):
        first = group.iloc[0]
        reference_dir = index_dir / _reference_path_component(str(first[REFERENCE_STRAND]))
        bin_dir = reference_dir / f"start-bin-{int(first['start_bin']):012d}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{Path(str(group_path)).name}.index.parquet"
        group = group.sort_values("group_row", kind="stable")
        group.to_parquet(
            bin_dir / filename,
            index=False,
            row_group_size=portable_parquet_row_group_rows(group),
        )
    return index_dir


def _write_molecule_index(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write molecule-to-segment pointer rows."""
    return _write_pointer_index(rows, output_dir, dirname=MOLECULE_INDEX_DIRNAME)


def _write_segment_index(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write segment-to-ragged-shard pointer rows."""
    return _write_pointer_index(rows, output_dir, dirname=SEGMENT_INDEX_DIRNAME)


def write_raw_pointer_indexes(
    segments: pd.DataFrame,
    artifact_root: str | Path,
) -> tuple[Path, Path]:
    """Rebuild molecule and segment indexes from a complete segment catalog."""
    required = {
        TEMPLATE_ID_COLUMN,
        "group_path",
        "group_row",
        REFERENCE_STRAND,
        "start_bin",
    }
    missing = sorted(required.difference(segments.columns))
    if missing:
        raise ValueError(f"segment catalog cannot rebuild pointer indexes; missing: {missing}")
    rows = segments.to_dict(orient="records")
    output_dir = Path(artifact_root) / "raw_outputs"
    segment_index = _write_segment_index(rows, output_dir)
    molecule_rows = [{**row, "read_id": str(row[TEMPLATE_ID_COLUMN])} for row in rows]
    molecule_index = _write_molecule_index(molecule_rows, output_dir)
    return molecule_index, segment_index


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
    reference_groups: Iterable[tuple[str, pd.DataFrame | None, bool]],
    output_dir: Path,
    raw_dir: Path,
    *,
    shard_size: int,
    start_bin_size: int,
    experiment_uid: str,
    on_reference_written=None,
) -> dict[str, object]:
    """Write ragged parquet shards one incoming group at a time.

    Never holds more than one incoming group's ragged array data (``sequence``/
    ``quality``/``mismatch``/``modification_signal``) in memory at once --
    each group is sorted, bucketed by ``start_bin``, shard-written, and
    dropped before the next group arrives. Catalog/molecules/barcode-index
    rows and the ``shard_by_read``/``row_by_read`` pointers (all small,
    scalar bookkeeping, not ragged arrays) accumulate across groups and are
    returned once every group has been consumed.

    Each item is ``(reference_strand, frame_or_None, is_final)``. A single
    ``reference_strand`` MAY now appear across multiple items -- a bounded-
    size streaming producer (e.g. ``cli/raw_adata.py``'s
    ``_ChromosomeGroupAccumulator``, flushing a chromosome's data in pieces
    to keep the parent process's memory experiment-size-independent, see
    dev/pipeline_scaling_audit.md) can't guarantee one reference arrives as a
    single group the way a whole-frame ``groupby`` naturally does. Per-
    ``(reference_strand, start_bin)`` shard-index numbering is tracked
    persistently across items rather than restarted per item, so repeated
    groups for the same reference append new shards instead of overwriting
    earlier ones (a real prior data-loss bug -- see
    ``_ChromosomeGroupAccumulator``'s docstring). ``frame`` may be ``None``
    (no new data, just a completion signal for a reference whose last actual
    rows arrived in an earlier item -- see ``_yield_flush_result``).

    Sorting a group's own rows by ``[start_bin, sample_or_reference_start,
    read_id]`` and writing shards group-by-group like this reproduces
    ``write_raw_store``'s shard *content* exactly when each reference arrives
    as one whole group (the historical, still-supported case) -- when a
    reference is split across multiple items instead, each item is
    independently sorted and shard-written rather than globally sorted
    first, so a ``start_bin`` that receives rows from more than one item
    ends up as more, smaller shards rather than one larger combined-and-
    sorted shard. Every row is still written exactly once, correctly sorted
    *within* its own shard, and correctly indexed -- this is a deliberate,
    accepted trade-off for bounded memory, not a correctness gap.

    ``on_reference_written``, when given, is called as ``on_reference_written
    (reference_name, group_or_None, shard_by_read, row_by_read, is_final)``
    for every item that has a non-empty frame, and additionally (with
    ``group=None``) for any item whose frame is ``None`` -- ``group`` is the
    same (sorted, ragged-array-bearing) frame just written (or ``None``),
    ``shard_by_read``/``row_by_read`` are the accumulated-so-far pointer
    dicts. Callers doing more than small scalar bookkeeping here (e.g.
    anything needing a reference's *true total* read count, like
    ``plan_references``) must accumulate across calls and only finalize when
    ``is_final`` is ``True`` -- obs-shaped bookkeeping is safe to build
    immediately per call, since a reference's rows never repeat across items.
    """
    shard_paths: list[Path] = []
    catalog_rows: list[dict[str, object]] = []
    barcode_index_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    segment_obs_frames: list[pd.DataFrame] = []
    shard_by_read: dict[str, str] = {}
    row_by_read: dict[str, int] = {}
    sample_column: str | None = None
    barcode_column: str | None = None
    canonical_row = 0
    n_reads_total = 0
    references_seen: set[str] = set()
    segment_uids_seen: set[str] = set()
    # Persists across items so a reference split into multiple groups keeps
    # numbering shards forward instead of colliding at index 0 -- the fix for
    # the overwrite bug this function's docstring describes.
    shard_index_by_bin: dict[tuple[str, int], int] = {}

    for reference_name, group, is_final in reference_groups:
        if group is None or group.empty:
            if on_reference_written is not None:
                on_reference_written(reference_name, None, shard_by_read, row_by_read, is_final)
            continue
        references_seen.add(reference_name)
        group_t0 = perf_counter()
        group = _normalize_segment_identities(group.reset_index(drop=True), experiment_uid)
        incoming_ids = group[READ_ID].astype(str)
        duplicated = incoming_ids[incoming_ids.duplicated()].unique().tolist()
        duplicated.extend(sorted(set(incoming_ids).intersection(shard_by_read)))
        if duplicated:
            preview = ", ".join(map(repr, duplicated[:5]))
            raise ValueError(
                f"raw read_id values must be experiment-global unique; repeated: {preview}"
            )
        incoming_segment_uids = group[SEGMENT_UID_COLUMN].astype(str)
        duplicated_segment_uids = (
            incoming_segment_uids[incoming_segment_uids.duplicated()].unique().tolist()
        )
        duplicated_segment_uids.extend(
            sorted(set(incoming_segment_uids).intersection(segment_uids_seen))
        )
        if duplicated_segment_uids:
            preview = ", ".join(map(repr, duplicated_segment_uids[:5]))
            raise ValueError(f"raw segment_uid values must be unique; repeated: {preview}")
        segment_uids_seen.update(incoming_segment_uids)
        group["reference_end"] = (group["reference_start"] + group["aligned_length"]).astype(
            "int64"
        )
        group["start_bin"] = (group["reference_start"] // start_bin_size).astype("int64")

        if sample_column is None:
            sample_column = next(
                (column for column in _OBS_COLUMN_ALIASES[SAMPLE] if column in group.columns),
                None,
            )
        if barcode_column is None:
            barcode_column = next(
                (column for column in _OBS_COLUMN_ALIASES[BARCODE] if column in group.columns),
                None,
            )
        sort_keys = ["start_bin"]
        if sample_column is not None and sample_column in group.columns:
            sort_keys.append(sample_column)
        else:
            sort_keys.append("reference_start")
        sort_keys.append(READ_ID)
        sorted_group = group.sort_values(sort_keys, kind="stable")
        n_reads_total += len(sorted_group)

        for start_bin, bucket in sorted_group.groupby("start_bin", sort=True, observed=True):
            reference_dir = raw_dir / f"reference={_reference_path_component(reference_name)}"
            bin_dir = reference_dir / f"start_bin={int(start_bin):012d}"
            bin_key = (reference_name, int(start_bin))
            next_shard_index = shard_index_by_bin.get(bin_key, 0)
            for offset, start in enumerate(range(0, len(bucket), shard_size)):
                shard_index = next_shard_index + offset
                shard = bucket.iloc[start : start + shard_size]
                path = bin_dir / RAW_SHARD_TEMPLATE.format(index=shard_index)
                relative_path = path.relative_to(output_dir).as_posix()
                write_ragged_parquet(shard.drop(columns=["reference_end", "start_bin"]), path)
                shard_paths.append(path)
                shard_read_ids = shard[READ_ID].astype(str).tolist()
                for row_number, read_id in enumerate(shard_read_ids):
                    shard_by_read[read_id] = relative_path
                    row_by_read[read_id] = row_number
                    source_row = shard.iloc[row_number]
                    scalar_payload = {
                        str(column): source_row[column]
                        for column in shard.columns
                        if column not in RAGGED_ARRAY_COLUMNS
                        and column not in {"reference_end", "start_bin"}
                        and not isinstance(source_row[column], (list, tuple, np.ndarray))
                    }
                    for canonical, candidates in _OBS_COLUMN_ALIASES.items():
                        source = next(
                            (column for column in candidates if column in source_row.index), None
                        )
                        if source is not None:
                            scalar_payload[canonical] = source_row[source]
                        for alias in candidates:
                            if alias != canonical:
                                scalar_payload.pop(alias, None)
                    segment_rows.append(
                        {
                            **scalar_payload,
                            "segment_read_id": read_id,
                            "canonical_row": canonical_row,
                            "reference_end": int(source_row["reference_end"]),
                            "start_bin": int(start_bin),
                            "group_path": relative_path,
                            "group_row": row_number,
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
            shard_index_by_bin[bin_key] = next_shard_index + len(range(0, len(bucket), shard_size))
        logger.debug(
            "Finished raw shard group reference=%s (%d reads) in %.2fs",
            reference_name,
            len(sorted_group),
            perf_counter() - group_t0,
        )
        if on_reference_written is not None:
            on_reference_written(reference_name, sorted_group, shard_by_read, row_by_read, is_final)
        segment_obs_frames.append(_build_segment_obs(sorted_group, shard_by_read, row_by_read))

    if not references_seen:
        raise ValueError("cannot write an empty raw store")
    logger.info(
        "Streamed raw shards for %d reference(s), %d reads total",
        len(references_seen),
        n_reads_total,
    )
    return {
        "shard_paths": shard_paths,
        "catalog_rows": catalog_rows,
        "segment_rows": segment_rows,
        "segment_obs": pd.concat(segment_obs_frames) if segment_obs_frames else pd.DataFrame(),
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
    experiment_uid = _resolve_experiment_uid(output_dir, extra_uns)
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
        (str(name), group, True)
        for name, group in normalized.groupby(REFERENCE_STRAND, sort=True, observed=True)
    )
    streamed = _write_raw_shards_streaming(
        reference_groups,
        output_dir,
        raw_dir,
        shard_size=shard_size,
        start_bin_size=start_bin_size,
        experiment_uid=experiment_uid,
    )
    shard_paths = streamed["shard_paths"]
    catalog_rows = streamed["catalog_rows"]
    segment_rows = streamed["segment_rows"]
    segment_obs = streamed["segment_obs"]
    barcode_index_rows = streamed["barcode_index_rows"]
    shard_by_read = streamed["shard_by_read"]
    row_by_read = streamed["row_by_read"]
    sample_column = streamed["sample_column"]

    catalog_path = output_dir / INTERVAL_CATALOG_FILENAME
    pd.DataFrame(catalog_rows).to_parquet(catalog_path, index=False)

    molecule_obs = _collapse_segment_obs(segment_obs)
    first_rows = (
        pd.DataFrame(segment_rows)
        .groupby(MOLECULE_UID_COLUMN, sort=False, observed=True)["canonical_row"]
        .min()
    )
    molecule_obs["canonical_row"] = molecule_obs[MOLECULE_UID_COLUMN].map(first_rows).astype(int)
    molecule_catalog = molecule_obs.sort_values("canonical_row", kind="stable")
    input_templates = (
        normalized[TEMPLATE_ID_COLUMN].fillna("").astype(str)
        if TEMPLATE_ID_COLUMN in normalized
        else normalized[READ_ID].astype(str)
    )
    input_templates = input_templates.where(
        input_templates.str.len() > 0, normalized[READ_ID].astype(str)
    )
    input_order: dict[str, int] = {}
    for position, template_id in enumerate(input_templates):
        input_order.setdefault(str(template_id), position)
    molecule_obs["_input_order"] = molecule_obs[TEMPLATE_ID_COLUMN].map(input_order)
    molecule_obs = molecule_obs.sort_values("_input_order", kind="stable").drop(
        columns="_input_order"
    )

    segments_path = output_dir.parent / SEGMENTS_FILENAME
    pd.DataFrame(segment_rows).to_parquet(segments_path, index=False)
    molecules_path = output_dir.parent / MOLECULES_FILENAME
    molecule_catalog.reset_index(drop=True).to_parquet(molecules_path, index=False)
    segment_index_path = _write_segment_index(segment_rows, output_dir)
    molecule_index_rows = [{**row, "read_id": str(row[TEMPLATE_ID_COLUMN])} for row in segment_rows]
    molecule_index_path = _write_molecule_index(molecule_index_rows, output_dir)

    barcode_index_path: Path | None = None
    if barcode_index_rows:
        barcode_index_path = output_dir / BARCODE_INDEX_FILENAME
        pd.DataFrame(barcode_index_rows).to_parquet(barcode_index_path, index=False)

    obs = molecule_obs.copy()
    obs.index.name = None
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
            "molecule_index": relative_uns_path(molecule_index_path, output_dir.parent),
            "segments_catalog": relative_uns_path(segments_path, output_dir.parent),
            "segment_index": relative_uns_path(segment_index_path, output_dir.parent),
            EXPERIMENT_UID_COLUMN: experiment_uid,
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "alignment_segment_policy": {
                "primary": "include",
                "secondary": "exclude",
                "supplementary": "exclude",
            },
            "pair_consensus_contract": {
                "schema_version": 1,
                "conversion": "conversion_quality_consensus",
                "deaminase": "conversion_quality_consensus",
                "direct": "unsupported",
            },
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
    register_sidecar(manifest_path, "molecule_index", molecule_index_path)
    register_sidecar(manifest_path, "segments", segments_path)
    register_sidecar(manifest_path, "segment_index", segment_index_path)
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
        "molecule_index": molecule_index_path,
        "segments": segments_path,
        "segment_index": segment_index_path,
        "barcode_index": barcode_index_path,
        "obs": obs_path,
        "manifest": manifest_path,
    }


def write_raw_store_streaming(
    reference_groups: Iterable[tuple[str, pd.DataFrame | None, bool]],
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
    refresh_experiment_spine: bool = True,
) -> dict[str, object]:
    """Streaming entry point: consumes bounded-size, possibly-repeated-per-
    reference groups (see ``_write_raw_shards_streaming``), never holding
    more than one incoming group's ragged data (``sequence``/``quality``/
    ``mismatch``/``modification_signal``) in memory at once.
    ``cli/raw_adata.py::build_ragged_records_streaming`` is the intended
    producer -- for a low-read-depth reference it may still extract and
    attach one reference's rows as a single group, but for a high-read-depth
    one it flushes that reference's data across several bounded-size groups
    instead of accumulating the whole thing first (see
    dev/pipeline_scaling_audit.md).

    Every other artifact this writes (catalog, molecules, barcode index,
    spine obs/uns, manifest) is identical in shape to ``write_raw_store``'s --
    this function differs only in how much of the experiment it needs
    resident in memory to produce them. Each group's ``obs`` contribution is
    built and appended immediately (safe regardless of how many groups a
    reference is split into -- a read's row never repeats across groups), but
    ``plan_references`` needs a reference's *true total* read count to choose
    a correct locus/genome plan, so it can't be computed from a partial
    group. This function accumulates a per-reference row count (a handful of
    integers, not ragged data) across a reference's groups and only calls
    ``plan_references`` -- using a synthetic single-column frame that
    reproduces the true final ``Reference_strand`` value_counts without ever
    holding the real ragged rows -- once ``is_final`` is ``True`` for that
    reference.

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
    experiment_uid = _resolve_experiment_uid(output_dir, extra_uns)
    raw_dir = output_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    plans: list = []
    reads_seen_by_reference: dict[str, int] = {}

    def _on_reference_written(reference_name, group, shard_by_read, row_by_read, is_final):
        if group is not None:
            reads_seen_by_reference[reference_name] = reads_seen_by_reference.get(
                reference_name, 0
            ) + len(group)
        if not is_final:
            return
        n_reads = reads_seen_by_reference.pop(reference_name, 0)
        if n_reads == 0:
            return
        synthetic_frame = pd.DataFrame({REFERENCE_STRAND: [reference_name] * n_reads})
        plans.extend(
            plan_references(
                synthetic_frame,
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
        experiment_uid=experiment_uid,
        on_reference_written=_on_reference_written,
    )
    shard_paths = streamed["shard_paths"]
    catalog_rows = streamed["catalog_rows"]
    segment_rows = streamed["segment_rows"]
    segment_obs = streamed["segment_obs"]
    barcode_index_rows = streamed["barcode_index_rows"]

    catalog_path = output_dir / INTERVAL_CATALOG_FILENAME
    pd.DataFrame(catalog_rows).to_parquet(catalog_path, index=False)

    molecule_obs = _collapse_segment_obs(segment_obs)
    first_rows = (
        pd.DataFrame(segment_rows)
        .groupby(MOLECULE_UID_COLUMN, sort=False, observed=True)["canonical_row"]
        .min()
    )
    molecule_obs["canonical_row"] = molecule_obs[MOLECULE_UID_COLUMN].map(first_rows).astype(int)
    molecule_obs = molecule_obs.sort_values("canonical_row", kind="stable")

    segments_path = output_dir.parent / SEGMENTS_FILENAME
    pd.DataFrame(segment_rows).to_parquet(segments_path, index=False)
    molecules_path = output_dir.parent / MOLECULES_FILENAME
    molecule_obs.reset_index(drop=True).to_parquet(molecules_path, index=False)
    segment_index_path = _write_segment_index(segment_rows, output_dir)
    molecule_index_rows = [{**row, "read_id": str(row[TEMPLATE_ID_COLUMN])} for row in segment_rows]
    molecule_index_path = _write_molecule_index(molecule_index_rows, output_dir)

    barcode_index_path: Path | None = None
    if barcode_index_rows:
        barcode_index_path = output_dir / BARCODE_INDEX_FILENAME
        pd.DataFrame(barcode_index_rows).to_parquet(barcode_index_path, index=False)

    obs = molecule_obs.copy()
    obs.index.name = None
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
            "molecule_index": relative_uns_path(molecule_index_path, output_dir.parent),
            "segments_catalog": relative_uns_path(segments_path, output_dir.parent),
            "segment_index": relative_uns_path(segment_index_path, output_dir.parent),
            EXPERIMENT_UID_COLUMN: experiment_uid,
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "alignment_segment_policy": {
                "primary": "include",
                "secondary": "exclude",
                "supplementary": "exclude",
            },
            "pair_consensus_contract": {
                "schema_version": 1,
                "conversion": "conversion_quality_consensus",
                "deaminase": "conversion_quality_consensus",
                "direct": "unsupported",
            },
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
    register_sidecar(manifest_path, "molecule_index", molecule_index_path)
    register_sidecar(manifest_path, "segments", segments_path)
    register_sidecar(manifest_path, "segment_index", segment_index_path)
    if barcode_index_path is not None:
        register_sidecar(manifest_path, "barcode_index", barcode_index_path)
    register_sidecar(manifest_path, "obs", obs_path)
    if refresh_experiment_spine:
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
        "molecule_index": molecule_index_path,
        "segments": segments_path,
        "segment_index": segment_index_path,
        "barcode_index": barcode_index_path,
        "obs": obs_path,
        "manifest": manifest_path,
    }
