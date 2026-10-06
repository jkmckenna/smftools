"""Bounded, partition-aware reads for immutable machine-learning manifests.

The adapter in this module keeps storage paths outside scientific manifests.
Callers inject the current stage-spine locations for each experiment, while the
dataset and split manifests remain the authority for row order, labels, channel
semantics, and train/validation/test membership.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from smftools.informatics.molecule_identity import validate_experiment_uid
from smftools.informatics.partition_read import materialize

from ..contracts import (
    InputChannelSchema,
    validate_mask_arrays,
    validate_mask_relationships,
)
from ..manifests import DatasetSnapshotManifest, SplitManifest

DEFAULT_BATCH_SIZE = 64
DEFAULT_BATCH_MEMORY_BYTES = 64 * 1024**2
DEFAULT_MATERIALIZATION_MEMORY_BYTES = 2 * 1024**3
# Rows decoded per store read before cutting batches (`MLX-09`): a read has
# a large fixed cost (spine load, store sections), so it is amortized.
DEFAULT_BLOCK_MEMORY_BYTES = 512 * 1024**2
DEFAULT_QUERY_MEMORY_MB = 64


class MLPartitionDataError(ValueError):
    """Raised when manifest-backed partition data cannot be read safely."""


class MLMemoryBudgetError(MLPartitionDataError):
    """Raised before a read whose conservative estimate exceeds its budget."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MLPartitionDataError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ExperimentPartitionSource:
    """Local stage-spine bindings for one manifest experiment.

    Paths are execution-time bindings and deliberately do not participate in
    dataset identity. The immutable source fingerprints in the dataset snapshot
    remain the scientific provenance authority.
    """

    experiment_uid: str
    modality: str
    stage_spines: Mapping[str, Path]
    # Optional per-stage read index (read_id -> store partition), used only to
    # order reads partition-major (`MLX-09`). Without it reads keep manifest order.
    stage_read_indexes: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_uid", validate_experiment_uid(self.experiment_uid))
        object.__setattr__(
            self,
            "stage_read_indexes",
            MappingProxyType(
                {
                    str(stage).strip().lower(): Path(path).resolve()
                    for stage, path in self.stage_read_indexes.items()
                }
            ),
        )
        modality = str(self.modality).strip().lower()
        if not modality:
            raise MLPartitionDataError("source modality must be a non-empty string")
        object.__setattr__(self, "modality", modality)
        normalized: dict[str, Path] = {}
        for stage, path in self.stage_spines.items():
            name = str(stage).strip().lower()
            if not name:
                raise MLPartitionDataError("source stage names must be non-empty")
            if name in normalized:
                raise MLPartitionDataError(f"duplicate source stage {name!r}")
            normalized[name] = Path(path).resolve()
        if not normalized:
            raise MLPartitionDataError("source must bind at least one stage spine")
        object.__setattr__(self, "stage_spines", MappingProxyType(normalized))


@dataclass(frozen=True)
class PartitionReadPolicy:
    """Explicit batch, query, and full-materialization memory limits."""

    batch_size: int = DEFAULT_BATCH_SIZE
    max_batch_bytes: int = DEFAULT_BATCH_MEMORY_BYTES
    max_materialization_bytes: int = DEFAULT_MATERIALIZATION_MEMORY_BYTES
    query_memory_mb: int = DEFAULT_QUERY_MEMORY_MB
    lazy: bool | None = None
    max_block_bytes: int = DEFAULT_BLOCK_MEMORY_BYTES

    def __post_init__(self) -> None:
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.max_block_bytes, "max_block_bytes")
        _positive_integer(self.max_batch_bytes, "max_batch_bytes")
        _positive_integer(self.max_materialization_bytes, "max_materialization_bytes")
        _positive_integer(self.query_memory_mb, "query_memory_mb")
        if self.lazy not in {None, True, False}:
            raise MLPartitionDataError("lazy must be true, false, or null")


@dataclass(frozen=True)
class PartitionReadEntry:
    """One deterministic manifest row in the executable read plan."""

    order_index: int
    molecule_uid: str
    experiment_uid: str
    read_id: str
    reference: str
    modality: str
    class_id: int | None
    split: str
    # Where the read is stored, e.g. (group_path, group_row); () when unknown.
    read_key: tuple = ()


@dataclass(frozen=True)
class MLPartitionDataPlan:
    """Validated row, channel, coordinate, and local-source read plan."""

    plan_id: str
    dataset: DatasetSnapshotManifest
    split: SplitManifest
    sources: Mapping[str, ExperimentPartitionSource]
    entries: tuple[PartitionReadEntry, ...]
    coordinate_start: int
    coordinate_end: int
    coordinates: np.ndarray
    bytes_per_row: int
    effective_batch_size: int
    policy: PartitionReadPolicy
    # `MLX-03`: canonical reference -> frame position by source position (-1:
    # none), and the source span of each mapped reference that covers the
    # selected frame positions.
    coordinate_maps: Mapping[str, np.ndarray] = field(default_factory=dict)
    read_spans: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(self, "entries", tuple(self.entries))
        coordinates = np.asarray(self.coordinates, dtype=np.int64).copy()
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)

    def entries_for(self, split: str) -> tuple[PartitionReadEntry, ...]:
        """Return rows for one represented split role in canonical order."""
        rows = tuple(entry for entry in self.entries if entry.split == split)
        if not rows:
            represented = sorted({entry.split for entry in self.entries})
            raise MLPartitionDataError(
                f"split role {split!r} is absent; represented roles are {represented}"
            )
        return rows

    def read_order(self, split: str) -> tuple[PartitionReadEntry, ...]:
        """Rows of one split in the order batches read them (`MLX-09`).

        Partition-major when read keys are known: canonical order scatters a
        batch over many store partitions, each opened from scratch, so a
        64-row batch on a real store cost about a minute (`F67`). Rows without
        a key keep canonical order, so datasets bound without read indexes
        read exactly as before.
        """
        rows = self.entries_for(split)
        if not any(entry.read_key for entry in rows):
            return rows
        return tuple(
            sorted(
                rows,
                key=lambda entry: (entry.experiment_uid, entry.read_key, entry.order_index),
            )
        )

    def estimate_batch_bytes(self, n_rows: int) -> int:
        """Return the conservative peak estimate for one decoded batch."""
        if n_rows < 0:
            raise MLPartitionDataError("n_rows cannot be negative")
        return max(1, n_rows) * self.bytes_per_row

    def estimate_materialization_bytes(self, split: str) -> int:
        """Return a conservative estimate including concatenation transients."""
        return 3 * len(self.entries_for(split)) * self.bytes_per_row


@dataclass(frozen=True)
class MLPartitionBatch:
    """One ordered fixed-width batch with distinct scientific masks."""

    order_indices: np.ndarray
    molecule_uids: tuple[str, ...]
    read_ids: tuple[str, ...]
    experiment_uids: tuple[str, ...]
    modalities: tuple[str, ...]
    coordinates: np.ndarray
    channel_names: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray | None
    observed_mask: np.ndarray
    availability_mask: np.ndarray
    design_mask: np.ndarray
    padding_mask: np.ndarray

    @property
    def X(self) -> np.ndarray:
        """Return a read-by-feature view suitable for sklearn pipelines."""
        return self.values.reshape(self.values.shape[0], -1)

    @property
    def y(self) -> np.ndarray | None:
        """Return integer class labels, if the dataset is supervised."""
        return self.labels

    def mask_arrays(self, dataset: DatasetSnapshotManifest) -> Mapping[str, np.ndarray]:
        """Return only masks declared by the supplied dataset input schema."""
        values = {
            "observed": self.observed_mask,
            "availability": self.availability_mask,
            "design": self.design_mask,
            "padding": self.padding_mask,
        }
        return MappingProxyType(
            {
                mask.name: values[mask.kind]
                for mask in dataset.input_schema.masks
                if mask.kind in values
            }
        )


@dataclass(frozen=True)
class MLMaterializedPartitionData:
    """A preflight-approved full split assembled from bounded batches."""

    split: str
    molecule_uids: tuple[str, ...]
    read_ids: tuple[str, ...]
    experiment_uids: tuple[str, ...]
    modalities: tuple[str, ...]
    coordinates: np.ndarray
    channel_names: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray | None
    observed_mask: np.ndarray
    availability_mask: np.ndarray
    design_mask: np.ndarray
    padding_mask: np.ndarray

    @property
    def X(self) -> np.ndarray:
        """Return the fixed-order dense feature matrix without copying."""
        return self.values.reshape(self.values.shape[0], -1)

    @property
    def y(self) -> np.ndarray | None:
        """Return integer class labels, if present."""
        return self.labels


def _coordinates(
    dataset: DatasetSnapshotManifest,
    start: int | None,
    end: int | None,
) -> np.ndarray:
    """Feature coordinates: one window, or several disjoint ones (`MLX-02`)."""
    if start is None and end is None:
        matching = sorted(
            (interval.start, interval.end)
            for interval in dataset.selection.intervals
            if interval.reference == dataset.input_schema.reference
        )
        if len(matching) > 1:
            for (_, previous_end), (next_start, _) in zip(matching, matching[1:]):
                if next_start < previous_end:
                    raise MLPartitionDataError("dataset intervals overlap")
            width = sum(window_end - window_start for window_start, window_end in matching)
            if width != dataset.input_schema.n_positions:
                raise MLPartitionDataError(
                    "coordinate width does not match input_schema.n_positions: "
                    f"{width} != {dataset.input_schema.n_positions}"
                )
            return np.concatenate(
                [
                    np.arange(window_start, window_end, dtype=np.int64)
                    for window_start, window_end in matching
                ]
            )
    window_start, window_end = _interval(dataset, start, end)
    return np.arange(window_start, window_end, dtype=np.int64)


def _interval(
    dataset: DatasetSnapshotManifest,
    start: int | None,
    end: int | None,
) -> tuple[int, int]:
    if (start is None) != (end is None):
        raise MLPartitionDataError("coordinate start and end must be provided together")
    if start is None:
        matching = [
            interval
            for interval in dataset.selection.intervals
            if interval.reference == dataset.input_schema.reference
        ]
        if matching:
            start, end = matching[0].start, matching[0].end
        else:
            filter_start = dataset.selection.filters.get("start")
            filter_end = dataset.selection.filters.get("end")
            if filter_start is not None or filter_end is not None:
                start, end = filter_start, filter_end
            else:
                start, end = 0, dataset.input_schema.n_positions
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise MLPartitionDataError("coordinates must satisfy 0 <= start < end")
    if end - start != dataset.input_schema.n_positions:
        raise MLPartitionDataError(
            "coordinate width does not match input_schema.n_positions: "
            f"{end - start} != {dataset.input_schema.n_positions}"
        )
    return start, end


def _bytes_per_row(n_positions: int, n_channels: int, *, labeled: bool) -> int:
    # values float32; observed/design bool per value; availability bool per
    # channel; padding bool per position; labels/order use int64. A 2x transient
    # allowance covers AnnData/Zarr projection and conversion copies.
    persistent = (
        n_positions * n_channels * (4 + 1 + 1)
        + n_channels
        + n_positions
        + 8
        + (8 if labeled else 0)
    )
    return 2 * persistent


def _slice_batch(batch: MLPartitionBatch, rows: slice) -> MLPartitionBatch:
    """Rows of a decoded block as their own batch."""
    return MLPartitionBatch(
        order_indices=batch.order_indices[rows],
        molecule_uids=batch.molecule_uids[rows],
        read_ids=batch.read_ids[rows],
        experiment_uids=batch.experiment_uids[rows],
        modalities=batch.modalities[rows],
        coordinates=batch.coordinates,
        channel_names=batch.channel_names,
        values=batch.values[rows],
        labels=None if batch.labels is None else batch.labels[rows],
        observed_mask=batch.observed_mask[rows],
        availability_mask=batch.availability_mask[rows],
        design_mask=batch.design_mask[rows] if batch.design_mask.ndim == 3 else batch.design_mask,
        padding_mask=batch.padding_mask[rows],
    )


def _read_keys(
    dataset: DatasetSnapshotManifest,
    bindings: Mapping[str, ExperimentPartitionSource],
) -> dict[str, tuple]:
    """``molecule_uid -> (group_path, group_row)`` from each source's read index.

    Uses the first required stage that has a read index; reads it lacks, and
    experiments without one, get no key.
    """
    import pyarrow.dataset as pa_ds

    wanted: dict[str, set[str]] = defaultdict(set)
    for observation in dataset.observations:
        wanted[observation.experiment_uid].add(observation.molecule_uid)
    keys: dict[str, tuple] = {}
    for experiment_uid, binding in bindings.items():
        stages = sorted(set(binding.stage_spines).intersection(binding.stage_read_indexes))
        if not stages:
            continue
        index = binding.stage_read_indexes[stages[0]]
        if not index.exists():
            continue
        source = pa_ds.dataset(index, format="parquet", partitioning="hive")
        columns = ["group_path", "group_row", "molecule_uid"]
        if not set(columns).issubset(source.schema.names):
            continue
        frame = source.to_table(columns=columns).to_pandas()
        frame = frame.loc[frame["molecule_uid"].astype(str).isin(wanted[experiment_uid])]
        for uid, group_path, group_row in zip(
            frame["molecule_uid"].astype(str),
            frame["group_path"].astype(str),
            frame["group_row"],
            strict=True,
        ):
            keys[uid] = (group_path, int(group_row))
    return keys


def build_partition_data_plan(
    dataset: DatasetSnapshotManifest,
    split: SplitManifest,
    sources: Sequence[ExperimentPartitionSource],
    *,
    coordinate_start: int | None = None,
    coordinate_end: int | None = None,
    policy: PartitionReadPolicy | None = None,
    coordinate_maps: Mapping[str, np.ndarray] | None = None,
) -> MLPartitionDataPlan:
    """Bind immutable manifests to local partition sources and preflight batches."""
    split.validate_against(dataset)
    policy = policy or PartitionReadPolicy()
    coordinates = _coordinates(dataset, coordinate_start, coordinate_end)
    start, end = int(coordinates[0]), int(coordinates[-1]) + 1
    bindings = {source.experiment_uid: source for source in sources}
    if len(bindings) != len(sources):
        raise MLPartitionDataError("experiment source bindings must have unique experiment UIDs")
    expected = {source.experiment_uid: source.modality for source in dataset.sources}
    missing = sorted(set(expected).difference(bindings))
    unknown = sorted(set(bindings).difference(expected))
    if missing or unknown:
        raise MLPartitionDataError(
            f"source bindings must cover dataset experiments exactly; "
            f"missing={missing}, unknown={unknown}"
        )
    for experiment_uid, modality in expected.items():
        binding = bindings[experiment_uid]
        if binding.modality != modality:
            raise MLPartitionDataError(
                f"source modality for {experiment_uid!r} is {binding.modality!r}, "
                f"expected {modality!r}"
            )
        required_stages = {
            source.stage
            for channel in dataset.input_schema.channels
            for source in channel.sources
            if source.modality == modality
        }
        absent = sorted(required_stages.difference(binding.stage_spines))
        if absent:
            raise MLPartitionDataError(
                f"source {experiment_uid!r} lacks required stage spines: {absent}"
            )
        nonexistent = sorted(
            str(binding.stage_spines[stage])
            for stage in required_stages
            if not binding.stage_spines[stage].is_file()
        )
        if nonexistent:
            raise MLPartitionDataError(f"required stage spines do not exist: {nonexistent}")

    assignments = {member.molecule_uid: member.split for member in split.members}
    read_keys = _read_keys(dataset, bindings)
    entries = tuple(
        PartitionReadEntry(
            order_index=index,
            molecule_uid=observation.molecule_uid,
            experiment_uid=observation.experiment_uid,
            read_id=observation.read_id,
            reference=observation.reference,
            modality=observation.modality,
            class_id=observation.class_id,
            split=assignments[observation.molecule_uid],
            read_key=read_keys.get(observation.molecule_uid, ()),
        )
        for index, observation in enumerate(dataset.observations)
    )
    row_bytes = _bytes_per_row(
        dataset.input_schema.n_positions,
        len(dataset.input_schema.channels),
        labeled=dataset.label_schema is not None,
    )
    if row_bytes > policy.max_batch_bytes:
        raise MLMemoryBudgetError(
            "one decoded row exceeds max_batch_bytes: "
            f"estimated {row_bytes:,} bytes > budget {policy.max_batch_bytes:,} bytes"
        )
    effective_batch_size = min(policy.batch_size, policy.max_batch_bytes // row_bytes)
    identity = {
        "dataset_snapshot_id": dataset.snapshot_id,
        "split_id": split.split_id,
        "input_schema_hash": dataset.input_schema.schema_hash,
        "coordinate_start": start,
        "coordinate_end": end,
    }
    if len(coordinates) != end - start:
        # Several windows: which positions, not just their bounds.
        identity["coordinates_sha256"] = _sha256(coordinates.tolist())
    maps = {
        str(reference): np.asarray(frame_of, dtype=np.int64)
        for reference, frame_of in (coordinate_maps or {}).items()
    }
    if dataset.input_schema.reference in maps:
        raise MLPartitionDataError("the frame reference cannot also be mapped onto itself")
    read_spans: dict[str, tuple[int, int]] = {}
    if maps:
        identity["coordinate_maps_sha256"] = _sha256(
            {reference: frame_of.tolist() for reference, frame_of in sorted(maps.items())}
        )
        for reference, frame_of in maps.items():
            sources = np.flatnonzero(np.isin(frame_of, coordinates))
            if sources.size:
                read_spans[reference] = (int(sources[0]), int(sources[-1]) + 1)
    return MLPartitionDataPlan(
        plan_id=_sha256(identity),
        dataset=dataset,
        split=split,
        sources=bindings,
        entries=entries,
        coordinate_start=start,
        coordinate_end=end,
        coordinates=coordinates,
        bytes_per_row=row_bytes,
        effective_batch_size=effective_batch_size,
        policy=policy,
        coordinate_maps=maps,
        read_spans=read_spans,
    )


def _channel_source(channel: InputChannelSchema, modality: str):
    matches = [source for source in channel.sources if source.modality == modality]
    if len(matches) > 1:
        raise MLPartitionDataError(
            f"channel {channel.name!r} has multiple physical sources for modality {modality!r}"
        )
    return matches[0] if matches else None


def _dense(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values)


def _position_columns(
    var_names: Sequence[Any], coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        source_positions = np.asarray(var_names, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise MLPartitionDataError("partition position names must be integer coordinates") from exc
    target = {int(position): index for index, position in enumerate(coordinates)}
    source_columns = []
    target_columns = []
    for source_column, position in enumerate(source_positions):
        target_column = target.get(int(position))
        if target_column is not None:
            source_columns.append(source_column)
            target_columns.append(target_column)
    return np.asarray(source_columns, dtype=np.int64), np.asarray(target_columns, dtype=np.int64)


def _mapped_position_columns(
    var_names: Sequence[Any], coordinates: np.ndarray, frame_of: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`_position_columns` for a mapped reference: source positions -> frame ones."""
    try:
        source_positions = np.asarray(var_names, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise MLPartitionDataError("partition position names must be integer coordinates") from exc
    in_map = (source_positions >= 0) & (source_positions < len(frame_of))
    frame_positions = np.full(len(source_positions), -1, dtype=np.int64)
    frame_positions[in_map] = frame_of[source_positions[in_map]]
    return _position_columns(frame_positions.tolist(), coordinates)


def _design_columns(var, reference: str, site_context: str) -> np.ndarray | None:
    candidates = (
        f"{reference}_{site_context}_site",
        f"{reference}_{site_context}_site_valid_coverage",
    )
    for column in candidates:
        if column in var:
            return np.asarray(var[column]).astype(bool)
    return None


class PartitionDataset:
    """Read deterministic batches or bounded sklearn-ready split matrices."""

    def __init__(self, plan: MLPartitionDataPlan):
        self.plan = plan

    def iter_batches(
        self,
        split: str,
        *,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterator[MLPartitionBatch]:
        """Yield non-overlapping deterministic batches for one worker shard."""
        _positive_integer(num_workers, "num_workers")
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            raise MLPartitionDataError("worker_id must be an integer")
        if worker_id < 0 or worker_id >= num_workers:
            raise MLPartitionDataError("worker_id must satisfy 0 <= worker_id < num_workers")
        entries = self.plan.read_order(split)
        batch_size = self.plan.effective_batch_size
        # Read whole blocks of batches at once and cut them up: each store read
        # has a fixed cost of seconds on real stores (`F67`). Workers take whole
        # blocks (block i -> worker i % num_workers), so each block is decoded
        # by exactly one worker; sharding by batch put batches of every block on
        # every worker, so N workers decoded every block N times (`F69`).
        batches_per_block = max(
            1, self.plan.policy.max_block_bytes // (self.plan.bytes_per_row * batch_size)
        )
        block_rows = batch_size * batches_per_block
        for block_index, block_offset in enumerate(range(0, len(entries), block_rows)):
            if block_index % num_workers != worker_id:
                continue
            block_entries = entries[block_offset : block_offset + block_rows]
            block = self._read_batch(block_entries)
            for start in range(0, len(block_entries), batch_size):
                yield _slice_batch(block, slice(start, start + batch_size))

    def materialize(self, split: str) -> MLMaterializedPartitionData:
        """Materialize one split only after a conservative peak-memory preflight."""
        estimate = self.plan.estimate_materialization_bytes(split)
        budget = self.plan.policy.max_materialization_bytes
        if estimate > budget:
            raise MLMemoryBudgetError(
                f"split {split!r} materialization is estimated at {estimate:,} bytes, "
                f"above the {budget:,}-byte budget; use iter_batches() or raise the explicit budget"
            )
        batches = tuple(self.iter_batches(split))
        if not batches:
            raise MLPartitionDataError(f"split role {split!r} produced no batches")
        # Batches arrive in read order; a materialized split is in manifest order.
        order = np.argsort(
            np.concatenate([batch.order_indices for batch in batches]), kind="stable"
        )

        def rows(values) -> tuple:
            flat = [item for batch in batches for item in values(batch)]
            return tuple(flat[index] for index in order)

        def stacked(values) -> np.ndarray:
            return np.concatenate([values(batch) for batch in batches])[order]

        labels = None if batches[0].labels is None else stacked(lambda batch: batch.labels)
        if batches[0].design_mask.ndim == 2 and any(
            not np.array_equal(batch.design_mask, batches[0].design_mask) for batch in batches[1:]
        ):
            raise MLPartitionDataError(
                "split batches disagree on a position-by-channel design mask"
            )
        return MLMaterializedPartitionData(
            split=split,
            molecule_uids=rows(lambda batch: batch.molecule_uids),
            read_ids=rows(lambda batch: batch.read_ids),
            experiment_uids=rows(lambda batch: batch.experiment_uids),
            modalities=rows(lambda batch: batch.modalities),
            coordinates=self.plan.coordinates,
            channel_names=tuple(
                channel.name for channel in self.plan.dataset.input_schema.channels
            ),
            values=stacked(lambda batch: batch.values),
            labels=labels,
            observed_mask=stacked(lambda batch: batch.observed_mask),
            availability_mask=stacked(lambda batch: batch.availability_mask),
            design_mask=(
                stacked(lambda batch: batch.design_mask)
                if batches[0].design_mask.ndim == 3
                else batches[0].design_mask
            ),
            padding_mask=stacked(lambda batch: batch.padding_mask),
        )

    def _read_batch(self, entries: Sequence[PartitionReadEntry]) -> MLPartitionBatch:
        schema = self.plan.dataset.input_schema
        n_rows = len(entries)
        n_positions = schema.n_positions
        n_channels = len(schema.channels)
        values = np.full((n_rows, n_positions, n_channels), np.nan, dtype=np.float32)
        observed = np.zeros_like(values, dtype=bool)
        availability = np.zeros((n_rows, n_channels), dtype=bool)
        design = np.zeros_like(values, dtype=bool)
        padding = np.ones((n_rows, n_positions), dtype=bool)

        rows_by_experiment: dict[str, list[int]] = defaultdict(list)
        for row, entry in enumerate(entries):
            rows_by_experiment[entry.experiment_uid].append(row)
        for experiment_uid, batch_rows in rows_by_experiment.items():
            binding = self.plan.sources[experiment_uid]
            stages: dict[str, list[tuple[int, Any]]] = defaultdict(list)
            for channel_index, channel in enumerate(schema.channels):
                source = _channel_source(channel, binding.modality)
                if source is None:
                    continue
                availability[batch_rows, channel_index] = True
                stages[source.stage].append((channel_index, source))
            for stage, stage_channels in stages.items():
                self._read_stage(
                    entries,
                    batch_rows,
                    binding.stage_spines[stage],
                    stage_channels,
                    values,
                    observed,
                    design,
                    padding,
                )

        labels: np.ndarray | None
        if self.plan.dataset.label_schema is None:
            labels = None
        else:
            if any(entry.class_id is None for entry in entries):
                raise MLPartitionDataError("supervised batch contains a missing class ID")
            labels = np.asarray([entry.class_id for entry in entries], dtype=np.int64)
        design_output = design
        design_spec = next(mask for mask in schema.masks if mask.kind == "design")
        if design_spec.axes == ("position", "channel"):
            # Rows that cover none of the window were never projected and carry
            # no design; the others must agree (`F68`).
            covered = design[~padding.all(axis=1)]
            if len(covered) and not np.all(covered == covered[0]):
                raise MLPartitionDataError(
                    "batch has observation-specific design masks but the input schema "
                    "declares position-by-channel design; use an observation-axis design mask"
                )
            design_output = covered[0] if len(covered) else design[0]
        batch = MLPartitionBatch(
            order_indices=np.asarray([entry.order_index for entry in entries], dtype=np.int64),
            molecule_uids=tuple(entry.molecule_uid for entry in entries),
            read_ids=tuple(entry.read_id for entry in entries),
            experiment_uids=tuple(entry.experiment_uid for entry in entries),
            modalities=tuple(entry.modality for entry in entries),
            coordinates=self.plan.coordinates,
            channel_names=tuple(channel.name for channel in schema.channels),
            values=values,
            labels=labels,
            observed_mask=observed,
            availability_mask=availability,
            design_mask=design_output,
            padding_mask=padding,
        )
        mask_arrays = batch.mask_arrays(self.plan.dataset)
        validate_mask_arrays(schema, mask_arrays, batch_size=n_rows, require_all=False)
        validate_mask_relationships(schema, mask_arrays)
        return batch

    def _read_stage(
        self,
        entries: Sequence[PartitionReadEntry],
        batch_rows: Sequence[int],
        spine: Path,
        stage_channels: Sequence[tuple[int, Any]],
        values: np.ndarray,
        observed: np.ndarray,
        design: np.ndarray,
        padding: np.ndarray,
    ) -> None:
        if not self.plan.coordinate_maps:
            self._read_stage_span(
                entries,
                batch_rows,
                spine,
                stage_channels,
                (self.plan.coordinate_start, self.plan.coordinate_end),
                values,
                observed,
                design,
                padding,
            )
            return
        # A position window projects one reference at a time, and each mapped
        # reference has its own source span (`MLX-03`).
        by_reference: dict[str, list[int]] = defaultdict(list)
        for row in batch_rows:
            by_reference[entries[row].reference].append(row)
        for reference, rows in sorted(by_reference.items()):
            span = self.plan.read_spans.get(
                reference, (self.plan.coordinate_start, self.plan.coordinate_end)
            )
            self._read_stage_span(
                entries, rows, spine, stage_channels, span, values, observed, design, padding
            )

    def _read_stage_span(
        self,
        entries: Sequence[PartitionReadEntry],
        batch_rows: Sequence[int],
        spine: Path,
        stage_channels: Sequence[tuple[int, Any]],
        span: tuple[int, int],
        values: np.ndarray,
        observed: np.ndarray,
        design: np.ndarray,
        padding: np.ndarray,
    ) -> None:
        read_ids = [entries[row].read_id for row in batch_rows]
        layers = sorted({source.layer for _, source in stage_channels if source.layer != "X"})
        try:
            projected = materialize(
                spine,
                read_ids=read_ids,
                layers=layers,
                start=span[0],
                end=span[1],
                lazy=self.plan.policy.lazy,
                query_memory_mb=self.plan.policy.query_memory_mb,
            )
        except ValueError as exc:
            if "selection matched no molecules" in str(exc):
                return
            raise MLPartitionDataError(f"partition projection failed for {spine}: {exc}") from exc

        expected_rows = {entries[row].read_id: row for row in batch_rows}
        actual_ids = tuple(map(str, projected.obs_names))
        unknown = sorted(set(actual_ids).difference(expected_rows))
        if unknown:
            raise MLPartitionDataError(f"partition projection returned unknown reads: {unknown}")
        var_names = tuple(projected.var_names)
        column_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def columns_for(row: int) -> tuple[np.ndarray, np.ndarray]:
            """Source/target columns for one row's reference, through its map if any."""
            reference = entries[row].reference
            if reference not in column_cache:
                frame_of = self.plan.coordinate_maps.get(reference)
                if frame_of is None:
                    column_cache[reference] = _position_columns(var_names, self.plan.coordinates)
                else:
                    column_cache[reference] = _mapped_position_columns(
                        var_names, self.plan.coordinates, frame_of
                    )
            return column_cache[reference]

        physical_references = (
            projected.obs["Reference_strand"].astype(str)
            if "Reference_strand" in projected.obs
            else None
        )
        for local_row, read_id in enumerate(actual_ids):
            target_row = expected_rows[read_id]
            source_columns, target_columns = columns_for(target_row)
            if not len(source_columns):
                continue
            valid_columns = np.ones(len(source_columns), dtype=bool)
            reference = (
                str(physical_references.iloc[local_row])
                if physical_references is not None
                else entries[target_row].reference
            )
            position_column = f"position_in_{reference}"
            if position_column in projected.var:
                valid_columns &= np.asarray(projected.var[position_column])[source_columns].astype(
                    bool
                )
            padding[target_row, target_columns[valid_columns]] = False

        for channel_index, source in stage_channels:
            if source.layer == "X":
                matrix = projected.X
            elif source.layer in projected.layers:
                matrix = projected.layers[source.layer]
            else:
                raise MLPartitionDataError(
                    f"layer {source.layer!r} is absent from projected stage spine {spine}"
                )
            dense = _dense(matrix)
            for local_row, read_id in enumerate(actual_ids):
                target_row = expected_rows[read_id]
                source_columns, target_columns = columns_for(target_row)
                if not len(source_columns):
                    continue
                reference = (
                    str(physical_references.iloc[local_row])
                    if physical_references is not None
                    else entries[target_row].reference
                )
                source_values = np.asarray(dense[local_row, source_columns], dtype=np.float32)
                values[target_row, target_columns, channel_index] = source_values
                design_columns = _design_columns(projected.var, reference, source.site_context)
                if design_columns is None:
                    # Conservative compatibility for older partitions that did not
                    # persist reference-context columns: observed cells are known
                    # to be designed, while entirely missing positions stay false.
                    designed = np.isfinite(source_values)
                else:
                    designed = design_columns[source_columns]
                # Design is a property of the reference position, not of the
                # read: a read that ends early is padded there, not "not a site"
                # (`F68`). Observation still needs coverage.
                not_padding = ~padding[target_row, target_columns]
                designed = np.asarray(designed, dtype=bool)
                design[target_row, target_columns, channel_index] = designed
                observed[target_row, target_columns, channel_index] = (
                    np.isfinite(source_values) & designed & not_padding
                )
