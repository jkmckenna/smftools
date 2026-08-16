"""Materialize and validate immutable selected-signal POD5 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ..informatics.input_manifest import InputManifestError, checksum_input_source
from ..informatics.pod5_identity import Pod5DatasetIndex, build_pod5_dataset_index
from ..informatics.pod5_source import Pod5SourceResolutionRow
from ..readwrite import atomic_write_json
from .rebasecall_plan import RebasecallPlan, build_rebasecall_plan
from .rebasecall_request import RebasecallRequest
from .rebasecall_selection import (
    FrozenRebasecallSelection,
    freeze_rebasecall_selection,
    read_frozen_rebasecall_selection,
)

REBASECALL_SIGNAL_SCHEMA_VERSION = 1
SIGNAL_MANIFEST_FILENAME = "signal_manifest.json"
SIGNAL_SOURCES_DIRNAME = "sources"


class RebasecallSignalError(RuntimeError):
    """Raised when selected signal cannot be materialized or validated safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class MaterializedRebasecallSignal:
    """One validated immutable filtered-signal artifact."""

    signal_id: str
    directory: Path
    manifest_path: Path
    source_paths: tuple[Path, ...]
    manifest: Mapping[str, Any]


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_ids_digest(read_ids: set[str] | tuple[str, ...]) -> str:
    return _sha256_payload(sorted(map(str, read_ids)))


def _is_sha256(value: object) -> bool:
    normalized = str(value)
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _manifest_count(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            f"materialized signal {field_name} is invalid",
        )
    return value


def _write_filtered_pod5(
    source_path: Path,
    output_path: Path,
    read_ids: tuple[str, ...],
) -> None:
    """Write exact selected records from one source POD5."""
    from ..optional_imports import require

    pod5 = require("pod5", extra="ont", purpose="selected POD5 materialization")
    with (
        pod5.Reader(source_path) as reader,
        pod5.Writer(
            output_path,
            software_name="smftools selective re-basecalling",
        ) as writer,
    ):
        for read in reader.reads(selection=read_ids, missing_ok=True):
            writer.add_reads([read.to_read()])


def _expected_parent_identity(plan: RebasecallPlan) -> dict[str, object]:
    if plan.raw_parent is None:
        raise RebasecallSignalError(
            "signal_parent_unavailable",
            "the accepted plan has no raw parent",
        )
    return {
        "raw": {
            "generation_id": plan.raw_parent.generation_id,
            "manifest_digest": plan.raw_parent.manifest_digest,
        },
        "preprocess": (
            None
            if plan.preprocess_parent is None
            else {
                "generation_id": plan.preprocess_parent.generation_id,
                "manifest_digest": plan.preprocess_parent.manifest_digest,
            }
        ),
    }


def _validate_selection_for_plan(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
) -> FrozenRebasecallSelection:
    validated = read_frozen_rebasecall_selection(
        selection.directory,
        expected_selection_id=selection.selection_id,
    )
    identity = validated.manifest["identity"]
    expected = {
        "experiment_uid": plan.experiment_uid,
        "parents": _expected_parent_identity(plan),
        "source_manifest_digest": plan.sources.manifest_digest,
        "selection": plan.request.selection.to_dict(),
        "source_column_fingerprints": dict(plan.selection.source_column_fingerprints),
    }
    if any(identity.get(field_name) != value for field_name, value in expected.items()):
        raise RebasecallSignalError(
            "signal_selection_mismatch",
            "the frozen selection does not belong to the accepted source and parent plan",
        )
    return validated


def _validated_source_rows(plan: RebasecallPlan) -> dict[str, Pod5SourceResolutionRow]:
    resolution = plan._source_resolution
    if resolution is None or not resolution.complete:
        raise RebasecallSignalError(
            "signal_sources_unavailable",
            "the accepted plan retained no complete source-signal resolution",
        )
    rows = {row.source_id: row for row in resolution.rows}
    for row in rows.values():
        if row.resolved_path is None:
            raise RebasecallSignalError(
                "signal_sources_unavailable",
                f"source {row.source_id!r} has no resolved POD5 path",
            )
        try:
            sha256, size_bytes = checksum_input_source(row.resolved_path)
        except (InputManifestError, OSError, ValueError) as exc:
            raise RebasecallSignalError(
                "signal_source_changed",
                f"source {row.source_id!r} is missing or unreadable",
            ) from exc
        if sha256 != row.sha256 or size_bytes != row.size_bytes:
            raise RebasecallSignalError(
                "signal_source_changed",
                f"source {row.source_id!r} changed after the accepted plan was produced",
            )
    return rows


def _selection_requests(
    selection: FrozenRebasecallSelection,
) -> tuple[dict[str, set[str]], int]:
    frame = pd.read_parquet(
        selection.rows_path,
        columns=["pod5_read_id", "pod5_source_id"],
    )
    requested: dict[str, set[str]] = {}
    for pod5_read_id, source_id in frame.itertuples(index=False, name=None):
        if pd.isna(pod5_read_id) or pd.isna(source_id):
            raise RebasecallSignalError(
                "signal_selection_invalid",
                "the frozen selection contains incomplete POD5 source identities",
            )
        requested.setdefault(str(source_id), set()).add(str(pod5_read_id))
    if not requested:
        raise RebasecallSignalError(
            "signal_selection_invalid",
            "the frozen selection contains no POD5 reads",
        )
    return requested, len(frame)


def _signal_identity(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    requested: Mapping[str, set[str]],
    source_rows: Mapping[str, Pod5SourceResolutionRow],
) -> dict[str, object]:
    resolution = plan._source_resolution
    assert resolution is not None
    sources = []
    for source_id, read_ids in sorted(requested.items()):
        row = source_rows.get(source_id)
        if row is None:
            raise RebasecallSignalError(
                "signal_selection_source_missing",
                f"selection source {source_id!r} is absent from the accepted source resolution",
            )
        sources.append(
            {
                "source_id": source_id,
                "source_sha256": row.sha256,
                "requested_read_count": len(read_ids),
                "requested_read_ids_digest": _read_ids_digest(read_ids),
            }
        )
    return {
        "schema_version": REBASECALL_SIGNAL_SCHEMA_VERSION,
        "selection_id": selection.selection_id,
        "source_manifest_digest": plan.sources.manifest_digest,
        "source_resolution_digest": resolution.digest,
        "sources": sources,
    }


def _indexed_ids_by_source(index: Pod5DatasetIndex) -> tuple[dict[str, set[str]], int]:
    by_source: dict[str, set[str]] = {}
    duplicate_count = 0
    for read_id, source_ids in index.sources_by_read_id.items():
        if len(source_ids) != 1:
            duplicate_count += 1
            continue
        by_source.setdefault(source_ids[0], set()).add(str(read_id))
    return by_source, duplicate_count


def _inspect_outputs(
    directory: Path,
    identity: Mapping[str, object],
    *,
    pod5_indexer: Callable[[tuple[tuple[str, Path], ...]], Pod5DatasetIndex],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    source_records = identity["sources"]
    output_pairs: list[tuple[str, Path]] = []
    expected_digests: dict[str, str] = {}
    expected_counts: dict[str, int] = {}
    for ordinal, record in enumerate(source_records):
        source_id = str(record["source_id"])
        path = directory / SIGNAL_SOURCES_DIRNAME / f"{ordinal:06d}.pod5"
        output_pairs.append((source_id, path))
        expected_digests[source_id] = str(record["requested_read_ids_digest"])
        expected_counts[source_id] = int(record["requested_read_count"])
    try:
        index = pod5_indexer(tuple(output_pairs))
    except Exception as exc:
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            f"materialized POD5 outputs could not be indexed: {type(exc).__name__}: {exc}",
        ) from exc
    actual_by_source, duplicate_count = _indexed_ids_by_source(index)
    if duplicate_count:
        raise RebasecallSignalError(
            "signal_uuid_duplicate",
            f"materialized outputs contain {duplicate_count} duplicate POD5 UUID(s)",
        )
    outputs: list[dict[str, object]] = []
    missing_total = 0
    found_total = 0
    for ordinal, (source_id, path) in enumerate(output_pairs):
        actual = actual_by_source.get(source_id, set())
        actual_digest = _read_ids_digest(actual)
        expected_count = expected_counts[source_id]
        expected_digest = expected_digests[source_id]
        found_count = len(actual)
        missing_count = max(0, expected_count - found_count)
        found_total += found_count
        missing_total += missing_count
        try:
            sha256, size_bytes = checksum_input_source(path)
        except (InputManifestError, OSError, ValueError) as exc:
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                f"materialized POD5 output {ordinal} is missing or unreadable",
            ) from exc
        outputs.append(
            {
                "source_id": source_id,
                "path": f"{SIGNAL_SOURCES_DIRNAME}/{ordinal:06d}.pod5",
                "sha256": sha256,
                "size_bytes": size_bytes,
                "requested_read_count": expected_count,
                "found_read_count": found_count,
                "missing_read_count": missing_count,
                "duplicate_read_id_count": 0,
                "read_ids_digest": actual_digest,
            }
        )
        if found_count != expected_count or actual_digest != expected_digest:
            raise RebasecallSignalError(
                "signal_uuid_count_mismatch",
                f"materialized source {source_id!r} does not contain its exact requested UUID set",
            )
    return outputs, {
        "requested_unique_read_count": sum(expected_counts.values()),
        "found_unique_read_count": found_total,
        "missing_read_count": missing_total,
        "duplicate_output_read_id_count": duplicate_count,
    }


def _manifest_payload(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    identity: Mapping[str, object],
    outputs: list[dict[str, object]],
    counts: Mapping[str, int],
    *,
    selection_record_count: int,
) -> dict[str, object]:
    unique_count = int(counts["requested_unique_read_count"])
    return {
        "schema_version": REBASECALL_SIGNAL_SCHEMA_VERSION,
        "signal_id": _sha256_payload(identity),
        "accepted_plan_id": plan.plan_id,
        "request_id": plan.request.request_id,
        "experiment_id": plan.experiment_id,
        "selection_id": selection.selection_id,
        "identity": dict(identity),
        "counts": {
            "selection_record_count": selection_record_count,
            **dict(counts),
            "duplicate_selection_reference_count": selection_record_count - unique_count,
        },
        "outputs": outputs,
    }


def read_materialized_rebasecall_signal(
    directory: str | Path,
    *,
    expected_signal_id: str | None = None,
    pod5_indexer: Callable[
        [tuple[tuple[str, Path], ...]], Pod5DatasetIndex
    ] = build_pod5_dataset_index,
) -> MaterializedRebasecallSignal:
    """Read and fully validate one filtered-signal artifact."""
    directory = Path(directory)
    manifest_path = directory / SIGNAL_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal manifest is missing or invalid",
        ) from exc
    required_manifest_keys = {
        "schema_version",
        "signal_id",
        "accepted_plan_id",
        "request_id",
        "experiment_id",
        "selection_id",
        "identity",
        "counts",
        "outputs",
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != REBASECALL_SIGNAL_SCHEMA_VERSION
        or set(manifest) != required_manifest_keys
    ):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal manifest does not match schema 1",
        )
    signal_id = str(manifest["signal_id"])
    if not _is_sha256(signal_id) or (
        expected_signal_id is not None and signal_id != expected_signal_id
    ):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal identity does not match the expected artifact",
        )
    identity = manifest["identity"]
    required_identity_keys = {
        "schema_version",
        "selection_id",
        "source_manifest_digest",
        "source_resolution_digest",
        "sources",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != required_identity_keys
        or identity.get("schema_version") != REBASECALL_SIGNAL_SCHEMA_VERSION
        or _sha256_payload(identity) != signal_id
        or manifest.get("selection_id") != identity.get("selection_id")
    ):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal semantic identity is inconsistent",
        )
    source_records = identity.get("sources")
    outputs = manifest.get("outputs")
    if (
        not isinstance(source_records, list)
        or not source_records
        or not isinstance(outputs, list)
        or len(outputs) != len(source_records)
    ):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal source records are invalid",
        )
    expected_source_keys = {
        "source_id",
        "source_sha256",
        "requested_read_count",
        "requested_read_ids_digest",
    }
    expected_output_keys = {
        "source_id",
        "path",
        "sha256",
        "size_bytes",
        "requested_read_count",
        "found_read_count",
        "missing_read_count",
        "duplicate_read_id_count",
        "read_ids_digest",
    }
    source_ids: list[str] = []
    output_pairs: list[tuple[str, Path]] = []
    for ordinal, (source, output) in enumerate(zip(source_records, outputs, strict=True)):
        if (
            not isinstance(source, dict)
            or set(source) != expected_source_keys
            or not isinstance(output, dict)
            or set(output) != expected_output_keys
        ):
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                "materialized signal source/output fields are invalid",
            )
        source_id = str(source["source_id"])
        expected_path = f"{SIGNAL_SOURCES_DIRNAME}/{ordinal:06d}.pod5"
        if (
            output.get("source_id") != source_id
            or output.get("path") != expected_path
            or not _is_sha256(source.get("source_sha256"))
            or not _is_sha256(source.get("requested_read_ids_digest"))
            or not _is_sha256(output.get("sha256"))
            or not _is_sha256(output.get("read_ids_digest"))
        ):
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                "materialized signal source/output identity is invalid",
            )
        requested_count = _manifest_count(
            source.get("requested_read_count"),
            "requested read count",
            minimum=1,
        )
        if output.get("requested_read_count") != requested_count:
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                "materialized signal requested counts are inconsistent",
            )
        _manifest_count(output.get("size_bytes"), "output size")
        _manifest_count(output.get("found_read_count"), "found read count")
        _manifest_count(output.get("missing_read_count"), "missing read count")
        _manifest_count(
            output.get("duplicate_read_id_count"),
            "duplicate read ID count",
        )
        path = directory / expected_path
        try:
            sha256, size_bytes = checksum_input_source(path)
        except (InputManifestError, OSError, ValueError) as exc:
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                "materialized signal output is missing or unreadable",
            ) from exc
        if sha256 != output["sha256"] or size_bytes != output["size_bytes"]:
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                "materialized signal output checksum does not match its manifest",
            )
        source_ids.append(source_id)
        output_pairs.append((source_id, path))
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal sources are not unique and deterministic",
        )
    try:
        index = pod5_indexer(tuple(output_pairs))
    except Exception as exc:
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            f"materialized signal outputs could not be indexed: {type(exc).__name__}: {exc}",
        ) from exc
    actual_by_source, duplicate_count = _indexed_ids_by_source(index)
    expected_counts = {
        "requested_unique_read_count": 0,
        "found_unique_read_count": 0,
        "missing_read_count": 0,
        "duplicate_output_read_id_count": duplicate_count,
    }
    for source, output in zip(source_records, outputs, strict=True):
        source_id = str(source["source_id"])
        actual = actual_by_source.get(source_id, set())
        actual_count = len(actual)
        actual_digest = _read_ids_digest(actual)
        requested_count = _manifest_count(
            source["requested_read_count"],
            "requested read count",
            minimum=1,
        )
        if (
            output["requested_read_count"] != requested_count
            or output["found_read_count"] != actual_count
            or output["missing_read_count"] != max(0, requested_count - actual_count)
            or output["duplicate_read_id_count"] != 0
            or output["read_ids_digest"] != actual_digest
            or source["requested_read_ids_digest"] != actual_digest
            or requested_count != actual_count
        ):
            raise RebasecallSignalError(
                "signal_artifact_invalid",
                "materialized signal UUID counts or identities do not match its outputs",
            )
        expected_counts["requested_unique_read_count"] += requested_count
        expected_counts["found_unique_read_count"] += actual_count
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal counts are invalid",
        )
    selection_record_count = _manifest_count(
        counts.get("selection_record_count"),
        "selection record count",
        minimum=1,
    )
    expected_complete_counts = {
        "selection_record_count": selection_record_count,
        **expected_counts,
        "duplicate_selection_reference_count": (
            selection_record_count - expected_counts["requested_unique_read_count"]
        ),
    }
    if (
        counts != expected_complete_counts
        or selection_record_count < expected_counts["requested_unique_read_count"]
    ):
        raise RebasecallSignalError(
            "signal_artifact_invalid",
            "materialized signal aggregate counts do not match its outputs",
        )
    return MaterializedRebasecallSignal(
        signal_id=signal_id,
        directory=directory,
        manifest_path=manifest_path,
        source_paths=tuple(path for _, path in output_pairs),
        manifest=manifest,
    )


def materialize_rebasecall_signal(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    signal_root: str | Path,
    *,
    accepted_plan_id: str,
    pod5_writer: Callable[[Path, Path, tuple[str, ...]], None] = _write_filtered_pod5,
    pod5_indexer: Callable[
        [tuple[tuple[str, Path], ...]], Pod5DatasetIndex
    ] = build_pod5_dataset_index,
) -> MaterializedRebasecallSignal:
    """Atomically materialize selected POD5 reads from an accepted plan."""
    if accepted_plan_id != plan.plan_id:
        raise RebasecallSignalError(
            "accepted_plan_mismatch",
            "the supplied accepted plan ID does not match the current plan",
        )
    if plan.status != "ready":
        raise RebasecallSignalError(
            "accepted_plan_blocked",
            "a blocked re-basecall plan cannot materialize signal",
        )
    if not plan.request.signal.materialize:
        raise RebasecallSignalError(
            "signal_materialization_not_requested",
            "the accepted request did not enable signal materialization",
        )
    validated_selection = _validate_selection_for_plan(plan, selection)
    source_rows = _validated_source_rows(plan)
    requested, selection_record_count = _selection_requests(validated_selection)
    identity = _signal_identity(plan, validated_selection, requested, source_rows)
    signal_id = _sha256_payload(identity)
    signal_root = Path(signal_root)
    destination = signal_root / signal_id
    if destination.exists():
        return read_materialized_rebasecall_signal(
            destination,
            expected_signal_id=signal_id,
            pod5_indexer=pod5_indexer,
        )

    signal_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{signal_id}.",
            suffix=".tmp",
            dir=signal_root,
        )
    )
    try:
        sources_dir = temporary / SIGNAL_SOURCES_DIRNAME
        sources_dir.mkdir()
        for ordinal, (source_id, read_ids) in enumerate(sorted(requested.items())):
            source_row = source_rows[source_id]
            assert source_row.resolved_path is not None
            pod5_writer(
                source_row.resolved_path,
                sources_dir / f"{ordinal:06d}.pod5",
                tuple(sorted(read_ids)),
            )
        _validated_source_rows(plan)
        outputs, counts = _inspect_outputs(
            temporary,
            identity,
            pod5_indexer=pod5_indexer,
        )
        manifest = _manifest_payload(
            plan,
            validated_selection,
            identity,
            outputs,
            counts,
            selection_record_count=selection_record_count,
        )
        atomic_write_json(temporary / SIGNAL_MANIFEST_FILENAME, manifest)
        read_materialized_rebasecall_signal(
            temporary,
            expected_signal_id=signal_id,
            pod5_indexer=pod5_indexer,
        )
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
            read_materialized_rebasecall_signal(
                destination,
                expected_signal_id=signal_id,
                pod5_indexer=pod5_indexer,
            )
            shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_materialized_rebasecall_signal(
        destination,
        expected_signal_id=signal_id,
        pod5_indexer=pod5_indexer,
    )


def prepare_rebasecall_signal(
    cfg: Any,
    request: RebasecallRequest,
    selection_root: str | Path,
    signal_root: str | Path,
    *,
    accepted_plan_id: str,
    pod5_writer: Callable[[Path, Path, tuple[str, ...]], None] = _write_filtered_pod5,
    pod5_indexer: Callable[
        [tuple[tuple[str, Path], ...]], Pod5DatasetIndex
    ] = build_pod5_dataset_index,
    bam_tag_reader: Callable[[Path], Mapping[str, Mapping[str, object]]] | None = None,
    parent_validator: Callable[[RebasecallPlan], None] | None = None,
) -> MaterializedRebasecallSignal:
    """Rebuild an accepted plan, freeze selection, and materialize signal."""
    planner_kwargs: dict[str, Any] = {"pod5_indexer": pod5_indexer}
    if bam_tag_reader is not None:
        planner_kwargs["bam_tag_reader"] = bam_tag_reader
    plan = build_rebasecall_plan(cfg, request, **planner_kwargs)
    freezer_kwargs: dict[str, Any] = {}
    if parent_validator is not None:
        freezer_kwargs["parent_validator"] = parent_validator
    selection = freeze_rebasecall_selection(
        plan,
        selection_root,
        accepted_plan_id=accepted_plan_id,
        **freezer_kwargs,
    )
    return materialize_rebasecall_signal(
        plan,
        selection,
        signal_root,
        accepted_plan_id=accepted_plan_id,
        pod5_writer=pod5_writer,
        pod5_indexer=pod5_indexer,
    )
