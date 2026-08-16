"""Freeze an accepted re-basecall selection without executing basecalling."""

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

from ..informatics.pod5_identity import Pod5DatasetIndex
from ..informatics.raw_generation import RawGenerationError, validate_raw_generation
from ..readwrite import atomic_write_json
from .rebasecall_plan import RebasecallPlan, build_rebasecall_plan
from .rebasecall_request import RebasecallRequest

REBASECALL_SELECTION_SCHEMA_VERSION = 1
SELECTION_MANIFEST_FILENAME = "selection_manifest.json"
SELECTION_ROWS_FILENAME = "selection_rows.parquet"
SELECTION_ROW_COLUMNS = (
    "selection_ordinal",
    "observation_id",
    "molecule_uid",
    "pod5_read_id",
    "pod5_source_id",
    "identity_evidence",
)


class RebasecallSelectionError(RuntimeError):
    """Raised when an accepted selection cannot be frozen safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class FrozenRebasecallSelection:
    """One validated immutable selection-result artifact."""

    selection_id: str
    directory: Path
    manifest_path: Path
    rows_path: Path
    manifest: Mapping[str, Any]


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_records(plan: RebasecallPlan) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if plan.request.selection.mode == "all-signal":
        index = plan._pod5_index
        if index is None:
            raise RebasecallSelectionError(
                "selection_rows_unavailable",
                "the accepted plan retained no authoritative POD5 index",
            )
        for pod5_read_id, source_ids in sorted(index.sources_by_read_id.items()):
            if len(source_ids) != 1:
                raise RebasecallSelectionError(
                    "selection_identity_ambiguous",
                    f"POD5 UUID {pod5_read_id!r} does not identify one source occurrence",
                )
            records.append(
                {
                    "observation_id": None,
                    "molecule_uid": None,
                    "pod5_read_id": str(pod5_read_id),
                    "pod5_source_id": source_ids[0],
                    "identity_evidence": "pod5_dataset_index",
                }
            )
    else:
        resolution = plan._identity_resolution
        if resolution is None:
            raise RebasecallSelectionError(
                "selection_rows_unavailable",
                "the accepted plan retained no selected-molecule identity rows",
            )
        for row in resolution.rows:
            if row.status != "resolved" or row.pod5_read_id is None or len(row.source_ids) != 1:
                raise RebasecallSelectionError(
                    "selection_identity_unresolved",
                    f"selected observation {row.observation_id!r} has no unique POD5 identity",
                )
            records.append(
                {
                    "observation_id": row.observation_id,
                    "molecule_uid": row.molecule_uid,
                    "pod5_read_id": row.pod5_read_id,
                    "pod5_source_id": row.source_ids[0],
                    "identity_evidence": row.evidence,
                }
            )

    records.sort(
        key=lambda row: (
            str(row["pod5_read_id"]),
            "" if row["observation_id"] is None else str(row["observation_id"]),
        )
    )
    return [{"selection_ordinal": ordinal, **record} for ordinal, record in enumerate(records)]


def _records_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=SELECTION_ROW_COLUMNS)
    frame["selection_ordinal"] = frame["selection_ordinal"].astype("int64")
    for column in SELECTION_ROW_COLUMNS[1:]:
        frame[column] = frame[column].astype("string")
    return frame


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    if tuple(frame.columns) != SELECTION_ROW_COLUMNS:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection rows do not match the required column schema",
        )
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False, name=None):
        record: dict[str, object] = {"selection_ordinal": int(row[0])}
        for column, value in zip(SELECTION_ROW_COLUMNS[1:], row[1:], strict=True):
            record[column] = None if pd.isna(value) else str(value)
        records.append(record)
    if [record["selection_ordinal"] for record in records] != list(range(len(records))):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection ordinals are not contiguous and deterministic",
        )
    required = ("pod5_read_id", "pod5_source_id", "identity_evidence")
    if any(record[column] is None for record in records for column in required):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection contains incomplete source-signal identity rows",
        )
    observation_ids = [
        str(record["observation_id"]) for record in records if record["observation_id"] is not None
    ]
    if len(observation_ids) != len(set(observation_ids)):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection contains duplicate observation identities",
        )
    source_by_pod5: dict[str, str] = {}
    for record in records:
        pod5_read_id = str(record["pod5_read_id"])
        source_id = str(record["pod5_source_id"])
        previous = source_by_pod5.setdefault(pod5_read_id, source_id)
        if previous != source_id:
            raise RebasecallSelectionError(
                "selection_artifact_invalid",
                "one frozen POD5 UUID maps to multiple source identities",
            )
    return records


def _parent_identity(plan: RebasecallPlan) -> dict[str, object]:
    assert plan.raw_parent is not None
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


def _selection_identity(
    plan: RebasecallPlan,
    *,
    rows_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": REBASECALL_SELECTION_SCHEMA_VERSION,
        "experiment_uid": plan.experiment_uid,
        "parents": _parent_identity(plan),
        "source_manifest_digest": plan.sources.manifest_digest,
        "selection": plan.request.selection.to_dict(),
        "source_column_fingerprints": dict(plan.selection.source_column_fingerprints),
        "rows_digest": rows_digest,
    }


def _manifest_payload(
    plan: RebasecallPlan,
    records: list[dict[str, object]],
    *,
    rows_sha256: str,
) -> dict[str, object]:
    rows_digest = _sha256_payload(records)
    identity = _selection_identity(plan, rows_digest=rows_digest)
    molecule_count = sum(record["observation_id"] is not None for record in records)
    unique_pod5_count = len({str(record["pod5_read_id"]) for record in records})
    return {
        "schema_version": REBASECALL_SELECTION_SCHEMA_VERSION,
        "selection_id": _sha256_payload(identity),
        "accepted_plan_id": plan.plan_id,
        "request_id": plan.request.request_id,
        "experiment_id": plan.experiment_id,
        "identity": identity,
        "counts": {
            "record_count": len(records),
            "molecule_count": molecule_count,
            "unique_pod5_read_count": unique_pod5_count,
            "duplicate_parent_reference_count": max(0, molecule_count - unique_pod5_count),
        },
        "rows": {
            "path": SELECTION_ROWS_FILENAME,
            "sha256": rows_sha256,
            "semantic_digest": rows_digest,
        },
    }


def _validated_parent_manifest_digests(plan: RebasecallPlan) -> dict[str, str]:
    if plan.raw_parent is None:
        raise RebasecallSelectionError(
            "selection_parent_unavailable",
            "the accepted plan has no raw parent",
        )
    try:
        raw_manifest = validate_raw_generation(
            plan.raw_parent.generation_dir,
            expected_generation_id=plan.raw_parent.generation_id,
            run_root=plan.run_root,
        )
    except (RawGenerationError, OSError, ValueError) as exc:
        raise RebasecallSelectionError("selection_parent_changed", str(exc)) from exc
    digests = {"raw": _sha256_payload(dict(raw_manifest))}

    if plan.preprocess_parent is not None:
        from ..preprocessing.preprocess_generation import (
            PreprocessGenerationError,
            validate_preprocess_generation,
        )

        try:
            preprocess_manifest = validate_preprocess_generation(
                plan.preprocess_parent.generation_dir,
                expected_generation_id=plan.preprocess_parent.generation_id,
                run_root=plan.run_root,
            )
        except (PreprocessGenerationError, OSError, ValueError) as exc:
            raise RebasecallSelectionError("selection_parent_changed", str(exc)) from exc
        digests["preprocess"] = _sha256_payload(dict(preprocess_manifest))
    return digests


def _validate_parent_state(plan: RebasecallPlan) -> None:
    observed = _validated_parent_manifest_digests(plan)
    expected = {"raw": plan.raw_parent.manifest_digest} if plan.raw_parent else {}
    if plan.preprocess_parent is not None:
        expected["preprocess"] = plan.preprocess_parent.manifest_digest
    if observed != expected:
        raise RebasecallSelectionError(
            "selection_parent_changed",
            "parent generation identity changed after the accepted plan was produced",
        )


def read_frozen_rebasecall_selection(
    directory: str | Path,
    *,
    expected_selection_id: str | None = None,
) -> FrozenRebasecallSelection:
    """Read and fully validate one frozen selection result."""
    directory = Path(directory)
    manifest_path = directory / SELECTION_MANIFEST_FILENAME
    rows_path = directory / SELECTION_ROWS_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection manifest is missing or invalid",
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection manifest has an unsupported schema",
        )
    required_manifest_keys = {
        "schema_version",
        "selection_id",
        "accepted_plan_id",
        "request_id",
        "experiment_id",
        "identity",
        "counts",
        "rows",
    }
    if set(manifest) != required_manifest_keys:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection manifest fields do not match schema 1",
        )
    selection_id = str(manifest.get("selection_id", ""))
    if (
        len(selection_id) != 64
        or any(character not in "0123456789abcdef" for character in selection_id)
        or (expected_selection_id and selection_id != expected_selection_id)
    ):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection identity does not match the expected selection",
        )
    identity = manifest.get("identity")
    required_identity_keys = {
        "schema_version",
        "experiment_uid",
        "parents",
        "source_manifest_digest",
        "selection",
        "source_column_fingerprints",
        "rows_digest",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != required_identity_keys
        or _sha256_payload(identity) != selection_id
    ):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection semantic identity does not match its selection ID",
        )
    rows_record = manifest.get("rows")
    if not isinstance(rows_record, dict) or set(rows_record) != {
        "path",
        "sha256",
        "semantic_digest",
    }:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection row record does not match schema 1",
        )
    try:
        rows_sha256 = _sha256_file(rows_path)
    except OSError as exc:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection row artifact is missing or unreadable",
        ) from exc
    if rows_sha256 != rows_record.get("sha256"):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection row artifact checksum does not match its manifest",
        )
    try:
        frame = pd.read_parquet(rows_path)
    except (OSError, ValueError, ImportError) as exc:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection rows are unreadable",
        ) from exc
    try:
        records = _records_from_frame(frame)
    except RebasecallSelectionError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection rows contain invalid scalar values",
        ) from exc
    rows_digest = _sha256_payload(records)
    if (
        rows_record.get("path") != SELECTION_ROWS_FILENAME
        or rows_digest != rows_record.get("semantic_digest")
        or identity.get("rows_digest") != rows_digest
    ):
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection row identities do not match their semantic digest",
        )
    counts = manifest.get("counts")
    molecule_count = sum(record["observation_id"] is not None for record in records)
    unique_pod5_count = len({str(record["pod5_read_id"]) for record in records})
    expected_counts = {
        "record_count": len(records),
        "molecule_count": molecule_count,
        "unique_pod5_read_count": unique_pod5_count,
        "duplicate_parent_reference_count": max(0, molecule_count - unique_pod5_count),
    }
    if counts != expected_counts:
        raise RebasecallSelectionError(
            "selection_artifact_invalid",
            "frozen selection counts do not match its rows",
        )
    return FrozenRebasecallSelection(
        selection_id=selection_id,
        directory=directory,
        manifest_path=manifest_path,
        rows_path=rows_path,
        manifest=manifest,
    )


def freeze_rebasecall_selection(
    plan: RebasecallPlan,
    selection_root: str | Path,
    *,
    accepted_plan_id: str,
    parent_validator: Callable[[RebasecallPlan], None] = _validate_parent_state,
) -> FrozenRebasecallSelection:
    """Atomically freeze the complete selection from an explicitly accepted plan.

    This operation writes only the immutable selection result. It does not run
    Dorado, create a raw generation, publish a lineage, or update a current
    selector.
    """
    if accepted_plan_id != plan.plan_id:
        raise RebasecallSelectionError(
            "accepted_plan_mismatch",
            "the supplied accepted plan ID does not match the current plan",
        )
    if plan.status != "ready":
        raise RebasecallSelectionError(
            "accepted_plan_blocked",
            "a blocked re-basecall plan cannot be frozen",
        )
    parent_validator(plan)
    records = _selection_records(plan)
    expected_count = plan.selection.selected_count
    if expected_count is None or len(records) != expected_count:
        raise RebasecallSelectionError(
            "selection_count_mismatch",
            "resolved selection rows do not match the accepted selection count",
        )

    rows_digest = _sha256_payload(records)
    identity = _selection_identity(plan, rows_digest=rows_digest)
    selection_id = _sha256_payload(identity)
    selection_root = Path(selection_root)
    destination = selection_root / selection_id
    if destination.exists():
        return read_frozen_rebasecall_selection(
            destination,
            expected_selection_id=selection_id,
        )

    selection_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{selection_id}.",
            suffix=".tmp",
            dir=selection_root,
        )
    )
    try:
        rows_path = temporary / SELECTION_ROWS_FILENAME
        _records_frame(records).to_parquet(rows_path, index=False)
        manifest = _manifest_payload(
            plan,
            records,
            rows_sha256=_sha256_file(rows_path),
        )
        atomic_write_json(temporary / SELECTION_MANIFEST_FILENAME, manifest)
        read_frozen_rebasecall_selection(
            temporary,
            expected_selection_id=selection_id,
        )
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
            read_frozen_rebasecall_selection(
                destination,
                expected_selection_id=selection_id,
            )
            shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_frozen_rebasecall_selection(
        destination,
        expected_selection_id=selection_id,
    )


def prepare_rebasecall_selection(
    cfg: Any,
    request: RebasecallRequest,
    selection_root: str | Path,
    *,
    accepted_plan_id: str,
    pod5_indexer: Callable[[tuple[tuple[str, Path], ...]], Pod5DatasetIndex] | None = None,
    bam_tag_reader: Callable[[Path], Mapping[str, Mapping[str, object]]] | None = None,
    parent_validator: Callable[[RebasecallPlan], None] = _validate_parent_state,
) -> FrozenRebasecallSelection:
    """Rebuild an accepted plan and freeze its selection for a future run."""
    planner_kwargs: dict[str, Any] = {}
    if pod5_indexer is not None:
        planner_kwargs["pod5_indexer"] = pod5_indexer
    if bam_tag_reader is not None:
        planner_kwargs["bam_tag_reader"] = bam_tag_reader
    plan = build_rebasecall_plan(cfg, request, **planner_kwargs)
    return freeze_rebasecall_selection(
        plan,
        selection_root,
        accepted_plan_id=accepted_plan_id,
        parent_validator=parent_validator,
    )
