"""Execute selective Dorado basecalling and publish immutable basecall artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ..informatics.dorado_model import (
    DoradoBasecallOptions,
    DoradoBasecallResolution,
    DoradoModelError,
    build_dorado_basecaller_argv,
    resolve_dorado_basecall,
)
from ..informatics.input_manifest import InputManifestError, checksum_input_source
from ..informatics.pod5_identity import Pod5DatasetIndex, build_pod5_dataset_index
from ..readwrite import atomic_write_json
from .rebasecall_plan import RebasecallPlan, build_rebasecall_plan
from .rebasecall_request import SELECTION_GENERATION_KINDS, RebasecallRequest
from .rebasecall_selection import (
    FrozenRebasecallSelection,
    freeze_rebasecall_selection,
    read_frozen_rebasecall_selection,
)
from .rebasecall_signal import MaterializedRebasecallSignal, materialize_rebasecall_signal

REBASECALL_BASECALL_SCHEMA_VERSION = 1
BASECALL_MANIFEST_FILENAME = "basecall_manifest.json"
BASECALL_BAM_FILENAME = "calls.bam"
BASECALL_SUMMARY_FILENAME = "sequencing_summary.tsv"
BASECALL_READ_IDS_FILENAME = "read_ids.txt"
BASECALL_ORIGIN_FILENAME = "read_to_pod5_origin.csv"

_SUMMARY_CANDIDATES = ("sequencing_summary.txt", "sequencing_summary.tsv")


class RebasecallBasecallError(RuntimeError):
    """Raised when a selective basecall cannot be executed or validated safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class BasecallOutputInspection:
    """Structural observations read back from one Dorado output BAM."""

    record_count: int
    read_ids: tuple[str, ...]
    parent_ids: tuple[str | None, ...]
    programs: tuple[Mapping[str, str], ...]
    read_groups: tuple[Mapping[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.read_ids) != self.record_count or len(self.parent_ids) != self.record_count:
            raise ValueError("basecall inspection read/parent counts must match record_count")


@dataclass(frozen=True)
class PublishedRebasecallBasecall:
    """One validated immutable basecall artifact."""

    basecall_id: str
    directory: Path
    manifest_path: Path
    bam_path: Path
    manifest: Mapping[str, Any]

    @property
    def generation_kind(self) -> str:
        """Return the selection-derived kind stamped on this basecall."""
        return str(self.manifest["generation_kind"])


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_ids_digest(read_ids: Sequence[str] | set[str]) -> str:
    return _sha256_payload(sorted(map(str, read_ids)))


def _is_sha256(value: object) -> bool:
    normalized = str(value)
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _manifest_count(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            f"published basecall {field_name} is invalid",
        )
    return value


def _inspect_bam(path: Path) -> BasecallOutputInspection:
    """Read structural identity from a Dorado BAM without loading payloads."""
    from ..optional_imports import require

    pysam = require("pysam", extra="ont", purpose="basecall output validation")
    read_ids: list[str] = []
    parent_ids: list[str | None] = []
    try:
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
            header = bam.header.to_dict()
            for record in bam.fetch(until_eof=True):
                if record.is_secondary or record.is_supplementary:
                    continue
                read_ids.append(str(record.query_name))
                parent = record.get_tag("pi") if record.has_tag("pi") else None
                parent_ids.append(None if parent is None else str(parent))
    except Exception as exc:  # pysam raises bare exceptions for malformed BAMs
        raise RebasecallBasecallError(
            "basecall_output_unreadable",
            f"the Dorado output BAM could not be read: {type(exc).__name__}: {exc}",
        ) from exc
    programs = tuple(
        {str(key): str(value) for key, value in record.items()}
        for record in header.get("PG", ())
        if isinstance(record, Mapping)
    )
    read_groups = tuple(
        {str(key): str(value) for key, value in record.items()}
        for record in header.get("RG", ())
        if isinstance(record, Mapping)
    )
    return BasecallOutputInspection(
        record_count=len(read_ids),
        read_ids=tuple(read_ids),
        parent_ids=tuple(parent_ids),
        programs=programs,
        read_groups=read_groups,
    )


def _run_dorado(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _generation_kind(request: RebasecallRequest) -> str:
    mode = str(request.selection.mode)
    kind = SELECTION_GENERATION_KINDS.get(mode)
    if kind is None:
        raise RebasecallBasecallError(
            "basecall_selection_mode_unsupported",
            f"selection mode {mode!r} has no basecall generation kind",
        )
    return kind


def _validated_selection(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
) -> FrozenRebasecallSelection:
    validated = read_frozen_rebasecall_selection(
        selection.directory,
        expected_selection_id=selection.selection_id,
    )
    identity = validated.manifest["identity"]
    if (
        identity.get("experiment_uid") != plan.experiment_uid
        or identity.get("source_manifest_digest") != plan.sources.manifest_digest
    ):
        raise RebasecallBasecallError(
            "basecall_selection_mismatch",
            "the frozen selection does not belong to the accepted plan",
        )
    return validated


def _selection_origin(selection: FrozenRebasecallSelection) -> tuple[dict[str, str], int]:
    """Return the exact ``pod5_read_id -> pod5_source_id`` map for the selection."""
    frame = pd.read_parquet(
        selection.rows_path,
        columns=["pod5_read_id", "pod5_source_id"],
    )
    origin: dict[str, str] = {}
    for pod5_read_id, source_id in frame.itertuples(index=False, name=None):
        if pd.isna(pod5_read_id) or pd.isna(source_id):
            raise RebasecallBasecallError(
                "basecall_selection_invalid",
                "the frozen selection contains incomplete POD5 source identities",
            )
        origin[str(pod5_read_id)] = str(source_id)
    if not origin:
        raise RebasecallBasecallError(
            "basecall_selection_invalid",
            "the frozen selection contains no POD5 reads",
        )
    return origin, len(frame)


def _resolution_for_plan(
    plan: RebasecallPlan,
    *,
    dorado_resolver: Callable[..., DoradoBasecallResolution] | None,
    source_paths: Sequence[Path],
    model_directory: str | Path | None,
) -> DoradoBasecallResolution:
    """Return the exact model bundle the accepted plan resolved, re-resolving if needed."""
    if plan._model_resolution is not None:
        return plan._model_resolution
    resolver = resolve_dorado_basecall if dorado_resolver is None else dorado_resolver
    basecall = plan.request.basecall
    options = DoradoBasecallOptions(
        model=basecall.model,
        modified_bases=basecall.modified_bases,
        read_splitting=basecall.read_splitting,
        trim=basecall.trim,
        emit_moves=basecall.emit_moves,
        min_qscore=basecall.min_qscore,
        barcode_kit=basecall.barcode_kit,
        barcode_both_ends=basecall.barcode_both_ends,
    )
    try:
        return resolver(options, tuple(source_paths), model_directory)
    except DoradoModelError as exc:
        raise RebasecallBasecallError(
            "basecall_model_unresolved",
            f"the accepted plan's Dorado model bundle could not be resolved: {exc}",
        ) from exc


def _signal_source_paths(
    plan: RebasecallPlan,
    signal: MaterializedRebasecallSignal | None,
) -> tuple[Path, ...]:
    if signal is not None:
        return tuple(signal.source_paths)
    resolution = plan._source_resolution
    if resolution is None or not resolution.complete:
        raise RebasecallBasecallError(
            "basecall_sources_unavailable",
            "the accepted plan retained no complete source-signal resolution",
        )
    paths: list[Path] = []
    for row in resolution.rows:
        if row.resolved_path is None:
            raise RebasecallBasecallError(
                "basecall_sources_unavailable",
                f"source {row.source_id!r} has no resolved POD5 path",
            )
        try:
            sha256, size_bytes = checksum_input_source(row.resolved_path)
        except (InputManifestError, OSError, ValueError) as exc:
            raise RebasecallBasecallError(
                "basecall_source_changed",
                f"source {row.source_id!r} is missing or unreadable",
            ) from exc
        if sha256 != row.sha256 or size_bytes != row.size_bytes:
            raise RebasecallBasecallError(
                "basecall_source_changed",
                f"source {row.source_id!r} changed after the accepted plan was produced",
            )
        paths.append(Path(row.resolved_path))
    return tuple(paths)


def _basecall_identity(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    resolution: DoradoBasecallResolution,
    signal: MaterializedRebasecallSignal | None,
    requested_read_ids: Sequence[str],
) -> dict[str, object]:
    """Return the path-neutral reuse identity for one selective basecall.

    Reuse turns on the frozen selection *and* the resolved model bundle, so two
    requests carrying the same floating alias over the same reads cannot reuse
    one another once the installed models differ.
    """
    source_resolution = plan._source_resolution
    return {
        "schema_version": REBASECALL_BASECALL_SCHEMA_VERSION,
        "selection_id": selection.selection_id,
        "signal_id": None if signal is None else signal.signal_id,
        "source_manifest_digest": plan.sources.manifest_digest,
        "source_resolution_digest": (
            None if source_resolution is None else source_resolution.digest
        ),
        "requested_read_count": len(requested_read_ids),
        "requested_read_ids_digest": _read_ids_digest(requested_read_ids),
        "generation_kind": _generation_kind(plan.request),
        "model": resolution.semantic_payload(),
    }


def _model_names(resolution: DoradoBasecallResolution) -> set[str]:
    return {resolution.simplex_model.name} | {
        model.name for model in resolution.modification_models
    }


def _validate_header(
    inspection: BasecallOutputInspection,
    resolution: DoradoBasecallResolution,
) -> dict[str, object]:
    """Confirm the BAM header agrees with the exact executed Dorado identity."""
    dorado_programs = [
        record
        for record in inspection.programs
        if str(record.get("PN", "")).lower() == "dorado"
        or str(record.get("ID", "")).lower().startswith("basecaller")
    ]
    if not dorado_programs:
        raise RebasecallBasecallError(
            "basecall_header_mismatch",
            "the output BAM header records no Dorado program group",
        )
    versions = {str(record.get("VN", "")) for record in dorado_programs}
    if resolution.dorado_version not in versions:
        raise RebasecallBasecallError(
            "basecall_header_mismatch",
            "the output BAM header does not record the resolved Dorado version",
        )
    expected_models = _model_names(resolution)
    observed_models: set[str] = set()
    for group in inspection.read_groups:
        description = str(group.get("DS", ""))
        for field in description.split():
            key, _, value = field.partition("=")
            if key == "basecall_model" and value:
                observed_models.add(value)
    if not observed_models:
        raise RebasecallBasecallError(
            "basecall_header_mismatch",
            "the output BAM header records no basecall model in any read group",
        )
    if not observed_models.issubset(expected_models):
        raise RebasecallBasecallError(
            "basecall_model_mismatch",
            "the output BAM was produced by a model outside the resolved bundle: "
            f"{sorted(observed_models - expected_models)}",
        )
    return {
        "dorado_version": resolution.dorado_version,
        "observed_models": sorted(observed_models),
        "read_group_count": len(inspection.read_groups),
        "program_count": len(inspection.programs),
    }


def _validate_records(
    inspection: BasecallOutputInspection,
    requested: set[str],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Confirm every output record descends from an exactly requested POD5 read."""
    seen: set[str] = set()
    duplicates = 0
    per_parent: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    foreign: set[str] = set()
    for read_id, parent_id in zip(inspection.read_ids, inspection.parent_ids, strict=True):
        if read_id in seen:
            duplicates += 1
        seen.add(read_id)
        parent = parent_id or read_id
        if parent not in requested:
            foreign.add(parent)
            continue
        per_parent[parent] = per_parent.get(parent, 0) + 1
        rows.append(
            {
                "read_id": read_id,
                "basecall_parent_read_id": parent_id,
                "pod5_read_id": parent,
            }
        )
    if duplicates:
        raise RebasecallBasecallError(
            "basecall_duplicate_read_id",
            f"the output BAM contains {duplicates} duplicate read ID(s)",
        )
    if foreign:
        raise RebasecallBasecallError(
            "basecall_foreign_parent",
            f"the output BAM contains {len(foreign)} record parent(s) outside the selection",
        )
    observed_parents = set(per_parent)
    split_children = sum(count - 1 for count in per_parent.values() if count > 1)
    counts = {
        "requested_unique_read_count": len(requested),
        "source_parent_observed_count": len(observed_parents),
        "output_record_count": inspection.record_count,
        "split_child_record_count": split_children,
        "missing_read_count": len(requested - observed_parents),
        "duplicate_output_read_id_count": duplicates,
    }
    return counts, rows


def _locate_outputs(output_directory: Path) -> tuple[Path, Path | None]:
    bams = sorted(path for path in output_directory.rglob("*.bam") if path.is_file())
    if not bams:
        raise RebasecallBasecallError(
            "basecall_output_missing",
            "Dorado produced no output BAM",
        )
    if len(bams) > 1:
        raise RebasecallBasecallError(
            "basecall_output_ambiguous",
            f"Dorado produced {len(bams)} output BAMs where exactly one was expected",
        )
    summary: Path | None = None
    for candidate in _SUMMARY_CANDIDATES:
        matches = sorted(path for path in output_directory.rglob(candidate) if path.is_file())
        if matches:
            summary = matches[0]
            break
    return bams[0], summary


def _artifact_record(directory: Path, name: str, kind: str) -> dict[str, object]:
    path = directory / name
    try:
        sha256, size_bytes = checksum_input_source(path)
    except (InputManifestError, OSError, ValueError) as exc:
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            f"published basecall artifact {name!r} is missing or unreadable",
        ) from exc
    return {"path": name, "kind": kind, "sha256": sha256, "size_bytes": size_bytes}


def read_published_rebasecall_basecall(
    directory: str | Path,
    *,
    expected_basecall_id: str | None = None,
) -> PublishedRebasecallBasecall:
    """Read and fully revalidate one published basecall artifact."""
    directory = Path(directory)
    manifest_path = directory / BASECALL_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall manifest is missing or invalid",
        ) from exc
    required_manifest_keys = {
        "schema_version",
        "basecall_id",
        "accepted_plan_id",
        "request_id",
        "experiment_id",
        "selection_id",
        "signal_id",
        "generation_kind",
        "identity",
        "dorado",
        "counts",
        "outputs",
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != REBASECALL_BASECALL_SCHEMA_VERSION
        or set(manifest) != required_manifest_keys
    ):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall manifest does not match schema 1",
        )
    basecall_id = str(manifest["basecall_id"])
    if not _is_sha256(basecall_id) or (
        expected_basecall_id is not None and basecall_id != expected_basecall_id
    ):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall identity does not match the expected artifact",
        )
    identity = manifest["identity"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version") != REBASECALL_BASECALL_SCHEMA_VERSION
        or _sha256_payload(identity) != basecall_id
        or identity.get("selection_id") != manifest.get("selection_id")
        or identity.get("signal_id") != manifest.get("signal_id")
        or identity.get("generation_kind") != manifest.get("generation_kind")
    ):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall semantic identity is inconsistent",
        )
    if manifest["generation_kind"] not in set(SELECTION_GENERATION_KINDS.values()):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall generation kind is invalid",
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall records no output artifacts",
        )
    expected_output_keys = {"path", "kind", "sha256", "size_bytes"}
    names: list[str] = []
    for output in outputs:
        if not isinstance(output, dict) or set(output) != expected_output_keys:
            raise RebasecallBasecallError(
                "basecall_artifact_invalid",
                "published basecall output fields are invalid",
            )
        name = str(output["path"])
        if not _is_sha256(output.get("sha256")):
            raise RebasecallBasecallError(
                "basecall_artifact_invalid",
                f"published basecall output {name!r} has an invalid checksum",
            )
        _manifest_count(output.get("size_bytes"), "output size")
        path = directory / name
        try:
            sha256, size_bytes = checksum_input_source(path)
        except (InputManifestError, OSError, ValueError) as exc:
            raise RebasecallBasecallError(
                "basecall_artifact_invalid",
                f"published basecall output {name!r} is missing or unreadable",
            ) from exc
        if sha256 != output["sha256"] or size_bytes != output["size_bytes"]:
            raise RebasecallBasecallError(
                "basecall_artifact_invalid",
                f"published basecall output {name!r} does not match its manifest checksum",
            )
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall outputs are not unique and deterministic",
        )
    if BASECALL_BAM_FILENAME not in names or BASECALL_ORIGIN_FILENAME not in names:
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall is missing its calls or origin-map artifact",
        )
    counts = manifest.get("counts")
    required_count_keys = {
        "selection_record_count",
        "requested_unique_read_count",
        "source_parent_observed_count",
        "output_record_count",
        "split_child_record_count",
        "missing_read_count",
        "duplicate_output_read_id_count",
    }
    if not isinstance(counts, dict) or set(counts) != required_count_keys:
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall counts are invalid",
        )
    for field_name, value in counts.items():
        _manifest_count(value, field_name.replace("_", " "))
    if (
        counts["requested_unique_read_count"] != identity.get("requested_read_count")
        or counts["source_parent_observed_count"] + counts["missing_read_count"]
        != counts["requested_unique_read_count"]
        or counts["output_record_count"]
        != counts["source_parent_observed_count"] + counts["split_child_record_count"]
        or counts["duplicate_output_read_id_count"] != 0
    ):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall aggregate counts are not self-consistent",
        )
    dorado = manifest.get("dorado")
    required_dorado_keys = {
        "version",
        "simplex_model",
        "modification_models",
        "model_bundle_digest",
        "capability_digest",
        "normalized_argv",
        "options",
        "header",
    }
    model = identity.get("model")
    if (
        not isinstance(dorado, dict)
        or set(dorado) != required_dorado_keys
        or not isinstance(model, dict)
        or dorado.get("model_bundle_digest") != model.get("model_bundle_digest")
        or dorado.get("capability_digest") != model.get("capability_digest")
        or dorado.get("version") != model.get("dorado_version")
    ):
        raise RebasecallBasecallError(
            "basecall_artifact_invalid",
            "published basecall Dorado identity is inconsistent with its reuse identity",
        )
    return PublishedRebasecallBasecall(
        basecall_id=basecall_id,
        directory=directory,
        manifest_path=manifest_path,
        bam_path=directory / BASECALL_BAM_FILENAME,
        manifest=manifest,
    )


def execute_rebasecall_basecall(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    basecall_root: str | Path,
    *,
    accepted_plan_id: str,
    signal: MaterializedRebasecallSignal | None = None,
    model_directory: str | Path | None = None,
    dorado_resolver: Callable[..., DoradoBasecallResolution] | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_dorado,
    bam_inspector: Callable[[Path], BasecallOutputInspection] = _inspect_bam,
) -> PublishedRebasecallBasecall:
    """Execute Dorado for an accepted plan and publish one immutable basecall."""
    if accepted_plan_id != plan.plan_id:
        raise RebasecallBasecallError(
            "accepted_plan_mismatch",
            "the supplied accepted plan ID does not match the current plan",
        )
    if plan.status != "ready":
        raise RebasecallBasecallError(
            "accepted_plan_blocked",
            "a blocked re-basecall plan cannot be executed",
        )
    if plan.request.signal.materialize and signal is None:
        # Silently basecalling the originals would produce a result the accepted
        # request did not ask for, under an identity that claims no signal.
        raise RebasecallBasecallError(
            "basecall_signal_missing",
            "the accepted request materializes signal, so a materialized artifact is required",
        )
    if signal is not None and signal.manifest.get("selection_id") != selection.selection_id:
        raise RebasecallBasecallError(
            "basecall_signal_mismatch",
            "the materialized signal was built from a different frozen selection",
        )
    validated_selection = _validated_selection(plan, selection)
    origin, selection_record_count = _selection_origin(validated_selection)
    requested_read_ids = tuple(sorted(origin))
    source_paths = _signal_source_paths(plan, signal)
    resolution = _resolution_for_plan(
        plan,
        dorado_resolver=dorado_resolver,
        source_paths=source_paths,
        model_directory=model_directory,
    )
    identity = _basecall_identity(
        plan,
        validated_selection,
        resolution,
        signal,
        requested_read_ids,
    )
    basecall_id = _sha256_payload(identity)
    basecall_root = Path(basecall_root)
    destination = basecall_root / basecall_id
    if destination.exists():
        return read_published_rebasecall_basecall(
            destination,
            expected_basecall_id=basecall_id,
        )

    basecall_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{basecall_id}.", suffix=".tmp", dir=basecall_root))
    try:
        read_ids_path = temporary / BASECALL_READ_IDS_FILENAME
        read_ids_path.write_text("\n".join(requested_read_ids) + "\n", encoding="utf-8")

        # Dorado reads one data path, so present every resolved source through a
        # staging directory rather than merging signal.
        signal_dir = temporary / "signal"
        signal_dir.mkdir()
        for ordinal, source_path in enumerate(source_paths):
            link = signal_dir / f"{ordinal:06d}.pod5"
            try:
                os.symlink(Path(source_path).resolve(), link)
            except OSError:
                shutil.copy2(Path(source_path), link)

        output_directory = temporary / "dorado"
        output_directory.mkdir()
        argv = build_dorado_basecaller_argv(
            resolution,
            signal_dir,
            read_ids_path,
            output_directory,
        )
        try:
            completed = runner(argv)
        except OSError as exc:
            raise RebasecallBasecallError(
                "basecall_execution_failed",
                f"Dorado could not be executed: {type(exc).__name__}: {exc}",
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise RebasecallBasecallError(
                "basecall_execution_failed",
                "Dorado exited with status "
                f"{completed.returncode}: {detail[-1] if detail else 'no diagnostic output'}",
            )

        produced_bam, produced_summary = _locate_outputs(output_directory)
        os.replace(produced_bam, temporary / BASECALL_BAM_FILENAME)
        if produced_summary is not None:
            os.replace(produced_summary, temporary / BASECALL_SUMMARY_FILENAME)

        inspection = bam_inspector(temporary / BASECALL_BAM_FILENAME)
        header = _validate_header(inspection, resolution)
        counts, origin_rows = _validate_records(inspection, set(requested_read_ids))

        for row in origin_rows:
            row["pod5_source_id"] = origin[str(row["pod5_read_id"])]
        pd.DataFrame(
            origin_rows,
            columns=[
                "read_id",
                "basecall_parent_read_id",
                "pod5_read_id",
                "pod5_source_id",
            ],
        ).to_csv(temporary / BASECALL_ORIGIN_FILENAME, index=False)

        # The staging tree is removed on publication, so nothing derived from it
        # may enter the manifest: only path-neutral argv and artifact names.
        shutil.rmtree(signal_dir)
        shutil.rmtree(output_directory)

        outputs = [
            _artifact_record(temporary, BASECALL_BAM_FILENAME, "calls"),
            _artifact_record(temporary, BASECALL_ORIGIN_FILENAME, "origin_map"),
            _artifact_record(temporary, BASECALL_READ_IDS_FILENAME, "read_ids"),
        ]
        if (temporary / BASECALL_SUMMARY_FILENAME).exists():
            outputs.append(
                _artifact_record(temporary, BASECALL_SUMMARY_FILENAME, "sequencing_summary")
            )
        outputs.sort(key=lambda record: str(record["path"]))

        manifest = {
            "schema_version": REBASECALL_BASECALL_SCHEMA_VERSION,
            "basecall_id": basecall_id,
            "accepted_plan_id": plan.plan_id,
            "request_id": plan.request.request_id,
            "experiment_id": plan.experiment_id,
            "selection_id": validated_selection.selection_id,
            "signal_id": None if signal is None else signal.signal_id,
            "generation_kind": identity["generation_kind"],
            "identity": dict(identity),
            "dorado": {
                "version": resolution.dorado_version,
                "simplex_model": resolution.simplex_model.to_dict(),
                "modification_models": [
                    model.to_dict() for model in resolution.modification_models
                ],
                "model_bundle_digest": resolution.model_bundle_digest,
                "capability_digest": resolution.capability_digest,
                "normalized_argv": list(resolution.normalized_argv),
                "options": resolution.options.semantic_payload(),
                "header": header,
            },
            "counts": {"selection_record_count": selection_record_count, **counts},
            "outputs": outputs,
        }
        atomic_write_json(temporary / BASECALL_MANIFEST_FILENAME, manifest)
        read_published_rebasecall_basecall(temporary, expected_basecall_id=basecall_id)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
            read_published_rebasecall_basecall(destination, expected_basecall_id=basecall_id)
            shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_published_rebasecall_basecall(destination, expected_basecall_id=basecall_id)


def prepare_rebasecall_basecall(
    cfg: Any,
    request: RebasecallRequest,
    selection_root: str | Path,
    basecall_root: str | Path,
    *,
    accepted_plan_id: str,
    signal_root: str | Path | None = None,
    pod5_indexer: Callable[
        [tuple[tuple[str, Path], ...]], Pod5DatasetIndex
    ] = build_pod5_dataset_index,
    bam_tag_reader: Callable[[Path], Mapping[str, Mapping[str, object]]] | None = None,
    parent_validator: Callable[[RebasecallPlan], None] | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_dorado,
    bam_inspector: Callable[[Path], BasecallOutputInspection] = _inspect_bam,
    dorado_resolver: Callable[..., DoradoBasecallResolution] | None = None,
) -> PublishedRebasecallBasecall:
    """Rebuild an accepted plan, freeze selection, and publish a validated basecall."""
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
    signal: MaterializedRebasecallSignal | None = None
    if request.signal.materialize:
        if signal_root is None:
            raise RebasecallBasecallError(
                "signal_root_required",
                "the request enables signal materialization but no signal root was supplied",
            )
        signal = materialize_rebasecall_signal(
            plan,
            selection,
            signal_root,
            accepted_plan_id=accepted_plan_id,
            pod5_indexer=pod5_indexer,
        )
    return execute_rebasecall_basecall(
        plan,
        selection,
        basecall_root,
        accepted_plan_id=accepted_plan_id,
        signal=signal,
        model_directory=getattr(cfg, "model_dir", None),
        dorado_resolver=dorado_resolver,
        runner=runner,
        bam_inspector=bam_inspector,
    )
