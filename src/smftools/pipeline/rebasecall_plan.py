"""Read-only parent and selection planning for selective POD5 re-basecalling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ..constants import PREPROCESS_DIR, RAW_DIR
from ..informatics.experiment_manifest import (
    read_experiment_manifest,
    resolve_artifact_record,
)
from ..informatics.generation import GENERATIONS_SUBDIR
from ..informatics.input_manifest import (
    ResolvedInputManifest,
    read_resolved_input_manifest,
)
from ..informatics.partition_read import resolve_relative_path
from ..informatics.pod5_identity import (
    Pod5DatasetIndex,
    Pod5IdentityResolution,
    build_pod5_dataset_index,
    resolve_pod5_identities,
)
from ..informatics.raw_generation import (
    RAW_GENERATIONS_SUBDIR,
    RawGenerationError,
    resolve_current_raw_generation,
    validate_raw_generation,
)
from ..preprocessing.partitioned_executor import PREPROCESS_STAGE_OBS
from .rebasecall_request import (
    RebasecallRequest,
    RebasecallRequestError,
    load_rebasecall_request,
)

REBASECALL_PLAN_SCHEMA_VERSION = 1
_GENERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEFERRED_CAPABILITIES = (
    "selection_freezing:srb-01b",
    "source_checksum_relocation_and_replayability:srb-03",
    "dorado_and_model_bundle_resolution:srb-04",
    "lineage_execution_and_publication:srb-05",
)


class RebasecallPlanError(ValueError):
    """Raised when read-only planning cannot safely inspect requested state."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class RebasecallPlanReason:
    """One stable blocking reason or scientific-scope warning."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ParentGeneration:
    """One validated immutable parent selected by the request."""

    stage: str
    selector: str
    generation_id: str
    generation_dir: Path
    manifest: Mapping[str, Any] = field(repr=False, compare=False)
    molecule_count: int | None = None

    def to_dict(self, run_root: Path) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "selector": self.selector,
            "generation_id": self.generation_id,
            "path": self.generation_dir.relative_to(run_root).as_posix(),
            "molecule_count": self.molecule_count,
        }


@dataclass(frozen=True)
class RebasecallSourcePlan:
    """Published source-manifest inventory used by the selection plan."""

    manifest_digest: str | None = None
    source_count: int = 0
    pod5_source_count: int = 0
    source_ids: tuple[str, ...] = ()
    recorded_paths_available: int = 0
    relocation_candidates: int = 0
    signal_read_count: int | None = None
    signal_count_complete: bool = False
    duplicate_read_id_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_digest": self.manifest_digest,
            "source_count": self.source_count,
            "pod5_source_count": self.pod5_source_count,
            "source_ids": list(self.source_ids),
            "recorded_paths_available": self.recorded_paths_available,
            "relocation_candidates": self.relocation_candidates,
            "signal_read_count": self.signal_read_count,
            "signal_count_complete": self.signal_count_complete,
            "duplicate_read_id_count": self.duplicate_read_id_count,
        }


@dataclass(frozen=True)
class RebasecallSelectionPlan:
    """Deterministic counts from one immutable parent selection."""

    mode: str
    universe_count: int | None = None
    selected_count: int | None = None
    consumed_columns: tuple[str, ...] = ()
    id_kind: str | None = None
    requested_id_count: int | None = None
    matched_id_count: int | None = None
    missing_ids: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.universe_count is not None and self.selected_count is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "complete": self.complete,
            "universe_count": self.universe_count,
            "selected_count": self.selected_count,
            "consumed_columns": list(self.consumed_columns),
            "id_kind": self.id_kind,
            "requested_id_count": self.requested_id_count,
            "matched_id_count": self.matched_id_count,
            "missing_ids": list(self.missing_ids),
        }


@dataclass(frozen=True)
class RebasecallIdentityPlan:
    """Bounded summary of selected-molecule to POD5 identity resolution."""

    mode: str
    status: str
    selected_molecule_count: int = 0
    resolved_molecule_count: int = 0
    unique_pod5_read_count: int = 0
    duplicate_parent_reference_count: int = 0
    unresolved_count: int = 0
    ambiguous_count: int = 0
    evidence_counts: Mapping[str, int] = field(default_factory=dict)
    resolution_digest: str | None = None
    failures: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "selected_molecule_count": self.selected_molecule_count,
            "resolved_molecule_count": self.resolved_molecule_count,
            "unique_pod5_read_count": self.unique_pod5_read_count,
            "duplicate_parent_reference_count": self.duplicate_parent_reference_count,
            "unresolved_count": self.unresolved_count,
            "ambiguous_count": self.ambiguous_count,
            "evidence_counts": dict(self.evidence_counts),
            "resolution_digest": self.resolution_digest,
            "failures": [dict(failure) for failure in self.failures],
        }


@dataclass(frozen=True)
class RebasecallPlan:
    """Stable schema-1 read-only re-basecall plan."""

    request: RebasecallRequest
    experiment_id: str
    experiment_uid: str | None
    run_root: Path
    raw_parent: ParentGeneration | None
    preprocess_parent: ParentGeneration | None
    sources: RebasecallSourcePlan
    selection: RebasecallSelectionPlan
    identity: RebasecallIdentityPlan
    blockers: tuple[RebasecallPlanReason, ...] = ()
    warnings: tuple[RebasecallPlanReason, ...] = ()
    schema_version: int = REBASECALL_PLAN_SCHEMA_VERSION

    @property
    def selection_status(self) -> str:
        selection_block_codes = {
            "raw_parent_unavailable",
            "preprocess_parent_unavailable",
            "parent_generation_mismatch",
            "parent_observations_unreadable",
            "predicate_evaluation_failed",
            "selection_identity_column_missing",
            "selection_ids_missing",
        }
        if not self.selection.complete or any(
            blocker.code in selection_block_codes for blocker in self.blockers
        ):
            return "blocked"
        return "ready"

    @property
    def status(self) -> str:
        return "blocked" if self.blockers else "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request.request_id,
            "request": self.request.to_dict(),
            "experiment_id": self.experiment_id,
            "experiment_uid": self.experiment_uid,
            "status": self.status,
            "selection_status": self.selection_status,
            "execution_status": "not_implemented",
            "raw_parent": (
                None if self.raw_parent is None else self.raw_parent.to_dict(self.run_root)
            ),
            "preprocess_parent": (
                None
                if self.preprocess_parent is None
                else self.preprocess_parent.to_dict(self.run_root)
            ),
            "sources": self.sources.to_dict(),
            "selection": self.selection.to_dict(),
            "identity": self.identity.to_dict(),
            "requested_model": {
                "selector": self.request.basecall.model,
                "resolution_status": "deferred",
                "reason_code": "resolved_model_identity_requires_srb_04",
            },
            "downstream_target": self.request.downstream_target,
            "deferred_capabilities": list(_DEFERRED_CAPABILITIES),
            "blockers": [reason.to_dict() for reason in self.blockers],
            "warnings": [reason.to_dict() for reason in self.warnings],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), indent=indent)


def _validate_generation_selector(selector: str, stage: str) -> str:
    normalized = str(selector).strip()
    if normalized == "current":
        return normalized
    if not _GENERATION_ID_PATTERN.fullmatch(normalized):
        raise RebasecallPlanError(
            f"{stage}_parent_unavailable",
            f"requested {stage} generation ID is not portable: {normalized!r}",
        )
    return normalized


def _resolve_raw_parent(run_root: Path, selector: str) -> ParentGeneration:
    selector = _validate_generation_selector(selector, "raw")
    output_dir = run_root / RAW_DIR
    try:
        if selector == "current":
            selected = resolve_current_raw_generation(output_dir)
            if selected is None:
                raise RebasecallPlanError(
                    "raw_parent_unavailable", "the experiment has no immutable raw generation"
                )
            generation, manifest = selected
        else:
            generation = output_dir / RAW_GENERATIONS_SUBDIR / selector
            manifest = validate_raw_generation(
                generation,
                expected_generation_id=selector,
                run_root=run_root,
            )
    except RawGenerationError as exc:
        raise RebasecallPlanError("raw_parent_unavailable", str(exc)) from exc
    return ParentGeneration(
        stage="raw",
        selector=selector,
        generation_id=str(manifest["generation_id"]),
        generation_dir=generation,
        manifest=manifest,
    )


def _resolve_preprocess_parent(run_root: Path, selector: str) -> ParentGeneration:
    from ..preprocessing.preprocess_generation import (
        PreprocessGenerationError,
        resolve_current_preprocess_generation,
        validate_preprocess_generation,
    )

    selector = _validate_generation_selector(selector, "preprocess")
    output_dir = run_root / PREPROCESS_DIR
    try:
        if selector == "current":
            selected = resolve_current_preprocess_generation(output_dir)
            if selected is None:
                raise RebasecallPlanError(
                    "preprocess_parent_unavailable",
                    "the experiment has no immutable preprocess generation",
                )
            generation, manifest = selected
        else:
            generation = output_dir / GENERATIONS_SUBDIR / selector
            manifest = validate_preprocess_generation(
                generation,
                expected_generation_id=selector,
                run_root=run_root,
            )
    except PreprocessGenerationError as exc:
        raise RebasecallPlanError("preprocess_parent_unavailable", str(exc)) from exc
    return ParentGeneration(
        stage="preprocess",
        selector=selector,
        generation_id=str(manifest["generation_id"]),
        generation_dir=generation,
        manifest=manifest,
    )


def _read_raw_observations(parent: ParentGeneration) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(parent.generation_dir / "obs.parquet")
    except (OSError, ValueError, ImportError) as exc:
        raise RebasecallPlanError(
            "parent_observations_unreadable", "raw parent observations are unreadable"
        ) from exc
    if "read_id" not in frame.columns or frame["read_id"].astype(str).duplicated().any():
        raise RebasecallPlanError(
            "parent_observations_unreadable",
            "raw parent observations require unique read_id values",
        )
    frame = frame.copy()
    frame["read_id"] = frame["read_id"].astype(str)
    return frame.set_index("read_id", drop=False)


def _read_preprocess_observations(parent: ParentGeneration) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(parent.generation_dir / PREPROCESS_STAGE_OBS)
    except (OSError, ValueError, ImportError) as exc:
        raise RebasecallPlanError(
            "parent_observations_unreadable",
            "preprocess parent observations are unreadable",
        ) from exc
    if "read_id" not in frame.columns or frame["read_id"].astype(str).duplicated().any():
        raise RebasecallPlanError(
            "parent_observations_unreadable",
            "preprocess parent observations require unique read_id values",
        )
    frame = frame.copy()
    frame["read_id"] = frame["read_id"].astype(str)
    return frame.set_index("read_id", drop=False)


def _read_input_manifest(parent: ParentGeneration) -> ResolvedInputManifest:
    try:
        return read_resolved_input_manifest(
            parent.generation_dir / "input_manifest" / "resolved_input_manifest.json"
        )
    except ValueError as exc:
        raise RebasecallPlanError(
            "source_manifest_unavailable",
            f"raw parent input manifest cannot establish source signal: {exc}",
        ) from exc


def _preprocess_source_generation_id(
    parent: ParentGeneration,
    run_root: Path,
) -> str | None:
    """Resolve and cross-check the raw generation named by preprocess provenance."""
    source = parent.manifest.get("source")
    if not isinstance(source, Mapping):
        return None
    source_stage = source.get("stage")
    if source_stage not in {None, "raw"}:
        raise RebasecallPlanError(
            "parent_generation_mismatch",
            f"preprocess parent declares an unexpected source stage {source_stage!r}",
        )

    identities: set[str] = set()
    explicit = source.get("generation_id")
    if explicit not in {None, ""}:
        identities.add(str(explicit))

    artifact = source.get("artifact")
    if isinstance(artifact, Mapping):
        artifact_path = resolve_artifact_record(run_root, dict(artifact))
        generation_root = (run_root / RAW_DIR / GENERATIONS_SUBDIR).resolve()
        if artifact_path is not None:
            resolved_path = artifact_path.resolve()
            try:
                relative = resolved_path.relative_to(generation_root)
            except ValueError:
                relative = None
            if relative is not None and len(relative.parts) == 2 and relative.name == "spine.h5ad":
                identities.add(relative.parts[0])

    if len(identities) > 1:
        raise RebasecallPlanError(
            "parent_generation_mismatch",
            "preprocess parent source identities disagree",
        )
    return next(iter(identities), None)


def _source_plan(
    request: RebasecallRequest,
    parent: ParentGeneration,
    *,
    pod5_indexer: Callable[[tuple[tuple[str, Path], ...]], Pod5DatasetIndex],
) -> tuple[RebasecallSourcePlan, list[RebasecallPlanReason], Pod5DatasetIndex | None]:
    manifest = _read_input_manifest(parent)
    rows = manifest.rows
    recorded_available = sum(Path(row.path).is_file() for row in rows)
    pod5_rows = tuple(
        row for row in rows if row.source_kind == "pod5" and row.source_role == "raw_signal"
    )
    relocation_candidates = 0
    for row in rows:
        for candidate in request.signal.relocations:
            if (
                (candidate.source_id is None or candidate.source_id == row.source_id)
                and (candidate.sha256 is None or candidate.sha256.lower() == row.sha256.lower())
                and Path(candidate.path).is_file()
            ):
                relocation_candidates += 1
                break

    blockers: list[RebasecallPlanReason] = []
    if len(pod5_rows) != len(rows):
        blockers.append(
            RebasecallPlanReason(
                "source_signal_manifest_missing",
                "the selected raw generation was not produced from authoritative POD5 sources",
            )
        )
    missing_recorded = len(rows) - recorded_available
    if missing_recorded:
        message = f"{missing_recorded} recorded source path(s) are unavailable"
        if relocation_candidates:
            message += (
                "; request-local candidates are present but checksum relocation validation "
                "is deferred to SRB-03"
            )
        blockers.append(RebasecallPlanReason("source_paths_unavailable", message))

    signal_count = None
    count_complete = False
    duplicate_read_id_count = 0
    pod5_index = None
    if pod5_rows and len(pod5_rows) == len(rows) and not missing_recorded:
        try:
            pod5_index = pod5_indexer(tuple((row.source_id, Path(row.path)) for row in pod5_rows))
            signal_count = pod5_index.unique_read_count
            duplicate_read_id_count = pod5_index.duplicate_read_id_count
            count_complete = True
        except Exception as exc:
            blockers.append(
                RebasecallPlanReason(
                    "signal_inventory_unavailable",
                    f"POD5 read inventory could not be counted: {type(exc).__name__}: {exc}",
                )
            )
    if duplicate_read_id_count:
        blockers.append(
            RebasecallPlanReason(
                "signal_identity_ambiguous",
                f"{duplicate_read_id_count} POD5 read UUID(s) occur in multiple source locations",
            )
        )
    return (
        RebasecallSourcePlan(
            manifest_digest=manifest.digest,
            source_count=len(rows),
            pod5_source_count=len(pod5_rows),
            source_ids=tuple(row.source_id for row in rows),
            recorded_paths_available=recorded_available,
            relocation_candidates=relocation_candidates,
            signal_read_count=signal_count,
            signal_count_complete=count_complete,
            duplicate_read_id_count=duplicate_read_id_count,
        ),
        blockers,
        pod5_index,
    )


def _selection_plan(
    request: RebasecallRequest,
    raw_observations: pd.DataFrame,
    preprocess_observations: pd.DataFrame | None,
    sources: RebasecallSourcePlan,
) -> tuple[RebasecallSelectionPlan, pd.DataFrame]:
    selection = request.selection
    if selection.mode == "all-signal":
        return (
            RebasecallSelectionPlan(
                mode=selection.mode,
                universe_count=sources.signal_read_count,
                selected_count=sources.signal_read_count,
            ),
            raw_observations.iloc[0:0],
        )
    if selection.mode == "all-parent-molecules":
        return (
            RebasecallSelectionPlan(
                mode=selection.mode,
                universe_count=len(raw_observations),
                selected_count=len(raw_observations),
            ),
            raw_observations,
        )
    if selection.mode == "ids":
        assert selection.id_kind is not None
        if selection.id_kind not in raw_observations.columns:
            raise RebasecallPlanError(
                "selection_identity_column_missing",
                f"raw parent does not expose selection identity {selection.id_kind!r}",
            )
        available = set(raw_observations[selection.id_kind].dropna().astype(str))
        requested = set(selection.ids)
        missing = tuple(sorted(requested.difference(available)))
        selected_rows = raw_observations[
            raw_observations[selection.id_kind].astype("string").isin(requested)
        ]
        return (
            RebasecallSelectionPlan(
                mode=selection.mode,
                universe_count=len(raw_observations),
                selected_count=len(selected_rows),
                consumed_columns=(selection.id_kind,),
                id_kind=selection.id_kind,
                requested_id_count=len(requested),
                matched_id_count=len(requested) - len(missing),
                missing_ids=missing,
            ),
            selected_rows,
        )

    assert selection.predicate is not None
    if preprocess_observations is None:
        raise RebasecallPlanError(
            "preprocess_parent_unavailable", "qc selection requires a preprocess parent"
        )
    overlap = [
        column for column in preprocess_observations.columns if column in raw_observations.columns
    ]
    joined = raw_observations.join(preprocess_observations.drop(columns=overlap), how="inner")
    try:
        selected = selection.predicate.evaluate(joined)
    except RebasecallRequestError as exc:
        raise RebasecallPlanError("predicate_evaluation_failed", str(exc)) from exc
    selected_rows = raw_observations.loc[joined.index[selected]]
    return (
        RebasecallSelectionPlan(
            mode=selection.mode,
            universe_count=len(joined),
            selected_count=len(selected_rows),
            consumed_columns=selection.predicate.columns,
        ),
        selected_rows,
    )


def _read_bam_pi(path: Path) -> Mapping[str, Mapping[str, object]]:
    """Read retained Dorado parent tags without loading alignment payloads."""
    from ..informatics.bam_functions import extract_read_tags_from_bam

    return extract_read_tags_from_bam(
        path,
        tag_names=["pi"],
        include_flags=False,
        include_cigar=False,
        primary_only=True,
    )


def _normalized_row_value(row: pd.Series, column: str) -> str | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _retained_bam_parents(
    observations: pd.DataFrame,
    run_root: Path,
    *,
    bam_tag_reader: Callable[[Path], Mapping[str, Mapping[str, object]]],
) -> tuple[dict[str, object], int]:
    """Recover historical ``pi`` values for selected rows that retain BAM paths."""
    if "bam_path" not in observations.columns:
        return {}, 0
    by_path: dict[Path, list[pd.Series]] = {}
    for _, row in observations.iterrows():
        stored_path = _normalized_row_value(row, "bam_path")
        path = resolve_relative_path(stored_path, run_root)
        if path is not None and path.is_file():
            by_path.setdefault(path, []).append(row)

    recovered: dict[str, object] = {}
    unreadable_count = 0
    for path in sorted(by_path, key=lambda item: item.as_posix()):
        try:
            tags = bam_tag_reader(path)
        except Exception:
            unreadable_count += 1
            continue
        for row in by_path[path]:
            observation_id = str(row["read_id"])
            aliases = (
                _normalized_row_value(row, "source_read_id"),
                _normalized_row_value(row, "basecall_read_id"),
                observation_id,
            )
            for alias in aliases:
                if alias is None or alias not in tags:
                    continue
                value = tags[alias].get("pi")
                if value is not None and str(value).strip():
                    recovered[observation_id] = value
                    break
    return recovered, unreadable_count


def _identity_plan_from_resolution(
    resolution: Pod5IdentityResolution,
) -> RebasecallIdentityPlan:
    failures = tuple(row.to_dict() for row in resolution.rows if row.status != "resolved")[:10]
    status = "blocked" if resolution.unresolved_count or resolution.ambiguous_count else "resolved"
    return RebasecallIdentityPlan(
        mode="molecule_resolution",
        status=status,
        selected_molecule_count=len(resolution.rows),
        resolved_molecule_count=resolution.resolved_count,
        unique_pod5_read_count=resolution.unique_pod5_read_count,
        duplicate_parent_reference_count=resolution.duplicate_parent_reference_count,
        unresolved_count=resolution.unresolved_count,
        ambiguous_count=resolution.ambiguous_count,
        evidence_counts=resolution.evidence_counts,
        resolution_digest=resolution.digest,
        failures=failures,
    )


def build_rebasecall_plan(
    cfg: Any,
    request: RebasecallRequest,
    *,
    pod5_indexer: Callable[
        [tuple[tuple[str, Path], ...]], Pod5DatasetIndex
    ] = build_pod5_dataset_index,
    bam_tag_reader: Callable[[Path], Mapping[str, Mapping[str, object]]] = _read_bam_pi,
) -> RebasecallPlan:
    """Inspect exact immutable parents and selection counts without writing artifacts."""
    run_root = Path(cfg.output_directory)
    experiment_manifest = read_experiment_manifest(run_root)
    experiment_id = str(
        getattr(cfg, "experiment_id", None)
        or getattr(cfg, "experiment_name", None)
        or experiment_manifest.get("experiment_id")
        or run_root.name
    )
    experiment_uid_raw = experiment_manifest.get("experiment_uid")
    experiment_uid = None if experiment_uid_raw is None else str(experiment_uid_raw)
    blockers: list[RebasecallPlanReason] = []
    warnings: list[RebasecallPlanReason] = []
    raw_parent = None
    preprocess_parent = None
    raw_observations = None
    preprocess_observations = None
    sources = RebasecallSourcePlan()
    selection = RebasecallSelectionPlan(mode=request.selection.mode)
    identity = RebasecallIdentityPlan(mode="unavailable", status="unavailable")
    pod5_index = None
    selected_observations = None

    if experiment_uid is None:
        blockers.append(
            RebasecallPlanReason(
                "experiment_identity_missing",
                "the experiment manifest has no durable experiment_uid",
            )
        )
    try:
        raw_parent = _resolve_raw_parent(run_root, request.source.raw_generation)
        raw_observations = _read_raw_observations(raw_parent)
        raw_parent = replace(raw_parent, molecule_count=len(raw_observations))
    except RebasecallPlanError as exc:
        blockers.append(RebasecallPlanReason(exc.code, str(exc)))

    if request.source.preprocess_generation is not None:
        try:
            preprocess_parent = _resolve_preprocess_parent(
                run_root, request.source.preprocess_generation
            )
            source_generation_id = _preprocess_source_generation_id(
                preprocess_parent,
                run_root,
            )
            if raw_parent is not None and source_generation_id != raw_parent.generation_id:
                raise RebasecallPlanError(
                    "parent_generation_mismatch",
                    "preprocess parent was not produced from the selected raw generation",
                )
            preprocess_observations = _read_preprocess_observations(preprocess_parent)
            preprocess_parent = replace(
                preprocess_parent,
                molecule_count=len(preprocess_observations),
            )
        except RebasecallPlanError as exc:
            blockers.append(RebasecallPlanReason(exc.code, str(exc)))

    if raw_parent is not None:
        try:
            sources, source_blockers, pod5_index = _source_plan(
                request,
                raw_parent,
                pod5_indexer=pod5_indexer,
            )
            blockers.extend(source_blockers)
        except RebasecallPlanError as exc:
            blockers.append(RebasecallPlanReason(exc.code, str(exc)))

    if raw_observations is not None:
        try:
            selection, selected_observations = _selection_plan(
                request,
                raw_observations,
                preprocess_observations,
                sources,
            )
            if selection.missing_ids:
                blockers.append(
                    RebasecallPlanReason(
                        "selection_ids_missing",
                        f"{len(selection.missing_ids)} requested selection ID(s) are absent",
                    )
                )
        except RebasecallPlanError as exc:
            blockers.append(RebasecallPlanReason(exc.code, str(exc)))

    if request.selection.mode == "all-signal" and pod5_index is not None:
        identity = RebasecallIdentityPlan(
            mode="signal_inventory",
            status="resolved" if not pod5_index.duplicate_read_id_count else "blocked",
            unique_pod5_read_count=pod5_index.unique_read_count,
            ambiguous_count=pod5_index.duplicate_read_id_count,
            evidence_counts={"pod5_dataset_index": pod5_index.unique_read_count},
        )
    elif selected_observations is not None and pod5_index is not None:
        initial = resolve_pod5_identities(selected_observations, pod5_index)
        fallback_ids = {
            row.observation_id
            for row in initial.rows
            if row.status == "unresolved" and row.evidence == "no_supported_identity"
        }
        bam_parents, unreadable_bam_count = _retained_bam_parents(
            selected_observations[selected_observations["read_id"].astype(str).isin(fallback_ids)],
            run_root,
            bam_tag_reader=bam_tag_reader,
        )
        if unreadable_bam_count:
            blockers.append(
                RebasecallPlanReason(
                    "retained_bam_identity_unavailable",
                    f"{unreadable_bam_count} retained BAM source(s) could not be read for pi recovery",
                )
            )
        resolution = resolve_pod5_identities(
            selected_observations,
            pod5_index,
            bam_parent_by_observation=bam_parents,
        )
        identity = _identity_plan_from_resolution(resolution)
        if resolution.unresolved_count:
            blockers.append(
                RebasecallPlanReason(
                    "pod5_identity_unresolved",
                    f"{resolution.unresolved_count} selected molecule(s) have no authoritative POD5 UUID",
                )
            )
        if resolution.ambiguous_count:
            blockers.append(
                RebasecallPlanReason(
                    "pod5_identity_ambiguous",
                    f"{resolution.ambiguous_count} selected molecule(s) map to non-unique POD5 signal",
                )
            )

    if request.selection.mode == "all-signal":
        warnings.append(
            RebasecallPlanReason(
                "full_signal_scope",
                "all-signal can reassess the full authoritative signal universe",
            )
        )
    elif request.selection.mode == "all-parent-molecules":
        warnings.append(
            RebasecallPlanReason(
                "parent_universe_scope",
                "the request cannot recover signal reads absent from the parent raw generation",
            )
        )
    else:
        warnings.append(
            RebasecallPlanReason(
                "selected_cohort_scope",
                "the request is conditioned on a parent cohort and is not a full-signal reanalysis",
            )
        )
    if request.selection.mode != "all-signal":
        warnings.append(
            RebasecallPlanReason(
                "cohort_dependent_recomputation",
                "deduplication and fitted downstream analyses may differ when run on a subset",
            )
        )

    return RebasecallPlan(
        request=request,
        experiment_id=experiment_id,
        experiment_uid=experiment_uid,
        run_root=run_root,
        raw_parent=raw_parent,
        preprocess_parent=preprocess_parent,
        sources=sources,
        selection=selection,
        identity=identity,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def plan_rebasecall(config_path: str | Path, request_path: str | Path) -> RebasecallPlan:
    """Load a config and request, then build a read-only plan."""
    from ..cli.helpers import load_experiment_config

    cfg = load_experiment_config(str(config_path))
    request = load_rebasecall_request(request_path)
    return build_rebasecall_plan(cfg, request)


def format_rebasecall_plan(plan: RebasecallPlan) -> str:
    """Render a deterministic human summary of the read-only plan."""
    raw_id = plan.raw_parent.generation_id if plan.raw_parent is not None else "unavailable"
    preprocess_id = (
        plan.preprocess_parent.generation_id
        if plan.preprocess_parent is not None
        else "not requested"
    )
    selected = (
        "unknown" if plan.selection.selected_count is None else str(plan.selection.selected_count)
    )
    universe = (
        "unknown" if plan.selection.universe_count is None else str(plan.selection.universe_count)
    )
    lines = [
        f"Re-basecall request: {plan.request.name} ({plan.request.request_id})",
        f"Experiment: {plan.experiment_id} (UID {plan.experiment_uid or 'missing'})",
        f"Plan status: {plan.status}; selection {plan.selection_status}",
        f"Raw parent: {raw_id}",
        f"Preprocess parent: {preprocess_id}",
        f"Selection: {plan.selection.mode}; {selected}/{universe} molecule(s)",
        f"POD5 identity: {plan.identity.status}; "
        f"{plan.identity.resolved_molecule_count}/{plan.identity.selected_molecule_count} molecule(s), "
        f"{plan.identity.unique_pod5_read_count} unique signal read(s)",
        f"Sources: {plan.sources.pod5_source_count}/{plan.sources.source_count} POD5; "
        f"signal reads {plan.sources.signal_read_count if plan.sources.signal_read_count is not None else 'unknown'}",
        f"Requested model: {plan.request.basecall.model} (resolution deferred to SRB-04)",
        f"Downstream target: {plan.request.downstream_target}",
        "Execution: unavailable; this command writes no scientific artifacts.",
        "",
        "Blockers:",
    ]
    lines.extend([f"- {reason.code}: {reason.message}" for reason in plan.blockers] or ["- none"])
    lines.extend(("", "Warnings:"))
    lines.extend([f"- {reason.code}: {reason.message}" for reason in plan.warnings] or ["- none"])
    return "\n".join(lines)
