"""Partitioned-native variant evidence over raw ragged molecule shards."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from ..constants import (
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    VARIANT_EVIDENCE_GENERATION_SCHEMA_VERSION,
    VARIANT_EVIDENCE_INDEX_SCHEMA_VERSION,
    VARIANT_EVIDENCE_TASK_SCHEMA_VERSION,
)
from ..informatics.experiment_manifest import artifact_record
from ..informatics.molecule_identity import (
    EXPERIMENT_UID_COLUMN,
    MOLECULE_UID_COLUMN,
)
from ..informatics.partition_read import load_spine
from ..informatics.physical_layout import portable_parquet_row_group_rows
from ..informatics.ragged_store import (
    CIGAR,
    READ_ID,
    REFERENCE_START,
    SEQUENCE,
    iter_cigar_aligned_pairs,
)
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..logging_utils import get_logger
from ..readwrite import atomic_write_json
from .variant_evidence import (
    call_observed_variant_sites,
    segment_sparse_variant_calls,
)
from .variant_reference import (
    VariantInformativeSiteCatalog,
    VariantReferenceSet,
    calculate_variant_informative_sites,
    conversion_substitutions_for_strand,
)

logger = get_logger(__name__)

VARIANT_TASK_CATALOG = "task_catalog.parquet"
VARIANT_OBS_SIDECAR = "variant_obs"
VARIANT_REFERENCE_CATALOG = "reference_catalog.json"
VARIANT_GENERATION_MANIFEST = "generation_manifest.json"
VARIANT_TASK_STORE = "task_store"
VARIANT_READ_INDEX = "read_index"

EVIDENCE_COMPLETE = "complete"
BLOCKED_MISSING_INPUT = "blocked_missing_input"

_BASE_DECODER = {
    int(value): str(base).upper() for base, value in MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT.items()
}
_TASK_REQUIRED_COLUMNS = frozenset({READ_ID, REFERENCE_START, CIGAR, SEQUENCE})


def _digest(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as parquet

    return set(parquet.ParquetFile(path).schema_arrow.names)


@dataclass(frozen=True)
class VariantEvidenceTask:
    """One raw-shard/reference-set task with one owner per molecule."""

    task_id: str
    experiment_uid: str
    variant_reference_set_id: str
    reference: str
    aligned_member_index: int
    group_path: str
    n_reads: int
    aligned_bases: int
    estimated_memory_bytes: int
    identities: tuple[tuple[str, str], ...]
    input_status: str = EVIDENCE_COMPLETE
    missing_inputs: tuple[str, ...] = ()
    schema_version: int = VARIANT_EVIDENCE_TASK_SCHEMA_VERSION

    def to_dict(self, *, include_identities: bool = False) -> dict[str, object]:
        """Return a stable scheduler/task-catalog record."""
        record = asdict(self)
        if not include_identities:
            record.pop("identities")
        return record


def plan_variant_evidence_tasks(
    spine_path: str | Path,
    reference_sets: Sequence[VariantReferenceSet],
) -> list[VariantEvidenceTask]:
    """Plan deterministic shard-owned tasks from the raw molecule spine."""
    spine_path = Path(spine_path)
    spine = load_spine(spine_path, verbose=False)
    required_obs = {
        "ragged_shard",
        "Reference_strand",
        EXPERIMENT_UID_COLUMN,
        MOLECULE_UID_COLUMN,
    }
    missing_obs = required_obs.difference(spine.obs.columns)
    if missing_obs:
        raise ValueError(f"raw spine lacks variant task identity columns: {sorted(missing_obs)}")
    if len({item.reference_set_id for item in reference_sets}) != len(reference_sets):
        raise ValueError("variant reference sets must have unique identities")

    tasks: list[VariantEvidenceTask] = []
    grouped = spine.obs.groupby("ragged_shard", sort=True, observed=True)
    for reference_set in sorted(reference_sets, key=lambda item: item.reference_set_id):
        for group_path, obs in grouped:
            references = obs["Reference_strand"].astype(str).unique()
            if len(references) != 1:
                raise ValueError(f"raw shard {group_path!r} spans multiple references")
            reference = str(references[0])
            try:
                member_index = reference_set.resolve_member_index(reference)
            except KeyError:
                continue
            experiment_values = obs[EXPERIMENT_UID_COLUMN].astype(str).unique()
            if len(experiment_values) != 1:
                raise ValueError(f"raw shard {group_path!r} spans multiple experiments")
            shard_path = spine_path.parent / str(group_path)
            if not shard_path.is_file():
                raise FileNotFoundError(f"raw ragged shard is missing: {shard_path}")
            missing_inputs = tuple(
                sorted(_TASK_REQUIRED_COLUMNS.difference(_parquet_columns(shard_path)))
            )
            status = BLOCKED_MISSING_INPUT if missing_inputs else EVIDENCE_COMPLETE
            aligned_bases = int(obs.get("aligned_length", pd.Series(0, index=obs.index)).sum())
            estimated_memory = max(1, aligned_bases * 24 + len(obs) * 4096)
            task_key = f"{experiment_values[0]}\0{reference_set.reference_set_id}\0{group_path}"
            tasks.append(
                VariantEvidenceTask(
                    task_id=f"variant-{_digest(task_key)}",
                    experiment_uid=str(experiment_values[0]),
                    variant_reference_set_id=reference_set.reference_set_id,
                    reference=reference,
                    aligned_member_index=member_index,
                    group_path=str(group_path),
                    n_reads=len(obs),
                    aligned_bases=aligned_bases,
                    estimated_memory_bytes=estimated_memory,
                    identities=tuple(
                        zip(
                            obs.get("read_id", pd.Series(obs.index, index=obs.index)).astype(str),
                            obs[MOLECULE_UID_COLUMN].astype(str),
                            strict=True,
                        )
                    ),
                    input_status=status,
                    missing_inputs=missing_inputs,
                )
            )
    return sorted(tasks, key=lambda task: task.task_id)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        path,
        index=False,
        row_group_size=portable_parquet_row_group_rows(frame),
    )
    return path


def strand_of_reference(reference: str) -> str:
    """Strand context for a reference-strand label such as ``6B6_top``.

    This is the reference-strand *assignment* -- which converted reference the
    read aligned to -- and is the signal that selects conversion chemistry.
    Deliberately not the BAM reverse flag: on the `241213` pilot the two are
    independent (top reads split 7,227 fwd / 11,911 rev) and only this one
    separates corrupted sites from clean ones.
    """
    label = str(reference).strip().lower()
    for suffix in ("top", "bottom"):
        if label.endswith(f"_{suffix}"):
            return suffix
    return ""


def _observed_bases(row: pd.Series) -> dict[int, str]:
    sequence = list(row[SEQUENCE])
    observed: dict[int, str] = {}
    for query_position, reference_position in iter_cigar_aligned_pairs(
        str(row[CIGAR]),
        int(row[REFERENCE_START]),
    ):
        if query_position >= len(sequence):
            raise ValueError("ragged sequence is shorter than its CIGAR query span")
        base = _BASE_DECODER.get(int(sequence[query_position]), "N")
        if base != "N":
            observed[reference_position] = base
    return observed


def _execute_variant_task(
    spine_path: Path,
    output_dir: Path,
    task: VariantEvidenceTask,
    catalog: VariantInformativeSiteCatalog,
    min_adjacent_sites: int = 1,
) -> dict[str, object]:
    task_root = (
        output_dir
        / VARIANT_TASK_STORE
        / f"reference_set_id={task.variant_reference_set_id}"
        / f"task-{_digest(task.task_id)}"
    )
    obs_path = (
        output_dir
        / VARIANT_OBS_SIDECAR
        / f"set-{_digest(task.variant_reference_set_id)}"
        / f"task-{_digest(task.task_id)}.parquet"
    )
    calls_path = task_root / "calls.parquet"
    events_path = task_root / "events.parquet"
    identity_by_read = dict(task.identities)
    obs_rows: list[dict[str, object]] = []
    call_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    if task.input_status == EVIDENCE_COMPLETE:
        frame = pd.read_parquet(spine_path.parent / task.group_path)
        frame[READ_ID] = frame[READ_ID].astype(str)
        if set(frame[READ_ID]) != set(identity_by_read):
            raise ValueError(f"variant task {task.task_id!r} raw identities do not match spine")
        for row in frame.sort_values(READ_ID, kind="stable").to_dict("records"):
            read_id = str(row[READ_ID])
            molecule_uid = identity_by_read[read_id]
            observed = _observed_bases(pd.Series(row))
            calls, summary = call_observed_variant_sites(
                observed,
                aligned_member_index=task.aligned_member_index,
                catalog=catalog,
            )
            span_start = int(row[REFERENCE_START])
            from ..informatics.ragged_store import cigar_reference_length

            span_end = span_start + cigar_reference_length(str(row[CIGAR]))
            segmentation = segment_sparse_variant_calls(
                calls,
                span_start=span_start,
                span_end=span_end,
                aligned_member_index=task.aligned_member_index,
                min_adjacent_sites=min_adjacent_sites,
            )
            common = {
                EXPERIMENT_UID_COLUMN: task.experiment_uid,
                "read_id": read_id,
                MOLECULE_UID_COLUMN: molecule_uid,
                "variant_reference_set_id": task.variant_reference_set_id,
                "task_id": task.task_id,
            }
            obs_rows.append(
                {
                    **common,
                    "reference": task.reference,
                    "aligned_member_index": task.aligned_member_index,
                    "evidence_status": EVIDENCE_COMPLETE,
                    "informative_site_count": summary.informative_site_count,
                    "callable_site_count": summary.callable_site_count,
                    "no_call_count": summary.no_call_count,
                    "member_1_call_count": summary.member_call_counts[0],
                    "member_2_call_count": summary.member_call_counts[1],
                    "breakpoint_count": len(segmentation.breakpoints),
                    "has_breakpoint": segmentation.has_breakpoint,
                    "has_other_reference_segment": (segmentation.has_other_reference_segment),
                    "other_reference_segment_type": (segmentation.other_reference_segment_type),
                    "self_base_count": segmentation.self_base_count,
                    "other_base_count": segmentation.other_base_count,
                    "segment_cigar": segmentation.segment_cigar,
                }
            )
            for call in calls:
                call_rows.append(
                    {
                        **common,
                        "site_id": call.site_id,
                        "position": call.position,
                        "call": call.call,
                        "observed_base": call.observed_base,
                    }
                )
            for segment_index, segment in enumerate(segmentation.segments):
                event_rows.append(
                    {
                        **common,
                        "event_type": "segment",
                        "event_index": segment_index,
                        "start": segment.start,
                        "end": segment.end,
                        "state": segment.state,
                        "breakpoint": np.nan,
                    }
                )
            for breakpoint_index, breakpoint in enumerate(segmentation.breakpoints):
                event_rows.append(
                    {
                        **common,
                        "event_type": "breakpoint",
                        "event_index": breakpoint_index,
                        "start": np.nan,
                        "end": np.nan,
                        "state": np.nan,
                        "breakpoint": breakpoint,
                    }
                )
    else:
        for read_id, molecule_uid in task.identities:
            obs_rows.append(
                {
                    EXPERIMENT_UID_COLUMN: task.experiment_uid,
                    "read_id": read_id,
                    MOLECULE_UID_COLUMN: molecule_uid,
                    "variant_reference_set_id": task.variant_reference_set_id,
                    "task_id": task.task_id,
                    "reference": task.reference,
                    "aligned_member_index": task.aligned_member_index,
                    "evidence_status": BLOCKED_MISSING_INPUT,
                    "missing_inputs": ",".join(task.missing_inputs),
                }
            )

    obs_frame = pd.DataFrame(obs_rows)
    calls_frame = (
        pd.DataFrame(call_rows)
        if call_rows
        else _empty_frame(
            (
                EXPERIMENT_UID_COLUMN,
                "read_id",
                MOLECULE_UID_COLUMN,
                "variant_reference_set_id",
                "task_id",
                "site_id",
                "position",
                "call",
                "observed_base",
            )
        )
    )
    events_frame = (
        pd.DataFrame(event_rows)
        if event_rows
        else _empty_frame(
            (
                EXPERIMENT_UID_COLUMN,
                "read_id",
                MOLECULE_UID_COLUMN,
                "variant_reference_set_id",
                "task_id",
                "event_type",
                "event_index",
                "start",
                "end",
                "state",
                "breakpoint",
            )
        )
    )
    _write_parquet(obs_frame, obs_path)
    _write_parquet(calls_frame, calls_path)
    _write_parquet(events_frame, events_path)
    return {
        **task.to_dict(),
        "obs_path": obs_path.relative_to(output_dir).as_posix(),
        "calls_path": calls_path.relative_to(output_dir).as_posix(),
        "events_path": events_path.relative_to(output_dir).as_posix(),
        "evidence_rows": len(obs_frame),
        "call_rows": len(calls_frame),
        "event_rows": len(events_frame),
        "outcome": task.input_status,
    }


def _write_variant_read_index(
    output_dir: Path,
    task_records: pd.DataFrame,
) -> tuple[Path, int]:
    index_root = output_dir / VARIANT_READ_INDEX
    seen_keys: set[bytes] = set()
    evidence_count = 0
    for record in task_records.sort_values("task_id", kind="stable").to_dict("records"):
        indexed = pd.read_parquet(output_dir / str(record["obs_path"]))
        evidence_count += len(indexed)
        indexed["obs_path"] = str(record["obs_path"])
        indexed["calls_path"] = str(record["calls_path"])
        indexed["events_path"] = str(record["events_path"])
        indexed["molecule_bucket"] = "b" + indexed[MOLECULE_UID_COLUMN].astype(str).str[:2]
        for experiment_uid, molecule_uid, reference_set_id in zip(
            indexed[EXPERIMENT_UID_COLUMN].astype(str),
            indexed[MOLECULE_UID_COLUMN].astype(str),
            indexed["variant_reference_set_id"].astype(str),
            strict=True,
        ):
            key = hashlib.sha256(
                f"{experiment_uid}\0{molecule_uid}\0{reference_set_id}".encode("utf-8")
            ).digest()[:16]
            if key in seen_keys:
                raise RuntimeError("variant evidence ownership produced duplicate molecule results")
            seen_keys.add(key)
        for (reference_set_id, bucket), group in indexed.groupby(
            ["variant_reference_set_id", "molecule_bucket"],
            sort=True,
            observed=True,
        ):
            path = (
                index_root
                / f"reference_set_id={reference_set_id}"
                / f"molecule_bucket={bucket}"
                / f"part-task-{_digest(str(record['task_id']))}.parquet"
            )
            _write_parquet(group.drop(columns=["molecule_bucket"]), path)
    return index_root, evidence_count


def _resolve_generation_artifact(output_dir: Path, record: Mapping[str, object]) -> Path:
    raw_path = record.get("path")
    relative = Path(str(raw_path or ""))
    resolved = (output_dir / relative).resolve()
    if (
        record.get("path_kind") != "relative"
        or record.get("anchor") != "run_root"
        or not raw_path
        or relative.is_absolute()
        or not resolved.is_relative_to(output_dir.resolve())
    ):
        raise ValueError("variant generation artifact path is not relocation-safe")
    return resolved


def validate_variant_evidence_generation(output_dir: str | Path) -> dict[str, object]:
    """Validate one complete, relocation-safe variant evidence generation."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / VARIANT_GENERATION_MANIFEST
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("variant evidence generation manifest is unreadable") from exc
    if int(manifest.get("schema_version", -1)) != VARIANT_EVIDENCE_GENERATION_SCHEMA_VERSION:
        raise ValueError("variant evidence generation schema is incompatible")
    required_artifacts = {
        "task_store",
        "task_catalog",
        "obs",
        "read_index",
        "reference_catalog",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not required_artifacts.issubset(artifacts):
        raise ValueError("variant evidence generation artifacts are incomplete")
    resolved: dict[str, Path] = {}
    for key in required_artifacts:
        record = artifacts[key]
        if not isinstance(record, dict):
            raise ValueError(f"variant evidence artifact record is invalid: {key}")
        path = _resolve_generation_artifact(output_dir, record)
        if not path.exists():
            raise ValueError(f"variant evidence artifact is missing: {key}")
        observed = artifact_record(path, output_dir, checksum=True)
        if observed.get("sha256") != record.get("sha256"):
            raise ValueError(f"variant evidence artifact checksum does not match: {key}")
        resolved[key] = path

    tasks = pd.read_parquet(resolved["task_catalog"])
    if len(tasks) != int(manifest.get("task_count", -1)):
        raise ValueError("variant evidence task count does not match")
    keys = [EXPERIMENT_UID_COLUMN, MOLECULE_UID_COLUMN, "variant_reference_set_id"]
    import pyarrow.dataset as arrow_dataset

    obs_dataset = arrow_dataset.dataset(resolved["obs"], format="parquet")
    if not set(keys).issubset(obs_dataset.schema.names):
        raise ValueError("variant evidence molecule identity columns are missing")
    evidence_count = 0
    seen_keys: set[bytes] = set()
    for batch in obs_dataset.scanner(columns=keys).to_batches():
        frame = batch.to_pandas()
        evidence_count += len(frame)
        for values in frame.itertuples(index=False, name=None):
            key = hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).digest()[:16]
            if key in seen_keys:
                raise ValueError("variant evidence molecule ownership is invalid")
            seen_keys.add(key)
    if evidence_count != int(manifest.get("evidence_count", -1)):
        raise ValueError("variant evidence molecule count does not match")
    allowed_outcomes = {EVIDENCE_COMPLETE, BLOCKED_MISSING_INPUT}
    if not set(tasks["outcome"].astype(str)).issubset(allowed_outcomes):
        raise ValueError("variant evidence task outcome is invalid")
    for column in ("obs_path", "calls_path", "events_path"):
        for raw_path in tasks[column].astype(str):
            relative = Path(raw_path)
            resolved_task_path = (output_dir / relative).resolve()
            if relative.is_absolute() or not resolved_task_path.is_relative_to(
                output_dir.resolve()
            ):
                raise ValueError("variant task artifact path is not relocation-safe")
            if not resolved_task_path.is_file():
                raise ValueError("variant task artifact is missing")
    return manifest


def execute_partitioned_variant_evidence(
    spine_path: str | Path,
    reference_sets: Sequence[VariantReferenceSet],
    output_dir: str | Path,
    *,
    tasks: Iterable[VariantEvidenceTask] | None = None,
    max_workers: int = 1,
    memory_budget_mb: int = 512,
    cfg=None,
) -> dict[str, Path]:
    """Compute and index all-molecule variant evidence without monolithic AnnData."""
    if max_workers <= 0 or memory_budget_mb <= 0:
        raise ValueError("max_workers and memory_budget_mb must be positive")
    # Default 1 (no floor) when there is no cfg, so library callers keep the
    # historical behavior; pipeline runs get the configured floor. See `F14`.
    minimum_chimera_sites = max(1, int(getattr(cfg, "variant_chimera_min_adjacent_sites", 1) or 1))
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    owned_outputs = (
        VARIANT_TASK_STORE,
        VARIANT_TASK_CATALOG,
        VARIANT_OBS_SIDECAR,
        VARIANT_READ_INDEX,
        VARIANT_REFERENCE_CATALOG,
        VARIANT_GENERATION_MANIFEST,
    )
    existing_outputs = [name for name in owned_outputs if (output_dir / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            "variant evidence output must be a fresh immutable generation; "
            f"already present: {existing_outputs}"
        )
    task_list = (
        list(tasks)
        if tasks is not None
        else plan_variant_evidence_tasks(spine_path, reference_sets)
    )
    sets_by_id = {item.reference_set_id: item for item in reference_sets}
    if len(sets_by_id) != len(reference_sets):
        raise ValueError("variant reference sets must have unique identities")
    if not task_list:
        raise ValueError("no raw molecules align to the requested variant reference sets")
    unknown_task_sets = {task.variant_reference_set_id for task in task_list}.difference(sets_by_id)
    if unknown_task_sets:
        raise ValueError(
            f"variant tasks reference unknown reference sets: {sorted(unknown_task_sets)}"
        )
    # One catalog per (reference set, strand). Acceptance depends on the
    # chemistry a read of that strand could carry, so informative-site status is
    # strand-dependent; the reference set and its id are shared, so task
    # grouping is unaffected.
    modality = getattr(cfg, "smf_modality", None)
    conversion_types = list(getattr(cfg, "conversion_types", []) or [])
    task_strands = {strand_of_reference(task.reference) for task in task_list}
    catalogs = {}
    for identity, reference_set in sets_by_id.items():
        for strand in sorted(task_strands):
            substitutions = conversion_substitutions_for_strand(modality, conversion_types, strand)
            catalogs[(identity, strand)] = calculate_variant_informative_sites(
                reference_set,
                conversion_substitutions=substitutions,
                conversion_semantics=(
                    "none"
                    if not substitutions
                    else f"{'+'.join(str(m) for m in conversion_types)}:{strand}"
                ),
            )
            logger.info(
                "Variant informative sites for %s strand=%s: %d site(s) (conversion %s)",
                identity,
                strand or "unknown",
                len(catalogs[(identity, strand)].informative_sites),
                substitutions or "none",
            )
    maximum_task_bytes = max(task.estimated_memory_bytes for task in task_list)
    memory_workers = max(1, int(memory_budget_mb * 1024**2) // maximum_task_bytes)
    bounded_workers = min(max_workers, memory_workers, len(task_list))
    arguments = [
        (
            spine_path,
            output_dir,
            task,
            catalogs[(task.variant_reference_set_id, strand_of_reference(task.reference))],
            minimum_chimera_sites,
        )
        for task in task_list
    ]
    if cfg is not None:
        from ..memory_guard import run_tasks_parallel

        records = run_tasks_parallel(
            _execute_variant_task,
            arguments,
            cfg=cfg,
            pool_label=f"variant evidence tasks ({len(arguments)} tasks)",
            per_item_memory_mb=maximum_task_bytes / 1024**2,
            estimator="variant_evidence_task_peak",
            force_sequential=bounded_workers == 1,
        )
    elif bounded_workers == 1:
        records = [_execute_variant_task(*arguments_item) for arguments_item in arguments]
    else:
        with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
            records = list(executor.map(lambda values: _execute_variant_task(*values), arguments))

    task_frame = pd.DataFrame(records).sort_values("task_id", kind="stable").reset_index(drop=True)
    task_catalog_path = _write_parquet(task_frame, output_dir / VARIANT_TASK_CATALOG)
    obs_path = output_dir / VARIANT_OBS_SIDECAR
    read_index, evidence_count = _write_variant_read_index(output_dir, task_frame)

    reference_catalog_path = output_dir / VARIANT_REFERENCE_CATALOG
    atomic_write_json(
        reference_catalog_path,
        {
            "schema_version": VARIANT_EVIDENCE_GENERATION_SCHEMA_VERSION,
            # One catalog per strand: informative-site status depends on the
            # chemistry a read of that strand could carry, so a single catalog
            # cannot describe both. The reference set is shared.
            "reference_sets": [
                {
                    "reference_set": sets_by_id[identity].to_dict(),
                    "informative_site_catalogs": [
                        {
                            "strand": strand,
                            "catalog": catalogs[(catalog_identity, strand)].to_dict(),
                        }
                        for (catalog_identity, strand) in sorted(catalogs)
                        if catalog_identity == identity
                    ],
                }
                for identity in sorted(sets_by_id)
            ],
        },
    )
    generation_manifest_path = output_dir / VARIANT_GENERATION_MANIFEST
    generation_id = uuid4().hex
    artifacts = {
        "task_store": artifact_record(output_dir / VARIANT_TASK_STORE, output_dir, checksum=True),
        "task_catalog": artifact_record(task_catalog_path, output_dir, checksum=True),
        "obs": artifact_record(obs_path, output_dir, checksum=True),
        "read_index": artifact_record(read_index, output_dir, checksum=True),
        "reference_catalog": artifact_record(
            reference_catalog_path,
            output_dir,
            checksum=True,
        ),
    }
    atomic_write_json(
        generation_manifest_path,
        {
            "schema_version": VARIANT_EVIDENCE_GENERATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "task_schema_version": VARIANT_EVIDENCE_TASK_SCHEMA_VERSION,
            "index_schema_version": VARIANT_EVIDENCE_INDEX_SCHEMA_VERSION,
            "task_count": len(task_frame),
            "evidence_count": evidence_count,
            "artifacts": artifacts,
        },
    )
    validate_variant_evidence_generation(output_dir)
    manifest = sidecar_manifest_path(output_dir)
    for key, path in {
        "variant_task_store": output_dir / VARIANT_TASK_STORE,
        "variant_task_catalog": task_catalog_path,
        "variant_obs": obs_path,
        "variant_read_index": read_index,
        "variant_reference_catalog": reference_catalog_path,
        "variant_generation_manifest": generation_manifest_path,
    }.items():
        register_sidecar(manifest, key, path)
    return {
        "task_store": output_dir / VARIANT_TASK_STORE,
        "task_catalog": task_catalog_path,
        "obs": obs_path,
        "read_index": read_index,
        "reference_catalog": reference_catalog_path,
        "generation_manifest": generation_manifest_path,
        "manifest": manifest,
    }


def query_partitioned_variant_evidence(
    output_dir: str | Path,
    *,
    variant_reference_set_ids: Iterable[str] | None = None,
    molecule_uids: Iterable[str] | None = None,
    experiment_uids: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Query selected reference sets/molecules without opening unrelated task stores."""
    output_dir = Path(output_dir)
    index_root = output_dir / VARIANT_READ_INDEX
    selected_sets = (
        None
        if variant_reference_set_ids is None
        else {str(value) for value in variant_reference_set_ids}
    )
    selected_molecules = None if molecule_uids is None else {str(value) for value in molecule_uids}
    selected_experiments = (
        None if experiment_uids is None else {str(value) for value in experiment_uids}
    )
    set_dirs = sorted(index_root.glob("reference_set_id=*"))
    if selected_sets is not None:
        set_dirs = [
            path
            for path in set_dirs
            if path.name.removeprefix("reference_set_id=") in selected_sets
        ]
    index_paths: list[Path] = []
    for set_dir in set_dirs:
        if selected_molecules is None:
            index_paths.extend(sorted(set_dir.glob("molecule_bucket=*/part-*.parquet")))
        else:
            buckets = sorted({"b" + value[:2] for value in selected_molecules})
            for bucket in buckets:
                index_paths.extend(
                    sorted((set_dir / f"molecule_bucket={bucket}").glob("part-*.parquet"))
                )
    if not index_paths:
        return {"obs": pd.DataFrame(), "calls": pd.DataFrame(), "events": pd.DataFrame()}
    index = pd.concat([pd.read_parquet(path) for path in index_paths], ignore_index=True)
    mask = pd.Series(True, index=index.index)
    if selected_molecules is not None:
        mask &= index[MOLECULE_UID_COLUMN].astype(str).isin(selected_molecules)
    if selected_experiments is not None:
        mask &= index[EXPERIMENT_UID_COLUMN].astype(str).isin(selected_experiments)
    index = index.loc[mask]
    if index.empty:
        return {"obs": pd.DataFrame(), "calls": pd.DataFrame(), "events": pd.DataFrame()}

    result: dict[str, pd.DataFrame] = {}
    for kind, path_column in (
        ("obs", "obs_path"),
        ("calls", "calls_path"),
        ("events", "events_path"),
    ):
        frames = []
        for relative_path in sorted(index[path_column].dropna().astype(str).unique()):
            frame = pd.read_parquet(output_dir / relative_path)
            frame_mask = (
                frame[MOLECULE_UID_COLUMN]
                .astype(str)
                .isin(set(index[MOLECULE_UID_COLUMN].astype(str)))
            )
            if selected_experiments is not None:
                frame_mask &= frame[EXPERIMENT_UID_COLUMN].astype(str).isin(selected_experiments)
            frames.append(frame.loc[frame_mask])
        result[kind] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result
