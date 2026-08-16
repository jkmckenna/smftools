"""Stage, publish, and validate immutable re-basecalling processing lineages.

A lineage is the publication unit: a complete, immutable descendant of an
experiment's earlier artifacts. It is not a second biological experiment and not
an in-place replacement raw generation.

Per `D1` in the generation-lifecycle plan, a lineage is a map ``stage ->
generation id``. Descendant stage generations are published beside the parent's,
in the experiment's ordinary stage directories, **without** advancing
``current.json``; the lineage records which ids belong to it. Selection stays
the project registry's ``active_lineage``, changed only by explicit promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..readwrite import atomic_write_json
from .rebasecall_basecall import PublishedRebasecallBasecall
from .rebasecall_plan import RebasecallPlan
from .rebasecall_selection import FrozenRebasecallSelection

REBASECALL_LINEAGE_SCHEMA_VERSION = 1
LINEAGE_MANIFEST_FILENAME = "lineage_manifest.json"
LINEAGE_REQUEST_FILENAME = "request.json"
LINEAGE_STAGE_GENERATIONS_FILENAME = "stage_generations.json"
LINEAGE_VALIDATION_FILENAME = "validation.json"
LINEAGES_SUBDIR = "lineages"
LINEAGE_STAGING_SUBDIR = ".staging"
LINEAGE_REQUESTS_SUBDIR = "requests"

LINEAGE_STAGES = ("raw", "preprocess", "spatial", "hmm", "latent")


class RebasecallLineageError(RuntimeError):
    """Raised when a lineage cannot be staged, published, or validated safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class PublishedRebasecallLineage:
    """One validated immutable processing lineage."""

    lineage_id: str
    directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]

    @property
    def stage_generations(self) -> dict[str, str]:
        """Return the ``stage -> generation id`` map this lineage publishes."""
        return dict(self.manifest["stage_generations"])

    @property
    def basecall_id(self) -> str:
        return str(self.manifest["basecall_id"])


@dataclass
class StagedRebasecallLineage:
    """A lineage being built under its staging root."""

    lineage_id: str
    staging_dir: Path
    final_dir: Path
    _stage_generations: dict[str, str] = field(default_factory=dict)

    def record_stage_generation(self, stage: str, generation_id: str) -> None:
        """Record one descendant stage generation as belonging to this lineage."""
        if stage not in LINEAGE_STAGES:
            raise RebasecallLineageError(
                "lineage_stage_unsupported",
                f"stage {stage!r} is not a lineage stage",
            )
        normalized = str(generation_id).strip()
        if not normalized:
            raise RebasecallLineageError(
                "lineage_stage_generation_invalid",
                f"stage {stage!r} recorded an empty generation id",
            )
        existing = self._stage_generations.get(stage)
        if existing is not None and existing != normalized:
            raise RebasecallLineageError(
                "lineage_stage_generation_conflict",
                f"stage {stage!r} already recorded generation {existing!r}",
            )
        self._stage_generations[stage] = normalized


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    normalized = str(value)
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def build_lineage_identity(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    basecall: PublishedRebasecallBasecall,
) -> dict[str, Any]:
    """Return the path-neutral identity of one processing lineage.

    This is the identity the design contract names: parent experiment and
    generations, the frozen selection, the source-signal resolution, the
    resolved basecall, and the requested terminal target.
    """
    if plan.raw_parent is None:
        raise RebasecallLineageError(
            "lineage_parent_unavailable",
            "a lineage requires a resolved immutable raw parent",
        )
    source_resolution = plan._source_resolution
    return {
        "schema_version": REBASECALL_LINEAGE_SCHEMA_VERSION,
        "origin_experiment_uid": plan.experiment_uid,
        "parent_raw_generation_id": plan.raw_parent.generation_id,
        "parent_preprocess_generation_id": (
            None if plan.preprocess_parent is None else plan.preprocess_parent.generation_id
        ),
        "selection_id": selection.selection_id,
        "source_resolution_digest": (
            None if source_resolution is None else source_resolution.digest
        ),
        "basecall_id": basecall.basecall_id,
        "request_id": plan.request.request_id,
        "downstream_target": plan.request.downstream_target,
    }


def descendant_raw_provenance(
    lineage_id: str,
    identity: Mapping[str, Any],
    basecall: PublishedRebasecallBasecall,
    *,
    identity_map: str | None = None,
) -> dict[str, Any]:
    """Build the lineage block a descendant raw generation records.

    Per `D2`, ``generation_kind`` is read from the basecall generation rather
    than restated here, so a descendant cannot disagree with the artifact whose
    contents the selection actually determined.
    """
    return {
        "lineage_id": str(lineage_id),
        "origin_experiment_uid": identity["origin_experiment_uid"],
        "parent_raw_generation_id": identity["parent_raw_generation_id"],
        "parent_preprocess_generation_id": identity["parent_preprocess_generation_id"],
        "selection_id": identity["selection_id"],
        "source_resolution_digest": identity["source_resolution_digest"],
        "basecall_id": basecall.basecall_id,
        "generation_kind": basecall.generation_kind,
        "identity_map": identity_map,
    }


def _validate_manifest_shape(manifest: Any, expected_lineage_id: str | None) -> str:
    required_keys = {
        "schema_version",
        "lineage_id",
        "status",
        "accepted_plan_id",
        "request_id",
        "experiment_id",
        "basecall_id",
        "identity",
        "stage_generations",
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != REBASECALL_LINEAGE_SCHEMA_VERSION
        or set(manifest) != required_keys
    ):
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage manifest does not match schema 1",
        )
    if manifest.get("status") != "complete":
        raise RebasecallLineageError(
            "lineage_incomplete",
            "published lineage is not complete",
        )
    lineage_id = str(manifest["lineage_id"])
    if not _is_sha256(lineage_id) or (
        expected_lineage_id is not None and lineage_id != expected_lineage_id
    ):
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage identity does not match the expected lineage",
        )
    identity = manifest["identity"]
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version") != REBASECALL_LINEAGE_SCHEMA_VERSION
        or _sha256_payload(identity) != lineage_id
        or identity.get("basecall_id") != manifest.get("basecall_id")
        or identity.get("request_id") != manifest.get("request_id")
    ):
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage semantic identity is inconsistent",
        )
    stage_generations = manifest["stage_generations"]
    if not isinstance(stage_generations, dict) or not stage_generations:
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage records no stage generations",
        )
    if "raw" not in stage_generations:
        raise RebasecallLineageError(
            "lineage_incomplete",
            "published lineage records no descendant raw generation",
        )
    for stage, generation_id in stage_generations.items():
        if stage not in LINEAGE_STAGES or not str(generation_id).strip():
            raise RebasecallLineageError(
                "lineage_artifact_invalid",
                f"published lineage stage entry {stage!r} is invalid",
            )
    return lineage_id


def read_published_rebasecall_lineage(
    directory: str | Path,
    *,
    expected_lineage_id: str | None = None,
) -> PublishedRebasecallLineage:
    """Read and revalidate one published lineage without mutating it."""
    directory = Path(directory)
    manifest_path = directory / LINEAGE_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage manifest is missing or unreadable",
        ) from exc
    lineage_id = _validate_manifest_shape(manifest, expected_lineage_id)
    for filename in (LINEAGE_REQUEST_FILENAME, LINEAGE_STAGE_GENERATIONS_FILENAME):
        if not (directory / filename).is_file():
            raise RebasecallLineageError(
                "lineage_artifact_invalid",
                f"published lineage is missing {filename}",
            )
    try:
        recorded = json.loads(
            (directory / LINEAGE_STAGE_GENERATIONS_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage stage-generation map is unreadable",
        ) from exc
    if recorded != manifest["stage_generations"]:
        raise RebasecallLineageError(
            "lineage_artifact_invalid",
            "published lineage stage-generation map disagrees with its manifest",
        )
    return PublishedRebasecallLineage(
        lineage_id=lineage_id,
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def list_published_rebasecall_lineages(
    rebasecall_root: str | Path,
) -> tuple[PublishedRebasecallLineage, ...]:
    """Return every complete published lineage, newest identity order aside."""
    lineages_dir = Path(rebasecall_root) / LINEAGES_SUBDIR
    if not lineages_dir.is_dir():
        return ()
    published = []
    for directory in sorted(path for path in lineages_dir.iterdir() if path.is_dir()):
        published.append(read_published_rebasecall_lineage(directory))
    return tuple(published)


@contextmanager
def staged_lineage(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    basecall: PublishedRebasecallBasecall,
    rebasecall_root: str | Path,
    *,
    accepted_plan_id: str,
) -> Iterator[StagedRebasecallLineage]:
    """Stage a lineage and publish it atomically, or leave nothing behind.

    The staging tree is removed on any failure, so a killed basecall, stage,
    validation, or publish leaves the parent run and every prior complete
    lineage unchanged and discoverable. Descendant stage generations published
    inside the block are addressable on their own; the lineage is what makes
    them a coherent set.
    """
    if accepted_plan_id != plan.plan_id:
        raise RebasecallLineageError(
            "accepted_plan_mismatch",
            "the supplied accepted plan ID does not match the current plan",
        )
    if plan.status != "ready":
        raise RebasecallLineageError(
            "accepted_plan_blocked",
            "a blocked re-basecall plan cannot publish a lineage",
        )
    if basecall.manifest.get("selection_id") != selection.selection_id:
        raise RebasecallLineageError(
            "lineage_basecall_mismatch",
            "the published basecall was produced from a different frozen selection",
        )
    identity = build_lineage_identity(plan, selection, basecall)
    lineage_id = _sha256_payload(identity)
    rebasecall_root = Path(rebasecall_root)
    destination = rebasecall_root / LINEAGES_SUBDIR / lineage_id
    if destination.exists():
        raise RebasecallLineageError(
            "lineage_already_published",
            f"lineage {lineage_id} is already published",
        )

    staging_parent = rebasecall_root / LINEAGE_STAGING_SUBDIR
    staging_parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{lineage_id}.", suffix=".tmp", dir=staging_parent))
    staged = StagedRebasecallLineage(
        lineage_id=lineage_id,
        staging_dir=staging_dir,
        final_dir=destination,
    )
    try:
        yield staged
        if "raw" not in staged._stage_generations:
            raise RebasecallLineageError(
                "lineage_incomplete",
                "a lineage must record a descendant raw generation before publication",
            )
        stage_generations = {
            stage: staged._stage_generations[stage]
            for stage in LINEAGE_STAGES
            if stage in staged._stage_generations
        }
        manifest = {
            "schema_version": REBASECALL_LINEAGE_SCHEMA_VERSION,
            "lineage_id": lineage_id,
            "status": "complete",
            "accepted_plan_id": plan.plan_id,
            "request_id": plan.request.request_id,
            "experiment_id": plan.experiment_id,
            "basecall_id": basecall.basecall_id,
            "identity": dict(identity),
            "stage_generations": stage_generations,
        }
        atomic_write_json(staging_dir / LINEAGE_REQUEST_FILENAME, plan.request.to_dict())
        atomic_write_json(staging_dir / LINEAGE_STAGE_GENERATIONS_FILENAME, stage_generations)
        atomic_write_json(staging_dir / LINEAGE_MANIFEST_FILENAME, manifest)
        read_published_rebasecall_lineage(staging_dir, expected_lineage_id=lineage_id)
        os.replace(staging_dir, destination)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    read_published_rebasecall_lineage(destination, expected_lineage_id=lineage_id)


def write_lineage_validation(
    lineage: PublishedRebasecallLineage,
    report: Mapping[str, Any],
) -> Path:
    """Record a validation report beside a published lineage.

    The report is deliberately outside the manifest and outside lineage
    identity: revalidating an unchanged lineage must not change what it is.
    """
    path = lineage.directory / LINEAGE_VALIDATION_FILENAME
    atomic_write_json(path, dict(report))
    return path
