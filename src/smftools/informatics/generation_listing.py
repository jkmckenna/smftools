"""Read-only enumeration of published immutable generations.

Four subsystems publish immutable generations today, each with its own
publisher/validator pair:

- raw (:mod:`smftools.informatics.raw_generation`)
- preprocess (:mod:`smftools.preprocessing.preprocess_generation`)
- latent (:mod:`smftools.tools.partitioned_latent`)
- project embeddings (:mod:`smftools.project.embedding_store`)

They converged independently on the same on-disk vocabulary -- a
``generations/<generation_id>/generation_manifest.json`` directory beside an
atomically-published ``current.json`` selecting the one generation readers may
consume -- and differ only in manifest payload and in whether the pointer
carries ``manifest_sha256``. This module reads that shared vocabulary.

It deliberately does **not** call the per-kind ``resolve_current_*`` validators.
Those raise on the first defect, which is correct for a reader about to consume
a generation and wrong for an inventory tool: listing is what you reach for when
something is *already* broken, so every defect is reported as a record field
rather than an exception. Nothing here writes, and no generation is opened
beyond its manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..constants import (
    BASECALL_DIR,
    CHIMERIC_DIR,
    HMM_DIR,
    LATENT_DIR,
    PREPROCESS_DIR,
    RAW_DIR,
    SPATIAL_DIR,
    VARIANT_DIR,
)
from .generation_retention import read_generation_retention_lenient

GENERATIONS_SUBDIR = "generations"
CURRENT_FILENAME = "current.json"
GENERATION_MANIFEST = "generation_manifest.json"
STAGING_SUBDIR = ".staging"

#: Stage directories that may own a ``generations/`` tree, in pipeline order.
#: Stages without a generation model yet (spatial, hmm, chimeric) are included
#: deliberately: they are scanned, found to have none, and simply contribute no
#: records -- so this table does not need editing when they gain one.
STAGE_GENERATION_DIRS: dict[str, str] = {
    "basecall": BASECALL_DIR,
    "raw": RAW_DIR,
    "preprocess": PREPROCESS_DIR,
    "variant": VARIANT_DIR,
    "spatial": SPATIAL_DIR,
    "hmm": HMM_DIR,
    "latent": LATENT_DIR,
    "chimeric": CHIMERIC_DIR,
}

_TIMESTAMP_KEYS = ("created_at", "completed_at", "published_at", "fitted_at")

STATE_OK = "ok"
STATE_UNREADABLE = "unreadable"
STATE_MISSING = "missing"


@dataclass(frozen=True)
class GenerationRecord:
    """One published generation, or one defect found where a generation should be."""

    scope: str
    """``"experiment"`` or ``"project"``."""

    kind: str
    """Stage name (``raw``, ``preprocess``, ``latent``, ...) or ``embedding``."""

    container: str
    """POSIX path, relative to the scanned root, of the directory owning ``generations/``."""

    generation_id: str
    path: str
    """POSIX path to the generation directory, relative to the scanned root."""

    is_current: bool
    state: str
    """:data:`STATE_OK`, :data:`STATE_UNREADABLE`, or :data:`STATE_MISSING`."""

    status: str | None
    config_hash: str | None
    manifest_schema_version: int | None
    created_at: str | None
    """Timestamp from the manifest when it records one; ``None`` otherwise."""

    modified_at: str
    """Directory mtime, ISO-8601 UTC. Always available, unlike ``created_at``."""

    artifact_count: int | None
    size_bytes: int | None
    pinned: bool
    retention_reasons: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retention_reasons"] = list(self.retention_reasons)
        payload["issues"] = list(self.issues)
        return payload


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(payload, issue)``; never raises for a missing or malformed file."""
    if not path.is_file():
        return None, f"{path.name} is missing"
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name} is unreadable: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, f"{path.name} is not a JSON object"
    return payload, None


def _iso_mtime(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()


def _directory_size(path: Path) -> int | None:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
    except OSError:
        return None
    return total


def _manifest_timestamp(manifest: dict[str, Any]) -> str | None:
    for key in _TIMESTAMP_KEYS:
        value = manifest.get(key)
        if value:
            return str(value)
    return None


def _artifact_count(manifest: dict[str, Any]) -> int | None:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, (dict, list, tuple)):
        return len(artifacts)
    return None


def _current_generation_id(container: Path) -> tuple[str | None, str | None]:
    """Read ``current.json`` leniently. Returns ``(generation_id, issue)``."""
    pointer_path = container / CURRENT_FILENAME
    if not pointer_path.is_file():
        return None, None
    pointer, issue = _read_json(pointer_path)
    if pointer is None:
        return None, issue
    generation_id = pointer.get("generation_id")
    if not generation_id:
        return None, "current.json names no generation_id"
    return str(generation_id), None


def _records_for_container(
    container: Path,
    *,
    root: Path,
    scope: str,
    kind: str,
    include_size: bool,
) -> list[GenerationRecord]:
    generations_dir = container / GENERATIONS_SUBDIR
    if not generations_dir.is_dir():
        return []

    container_rel = container.relative_to(root).as_posix()
    current_id, pointer_issue = _current_generation_id(container)
    retention, retention_issue = read_generation_retention_lenient(container)

    records: list[GenerationRecord] = []
    seen: set[str] = set()

    for generation_dir in sorted(generations_dir.iterdir()):
        if not generation_dir.is_dir() or generation_dir.name == STAGING_SUBDIR:
            continue
        seen.add(generation_dir.name)
        manifest, manifest_issue = _read_json(generation_dir / GENERATION_MANIFEST)
        issues: list[str] = []
        if pointer_issue:
            issues.append(pointer_issue)
        if retention_issue:
            issues.append(retention_issue)
        if manifest_issue:
            issues.append(manifest_issue)
        retention_entry = retention.get(generation_dir.name)
        pinned = retention_entry is not None and retention_entry.pinned
        retention_reasons = (
            tuple(reason.reason for reason in retention_entry.reasons)
            if retention_entry is not None
            else ()
        )

        if manifest is None:
            records.append(
                GenerationRecord(
                    scope=scope,
                    kind=kind,
                    container=container_rel,
                    generation_id=generation_dir.name,
                    path=generation_dir.relative_to(root).as_posix(),
                    is_current=generation_dir.name == current_id,
                    state=STATE_UNREADABLE,
                    status=None,
                    config_hash=None,
                    manifest_schema_version=None,
                    created_at=None,
                    modified_at=_iso_mtime(generation_dir),
                    artifact_count=None,
                    size_bytes=_directory_size(generation_dir) if include_size else None,
                    pinned=pinned,
                    retention_reasons=retention_reasons,
                    issues=tuple(issues),
                )
            )
            continue

        declared_id = str(manifest.get("generation_id", "") or "")
        if declared_id and declared_id != generation_dir.name:
            issues.append(f"manifest generation_id {declared_id!r} does not match directory name")
        schema_version = manifest.get("schema_version")
        records.append(
            GenerationRecord(
                scope=scope,
                kind=kind,
                container=container_rel,
                generation_id=declared_id or generation_dir.name,
                path=generation_dir.relative_to(root).as_posix(),
                is_current=generation_dir.name == current_id,
                state=STATE_OK,
                status=str(manifest.get("status")) if manifest.get("status") else None,
                config_hash=(
                    str(manifest.get("config_hash")) if manifest.get("config_hash") else None
                ),
                manifest_schema_version=(
                    int(schema_version) if isinstance(schema_version, int) else None
                ),
                created_at=_manifest_timestamp(manifest),
                modified_at=_iso_mtime(generation_dir),
                artifact_count=_artifact_count(manifest),
                size_bytes=_directory_size(generation_dir) if include_size else None,
                pinned=pinned,
                retention_reasons=retention_reasons,
                issues=tuple(issues),
            )
        )

    missing_ids = set(retention).difference(seen)
    if current_id and current_id not in seen:
        missing_ids.add(current_id)
    for missing_id in sorted(missing_ids):
        # A dangling pointer is the most dangerous state here: readers resolve
        # `current` and fail, while the inventory would otherwise look empty.
        retention_entry = retention.get(missing_id)
        issues = []
        if missing_id == current_id:
            issues.append("current.json points at a generation directory that does not exist")
        if retention_entry is not None:
            issues.append("retention.json pins a generation directory that does not exist")
        if retention_issue:
            issues.append(retention_issue)
        records.append(
            GenerationRecord(
                scope=scope,
                kind=kind,
                container=container_rel,
                generation_id=missing_id,
                path="",
                is_current=missing_id == current_id,
                state=STATE_MISSING,
                status=None,
                config_hash=None,
                manifest_schema_version=None,
                created_at=None,
                modified_at="",
                artifact_count=None,
                size_bytes=None,
                pinned=retention_entry is not None,
                retention_reasons=(
                    tuple(reason.reason for reason in retention_entry.reasons)
                    if retention_entry is not None
                    else ()
                ),
                issues=tuple(issues),
            )
        )

    return records


def _iter_experiment_containers(run_root: Path) -> Iterator[tuple[str, Path]]:
    for kind, stage_dir in STAGE_GENERATION_DIRS.items():
        container = run_root / stage_dir
        if container.is_dir():
            yield kind, container


def list_experiment_generations(
    run_root: str | Path,
    *,
    include_size: bool = False,
) -> list[GenerationRecord]:
    """Enumerate every generation published under one experiment output root.

    Args:
        run_root: An experiment's output directory (the one holding
            ``raw_outputs/``, ``preprocess_adata_outputs/``, ...).
        include_size: Walk each generation to total its bytes. Off by default
            because a large store makes it markedly slower, and the common use
            of this function is "what exists", not "how big".

    Returns:
        Records in pipeline-stage order. Stages with no generation model, and
        experiments predating the generation model entirely, contribute nothing
        rather than raising.
    """
    run_root = Path(run_root)
    records: list[GenerationRecord] = []
    for kind, container in _iter_experiment_containers(run_root):
        records.extend(
            _records_for_container(
                container,
                root=run_root,
                scope="experiment",
                kind=kind,
                include_size=include_size,
            )
        )
    return records


def list_project_generations(
    project_dir: str | Path,
    *,
    include_size: bool = False,
) -> list[GenerationRecord]:
    """Enumerate project-owned generations (currently: embeddings).

    Experiment generations are not included; resolve registered experiments and
    call :func:`list_experiment_generations` per experiment for those, so the
    caller controls how unreachable experiment paths are reported.
    """
    project_dir = Path(project_dir)
    from ..project.set_store import sets_root

    records: list[GenerationRecord] = []
    root = sets_root(project_dir)
    if not root.is_dir():
        return records
    for label_dir in sorted(root.iterdir()):
        embeddings_dir = label_dir / "embeddings"
        if not embeddings_dir.is_dir():
            continue
        for definition_dir in sorted(embeddings_dir.iterdir()):
            if not definition_dir.is_dir():
                continue
            records.extend(
                _records_for_container(
                    definition_dir,
                    root=project_dir,
                    scope="project",
                    kind="embedding",
                    include_size=include_size,
                )
            )
    return records
