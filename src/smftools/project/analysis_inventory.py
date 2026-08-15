"""Read-only inventory for project-owned analysis caches.

The inventory inspects cache definitions, selectors, manifests, and file metadata.
It deliberately does not load result tables, NumPy arrays, or persisted estimator
pickles. This makes upgrade-impact discovery safe to run before deciding whether
to recompute an analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..constants import SEMANTIC_GRAPH_DEFINITION_VERSION
from ..informatics.generation import (
    CURRENT_FILENAME,
    CURRENT_SCHEMA_VERSION,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
)
from . import embedding_store, sample_analysis
from .sample_store import _per_sample_root
from .set_store import sets_root

ANALYSIS_INVENTORY_SCHEMA_VERSION = 1

CURRENT_STATUS = "current"
STALE_STATUS = "stale"
INVALID_STATUS = "invalid"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if path.is_symlink():
        return None, "symlink_json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing_json"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None, "unreadable_json"
    if not isinstance(payload, dict):
        return None, "json_is_not_an_object"
    return payload, None


def _directory_metadata(root: Path) -> tuple[int | None, int | None, str | None]:
    """Return ``(size_bytes, file_count, error)`` without following symlinks."""
    if root.is_symlink():
        return None, None, "cache_symlink"
    size_bytes = 0
    file_count = 0
    try:
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            size_bytes += path.stat().st_size
            file_count += 1
    except OSError:
        return None, None, "metadata_unreadable"
    return size_bytes, file_count, None


def _identity_reasons(
    definition: dict[str, Any] | None,
    *,
    current_algorithm_version: str,
) -> list[str]:
    if definition is None:
        return []

    reasons = []
    stored_algorithm = definition.get("algorithm_version")
    if stored_algorithm is None:
        reasons.append("missing_algorithm_version")
    elif str(stored_algorithm) != str(current_algorithm_version):
        reasons.append("algorithm_version_mismatch")

    stored_graph = definition.get("graph_definition_version")
    if stored_graph is None:
        reasons.append("missing_graph_definition_version")
    elif stored_graph != SEMANTIC_GRAPH_DEFINITION_VERSION:
        reasons.append("graph_definition_version_mismatch")
    return reasons


def _status(*, invalid_reasons: list[str], stale_reasons: list[str]) -> str:
    if invalid_reasons:
        return INVALID_STATUS
    if stale_reasons:
        return STALE_STATUS
    return CURRENT_STATUS


def _base_record(
    project_dir: Path,
    cache_dir: Path,
    *,
    analysis: str,
    definition: dict[str, Any] | None,
    current_algorithm_version: str,
    invalid_reasons: list[str],
    stale_reasons: list[str],
) -> dict[str, Any]:
    size_bytes, file_count, metadata_error = _directory_metadata(cache_dir)
    if metadata_error is not None:
        invalid_reasons.append(metadata_error)
    invalid_reasons = sorted(set(invalid_reasons))
    stale_reasons = sorted(set(stale_reasons))
    return {
        "analysis": analysis,
        "status": _status(invalid_reasons=invalid_reasons, stale_reasons=stale_reasons),
        "reasons": invalid_reasons or stale_reasons,
        "cache_path": cache_dir.relative_to(project_dir).as_posix(),
        "definition_hash": cache_dir.name,
        "stored_algorithm_version": (
            None if definition is None else definition.get("algorithm_version")
        ),
        "current_algorithm_version": str(current_algorithm_version),
        "stored_graph_definition_version": (
            None if definition is None else definition.get("graph_definition_version")
        ),
        "current_graph_definition_version": SEMANTIC_GRAPH_DEFINITION_VERSION,
        "size_bytes": size_bytes,
        "file_count": file_count,
    }


def _periodicity_records(project_dir: Path) -> list[dict[str, Any]]:
    root = _per_sample_root(project_dir)
    if not root.is_dir():
        return []

    records = []
    for cache_dir in sorted(root.glob("*/*/*/analyses/*/*")):
        if not cache_dir.is_dir():
            continue
        relative = cache_dir.relative_to(root)
        experiment_id, reference_strand, sample, _, analysis, _ = relative.parts
        definition_path = cache_dir / sample_analysis.DEFINITION_FILENAME
        definition, definition_error = _read_json(definition_path)
        invalid_reasons = []
        if definition_error is not None:
            invalid_reasons.append(f"definition_{definition_error}")
        result_path = cache_dir / sample_analysis.RESULT_FILENAME
        if result_path.is_symlink() or not result_path.is_file():
            invalid_reasons.append("missing_result")
        if analysis != sample_analysis.PERIODICITY_ANALYSIS_NAME:
            invalid_reasons.append("unsupported_analysis")

        if definition is not None:
            hash_definition = {
                key: value for key, value in definition.items() if key != "cache_schema_version"
            }
            if sample_analysis._definition_hash(hash_definition) != cache_dir.name:
                invalid_reasons.append("definition_hash_mismatch")
            if definition.get("analysis") != analysis:
                invalid_reasons.append("analysis_name_mismatch")

        stale_reasons = _identity_reasons(
            definition,
            current_algorithm_version=sample_analysis.PERIODICITY_ALGORITHM_VERSION,
        )
        record = _base_record(
            project_dir,
            cache_dir,
            analysis=analysis,
            definition=definition,
            current_algorithm_version=sample_analysis.PERIODICITY_ALGORITHM_VERSION,
            invalid_reasons=invalid_reasons,
            stale_reasons=stale_reasons,
        )
        record.update(
            {
                "scope": "partition",
                "experiment_id": experiment_id,
                "reference_strand": reference_strand,
                "sample": sample,
                "set_label": None,
                "generation_count": None,
                "current_generation_id": None,
            }
        )
        records.append(record)
    return records


def _embedding_manifest(
    cache_dir: Path,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    invalid_reasons = []
    pointer, pointer_error = _read_json(cache_dir / CURRENT_FILENAME)
    if pointer_error is not None:
        invalid_reasons.append(f"current_{pointer_error}")
        return None, None, invalid_reasons
    try:
        pointer_schema = int(pointer.get("schema_version", -1))
    except (TypeError, ValueError):
        pointer_schema = -1
    if pointer_schema != CURRENT_SCHEMA_VERSION:
        invalid_reasons.append("current_schema_incompatible")

    current_generation_id = str(pointer.get("generation_id", "")).strip()
    raw_generation_path = str(pointer.get("generation_path", "")).strip()
    generation_path = Path(raw_generation_path)
    if (
        not current_generation_id
        or not raw_generation_path
        or generation_path.is_absolute()
        or ".." in generation_path.parts
        or len(Path(current_generation_id).parts) != 1
        or generation_path != Path(GENERATIONS_SUBDIR, current_generation_id)
    ):
        invalid_reasons.append("current_pointer_not_portable")
        return None, current_generation_id or None, invalid_reasons
    try:
        resolved_root = cache_dir.resolve()
        resolved_generation = (cache_dir / generation_path).resolve()
    except (OSError, RuntimeError):
        invalid_reasons.append("current_pointer_unresolvable")
        return None, current_generation_id, invalid_reasons
    if resolved_generation == resolved_root or not resolved_generation.is_relative_to(
        resolved_root
    ):
        invalid_reasons.append("current_pointer_not_portable")
        return None, current_generation_id, invalid_reasons

    manifest, manifest_error = _read_json(resolved_generation / GENERATION_MANIFEST)
    if manifest_error is not None:
        invalid_reasons.append(f"manifest_{manifest_error}")
        return None, current_generation_id, invalid_reasons
    if str(manifest.get("generation_id", "")) != current_generation_id:
        invalid_reasons.append("generation_id_mismatch")
    try:
        manifest_schema = int(manifest.get("schema_version", -1))
    except (TypeError, ValueError):
        manifest_schema = -1
    if manifest_schema != embedding_store.GENERATION_SCHEMA_VERSION:
        invalid_reasons.append("generation_schema_incompatible")
    if manifest.get("status") != "complete":
        invalid_reasons.append("generation_incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        invalid_reasons.append("artifact_manifest_missing")
    else:
        missing_manifest_entries = set(embedding_store._ARTIFACT_FILENAMES) - set(artifacts)
        if missing_manifest_entries:
            invalid_reasons.append("artifact_manifest_incomplete")
        for filename in embedding_store._ARTIFACT_FILENAMES:
            artifact_path = Path(str(filename))
            resolved_artifact = resolved_generation / artifact_path
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                invalid_reasons.append("artifact_path_not_portable")
                break
            if filename not in artifacts:
                continue
            if resolved_artifact.is_symlink() or not resolved_artifact.is_file():
                invalid_reasons.append("artifact_missing")
                break
    return manifest, current_generation_id, invalid_reasons


def _embedding_records(project_dir: Path) -> list[dict[str, Any]]:
    root = sets_root(project_dir)
    if not root.is_dir():
        return []

    records = []
    for cache_dir in sorted(root.glob("*/embeddings/*")):
        if not cache_dir.is_dir():
            continue
        set_label = cache_dir.relative_to(root).parts[0]
        manifest, current_generation_id, invalid_reasons = _embedding_manifest(cache_dir)
        definition = manifest.get("definition") if manifest is not None else None
        if definition is not None and not isinstance(definition, dict):
            invalid_reasons.append("definition_json_is_not_an_object")
            definition = None
        if definition is None:
            invalid_reasons.append("definition_missing")
        else:
            definition_hash = embedding_store._definition_hash(definition)
            if (
                definition_hash != cache_dir.name
                or manifest.get("definition_hash") != cache_dir.name
            ):
                invalid_reasons.append("definition_hash_mismatch")

        generations_dir = cache_dir / GENERATIONS_SUBDIR
        generation_count = (
            sum(path.is_dir() for path in generations_dir.iterdir())
            if generations_dir.is_dir()
            else 0
        )
        stale_reasons = _identity_reasons(
            definition,
            current_algorithm_version=embedding_store.EMBEDDING_ALGORITHM_VERSION,
        )
        record = _base_record(
            project_dir,
            cache_dir,
            analysis="embedding",
            definition=definition,
            current_algorithm_version=embedding_store.EMBEDDING_ALGORITHM_VERSION,
            invalid_reasons=invalid_reasons,
            stale_reasons=stale_reasons,
        )
        record.update(
            {
                "scope": "set",
                "experiment_id": None,
                "reference_strand": None,
                "sample": None,
                "set_label": set_label,
                "generation_count": generation_count,
                "current_generation_id": current_generation_id,
            }
        )
        records.append(record)
    return records


def analysis_cache_inventory(
    project_dir: str | Path,
    *,
    stale_only: bool = False,
) -> dict[str, Any]:
    """Inventory project analysis caches without loading analysis artifacts.

    Args:
        project_dir: Project root containing ``project_outputs``.
        stale_only: Keep only stale or invalid entries requiring attention.

    Returns:
        A versioned JSON-compatible inventory ordered by cache path.
    """
    project_dir = Path(project_dir)
    records = sorted(
        [*_periodicity_records(project_dir), *_embedding_records(project_dir)],
        key=lambda record: record["cache_path"],
    )
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in (CURRENT_STATUS, STALE_STATUS, INVALID_STATUS)
    }
    visible = (
        [record for record in records if record["status"] != CURRENT_STATUS]
        if stale_only
        else records
    )
    return {
        "schema_version": ANALYSIS_INVENTORY_SCHEMA_VERSION,
        "stale_only": bool(stale_only),
        "counts": counts,
        "entries": visible,
    }
