"""Immutable generation publication for raw-ingestion outputs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..readwrite import (
    atomic_write_json,
    normalize_uns_string_lists,
    safe_read_h5ad,
    safe_write_h5ad,
)
from .experiment_manifest import artifact_record
from .generation import (
    CURRENT_FILENAME,
    CURRENT_SCHEMA_VERSION,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    STAGING_SUBDIR,
    GenerationError,
    resolve_current_generation,
    staged_generation,
)
from .partition_read import relative_uns_path, resolve_relative_path
from .sidecar_manifest import register_sidecar, resolve_sidecar, sidecar_manifest_path

RAW_GENERATIONS_SUBDIR = GENERATIONS_SUBDIR
RAW_STAGING_SUBDIR = STAGING_SUBDIR
RAW_CURRENT_FILENAME = CURRENT_FILENAME
RAW_GENERATION_MANIFEST = GENERATION_MANIFEST
RAW_GENERATION_SCHEMA_VERSION = 3
RAW_CURRENT_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION

# Schema 3 adds the optional ``lineage`` block marking a generation as a
# re-basecalled descendant. Its absence is meaningful: an ordinary generation
# has no lineage provenance, and a reader must not invent one. Per `D2`,
# ``generation_kind`` is *derived* from the basecall generation rather than
# independently asserted here.
RAW_LINEAGE_PROVENANCE_KEYS = frozenset(
    {
        "lineage_id",
        "origin_experiment_uid",
        "parent_raw_generation_id",
        "parent_preprocess_generation_id",
        "selection_id",
        "source_resolution_digest",
        "basecall_id",
        "generation_kind",
        "identity_map",
    }
)
_LINEAGE_REQUIRED_TEXT_KEYS = (
    "lineage_id",
    "origin_experiment_uid",
    "parent_raw_generation_id",
    "selection_id",
    "basecall_id",
    "generation_kind",
)

RAW_GENERATION_ARTIFACT_PATHS: dict[str, str] = {
    "spine": "spine.h5ad",
    "ragged_store": "raw",
    "interval_catalog": "interval_catalog.parquet",
    "obs": "obs.parquet",
    "molecules": "molecules.parquet",
    "molecule_index": "molecule_index",
    "segments": "segments.parquet",
    "segment_index": "segment_index",
    "reference_interval_map": "reference_interval_map.parquet",
    "sidecar_manifest": "sidecar_manifest.json",
    "input_manifest_csv": "input_manifest/resolved_input_manifest.csv",
    "input_manifest_json": "input_manifest/resolved_input_manifest.json",
    "input_resolution_report": "input_manifest/input_resolution_report.json",
}
RAW_REQUIRED_ARTIFACTS = tuple(RAW_GENERATION_ARTIFACT_PATHS)
_RAW_GENERATION_V1_ARTIFACT_PATHS = {
    key: value
    for key, value in RAW_GENERATION_ARTIFACT_PATHS.items()
    if key not in {"segments", "segment_index"}
}
_NONEMPTY_DIRECTORIES = frozenset({"ragged_store", "molecule_index", "segment_index"})


class RawGenerationError(RuntimeError):
    """Raised when a raw generation cannot be published or selected safely."""


def _checksum(path: Path) -> str:
    return str(artifact_record(path, path.parent, checksum=True)["sha256"])


def _generation_artifact_record(path: Path, generation_root: Path) -> dict[str, Any]:
    record = artifact_record(path, generation_root, checksum=True)
    record["anchor"] = "generation_root"
    return record


def _resolve_generation_artifact(
    generation_root: Path,
    record: Mapping[str, Any],
) -> Path:
    raw_path = record.get("path")
    relative = Path(str(raw_path or ""))
    resolved = (generation_root / relative).resolve()
    if (
        record.get("path_kind") != "relative"
        or record.get("anchor") != "generation_root"
        or not raw_path
        or relative.is_absolute()
        or ".." in relative.parts
        or not resolved.is_relative_to(generation_root.resolve())
    ):
        raise RawGenerationError("raw generation artifact path is not portable")
    return resolved


def _copy_artifact(
    source: Path,
    destination: Path,
    *,
    reuse_source: Path | None = None,
    reuse_stats: dict[str, int] | None = None,
) -> None:
    """Copy an artifact, hardlinking checksum-identical immutable files."""
    stats = reuse_stats if reuse_stats is not None else {}

    def copy_file(source_file: Path, destination_file: Path, candidate: Path | None) -> None:
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        size = source_file.stat().st_size
        if (
            candidate is not None
            and candidate.is_file()
            and candidate.stat().st_size == size
            and _checksum(candidate) == _checksum(source_file)
        ):
            os.link(candidate, destination_file)
            stats["reused_files"] = stats.get("reused_files", 0) + 1
            stats["reused_bytes"] = stats.get("reused_bytes", 0) + size
        else:
            shutil.copy2(source_file, destination_file)
            stats["new_files"] = stats.get("new_files", 0) + 1
            stats["new_bytes"] = stats.get("new_bytes", 0) + size

    if source.is_dir():
        destination.mkdir(parents=True)
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            target = destination / relative
            candidate = reuse_source / relative if reuse_source is not None else None
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                copy_file(path, target, candidate)
            else:
                raise RawGenerationError(f"unsupported raw publication artifact: {path}")
    else:
        copy_file(source, destination, reuse_source)


def _artifact_file_totals(path: Path) -> tuple[int, int]:
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def raw_generation_dependencies(
    spine_path: str | Path,
    source_manifest: str | Path,
    *,
    run_root: str | Path,
    owned_artifacts: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Discover shared immutable BAM and annotation dependencies for a generation."""
    run_root = Path(run_root)
    owned = [Path(path).resolve() for path in owned_artifacts.values() if Path(path).exists()]

    def is_owned(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or resolved.is_relative_to(root) for root in owned)

    dependencies: dict[str, Path] = {}
    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    if "bam_path" in spine.obs:
        for index, value in enumerate(sorted(set(spine.obs["bam_path"].dropna().astype(str)))):
            path = resolve_relative_path(value, run_root)
            if path is not None and path.exists() and not is_owned(path):
                dependencies[f"aligned-bam:{index}"] = path

    manifest_path = Path(source_manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    sidecars = manifest.get("sidecars", {})
    if isinstance(sidecars, dict):
        for key in sorted(sidecars):
            path = resolve_sidecar(manifest_path, key)
            if path is not None and not is_owned(path):
                dependencies[f"sidecar:{key}"] = path
    return dependencies


def _bind_generation_spine(
    spine_path: Path,
    *,
    generation_id: str,
    publication_dir: Path,
    run_root: Path,
    region_artifacts: Mapping[str, str],
) -> None:
    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    spine.uns["molecules_catalog"] = relative_uns_path(
        publication_dir / "molecules.parquet", run_root
    )
    spine.uns["molecule_index"] = relative_uns_path(publication_dir / "molecule_index", run_root)
    spine.uns["segments_catalog"] = relative_uns_path(
        publication_dir / "segments.parquet", run_root
    )
    spine.uns["segment_index"] = relative_uns_path(publication_dir / "segment_index", run_root)
    spine.uns["reference_interval_map"] = relative_uns_path(
        publication_dir / "reference_interval_map.parquet", run_root
    )
    spine.uns["region_catalogs"] = {
        scope: relative_uns_path(publication_dir / relative, run_root)
        for scope, relative in sorted(region_artifacts.items())
    }
    spine.uns["raw_generation_id"] = generation_id
    # The spine was just read from disk, so its string-list uns entries are numpy
    # arrays that the writer's sanitizer would store as string representations.
    normalize_uns_string_lists(spine)
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)


def _write_generation_sidecar_manifest(
    generation_dir: Path,
    artifact_paths: Mapping[str, str],
) -> Path:
    manifest_path = sidecar_manifest_path(generation_dir)
    manifest_path.unlink(missing_ok=True)
    for key, relative in sorted(artifact_paths.items()):
        if key in {
            "sidecar_manifest",
            "input_manifest_csv",
            "input_manifest_json",
            "input_resolution_report",
        }:
            continue
        register_sidecar(manifest_path, key, generation_dir / relative)
    return manifest_path


def validate_raw_lineage_provenance(lineage: Any) -> dict[str, Any] | None:
    """Validate a descendant generation's lineage block, if it carries one.

    Returns ``None`` for an ordinary generation. A malformed block is an error
    rather than a warning: a descendant that cannot state which selection and
    basecall produced it is exactly the artifact this program exists to prevent.
    """
    if lineage is None:
        return None
    if not isinstance(lineage, dict) or set(lineage) != RAW_LINEAGE_PROVENANCE_KEYS:
        raise RawGenerationError("raw generation lineage provenance is malformed")
    for key in _LINEAGE_REQUIRED_TEXT_KEYS:
        value = lineage.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RawGenerationError(f"raw generation lineage provenance lacks {key}")
    if lineage["generation_kind"] not in {"full_source", "parent_universe", "selected_cohort"}:
        raise RawGenerationError("raw generation lineage generation kind is invalid")
    for key in ("parent_preprocess_generation_id", "source_resolution_digest", "identity_map"):
        value = lineage.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise RawGenerationError(f"raw generation lineage provenance has an invalid {key}")
    return lineage


def validate_raw_generation(
    generation_dir: str | Path,
    *,
    expected_generation_id: str | None = None,
    final_dir: str | Path | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one complete raw generation without mutating it."""
    generation_dir = Path(generation_dir)
    manifest_path = generation_dir / RAW_GENERATION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawGenerationError("raw generation manifest is missing or unreadable") from exc
    generation_schema = int(manifest.get("schema_version", -1))
    if generation_schema not in {1, 2, RAW_GENERATION_SCHEMA_VERSION}:
        raise RawGenerationError("raw generation schema is incompatible")
    validate_raw_lineage_provenance(manifest.get("lineage"))
    if manifest.get("status") != "complete":
        raise RawGenerationError("raw generation is not complete")
    generation_id = str(manifest.get("generation_id", ""))
    if not generation_id or (
        expected_generation_id is not None and generation_id != expected_generation_id
    ):
        raise RawGenerationError("raw generation ID does not match")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RawGenerationError("raw generation artifact manifest is missing")
    required_paths = (
        RAW_GENERATION_ARTIFACT_PATHS
        if generation_schema >= 2
        else _RAW_GENERATION_V1_ARTIFACT_PATHS
    )
    missing = sorted(set(required_paths).difference(artifacts))
    if missing:
        raise RawGenerationError(f"raw generation artifacts are incomplete: {missing}")
    artifact_paths = dict(required_paths)
    region_artifacts = manifest.get("region_artifacts", {})
    if not isinstance(region_artifacts, dict):
        raise RawGenerationError("raw generation region artifact map is invalid")
    artifact_paths.update(
        {f"region:{scope}": str(path) for scope, path in region_artifacts.items()}
    )
    if "barcode_index" in artifacts:
        artifact_paths["barcode_index"] = "barcode_index.parquet"

    for key, expected_relative in artifact_paths.items():
        record = artifacts.get(key)
        if not isinstance(record, dict):
            raise RawGenerationError(f"raw generation artifact is missing: {key}")
        path = _resolve_generation_artifact(generation_dir, record)
        if Path(str(record.get("path"))) != Path(expected_relative):
            raise RawGenerationError(f"raw generation artifact path is invalid: {key}")
        if not path.exists() or str(record.get("sha256", "")) != _checksum(path):
            raise RawGenerationError(f"raw generation artifact is missing or corrupt: {key}")
        if record.get("kind") == "file" and not path.is_file():
            raise RawGenerationError(f"raw generation artifact is not a file: {key}")
        if record.get("kind") == "directory" and (
            not path.is_dir() or (key in _NONEMPTY_DIRECTORIES and not any(path.iterdir()))
        ):
            raise RawGenerationError(f"raw generation artifact directory is invalid: {key}")

    final_dir = Path(final_dir) if final_dir is not None else generation_dir
    run_root = Path(run_root) if run_root is not None else final_dir.parents[2]
    spine, _ = safe_read_h5ad(generation_dir / "spine.h5ad", verbose=False)
    if str(spine.uns.get("raw_generation_id", "")) != generation_id:
        raise RawGenerationError("raw spine generation ID does not match")
    expected_pointers = {
        "molecules_catalog": final_dir / "molecules.parquet",
        "molecule_index": final_dir / "molecule_index",
        "reference_interval_map": final_dir / "reference_interval_map.parquet",
    }
    if generation_schema >= 2:
        expected_pointers.update(
            {
                "segments_catalog": final_dir / "segments.parquet",
                "segment_index": final_dir / "segment_index",
            }
        )
    for key, path in expected_pointers.items():
        if spine.uns.get(key) != relative_uns_path(path, run_root):
            raise RawGenerationError(f"raw spine pointer is unsafe: {key}")
    expected_regions = {
        scope: relative_uns_path(final_dir / relative, run_root)
        for scope, relative in sorted(region_artifacts.items())
    }
    if dict(spine.uns.get("region_catalogs", {})) != expected_regions:
        raise RawGenerationError("raw spine region catalog pointers are unsafe")

    dependencies = manifest.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise RawGenerationError("raw generation dependency manifest is invalid")
    for key, record in dependencies.items():
        if not isinstance(record, dict):
            raise RawGenerationError(f"raw generation dependency is invalid: {key}")
        raw_path = Path(str(record.get("path", "")))
        if record.get("path_kind") != "relative" or record.get("anchor") != "run_root":
            raise RawGenerationError(f"raw generation dependency is not portable: {key}")
        dependency = (run_root / raw_path).resolve()
        if not dependency.is_relative_to(run_root.resolve()):
            raise RawGenerationError(f"raw generation dependency escapes run root: {key}")
        if not dependency.exists() or str(record.get("sha256", "")) != _checksum(dependency):
            raise RawGenerationError(f"raw generation dependency is missing or corrupt: {key}")
    transition = manifest.get("source_transition", {})
    if transition:
        if not isinstance(transition, dict) or int(transition.get("schema_version", -1)) != 1:
            raise RawGenerationError("raw generation source transition is invalid")
        if transition.get("kind") != "append_only" or not transition.get("added_source_ids"):
            raise RawGenerationError("raw generation append transition is incomplete")
        reuse = manifest.get("reuse")
        if not isinstance(reuse, dict) or not reuse.get("generation_id"):
            raise RawGenerationError("raw generation append reuse provenance is missing")
        for field in ("reused_files", "reused_bytes", "new_files", "new_bytes"):
            if int(reuse.get(field, -1)) < 0:
                raise RawGenerationError("raw generation append reuse counts are invalid")
    return manifest


def resolve_current_raw_generation(
    raw_output_dir: str | Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve and validate the generation selected by raw ``current.json``."""
    raw_output_dir = Path(raw_output_dir)
    try:
        selected = resolve_current_generation(
            raw_output_dir,
            manifest_checksum=_checksum,
            require_generation_id=True,
        )
    except GenerationError as exc:
        raise RawGenerationError(str(exc)) from exc
    if selected is None:
        return None
    generation, pointer_manifest = selected
    manifest = validate_raw_generation(
        generation,
        expected_generation_id=str(pointer_manifest.get("generation_id", "")),
        final_dir=generation,
        run_root=raw_output_dir.parent,
    )
    return generation, manifest


def publish_raw_generation(
    run_root: str | Path,
    source_artifacts: Mapping[str, str | Path],
    *,
    config_hash: str,
    input_artifact_ids: list[str],
    dependencies: Mapping[str, str | Path] | None = None,
    region_artifacts: Mapping[str, str | Path] | None = None,
    generation_id: str | None = None,
    reuse_generation: str | Path | None = None,
    source_transition: Mapping[str, Any] | None = None,
    lineage_provenance: Mapping[str, Any] | None = None,
    select_current: bool = True,
) -> dict[str, Path | str]:
    """Snapshot, validate, and atomically publish one immutable raw generation.

    ``lineage_provenance`` marks the result as a re-basecalled descendant and is
    validated before anything is published. Such a generation is normally
    published with ``select_current=False``: it becomes addressable beside the
    parent's without changing what ordinary readers resolve, which only explicit
    promotion does.
    """
    run_root = Path(run_root)
    raw_output_dir = run_root / "raw_outputs"
    lineage = validate_raw_lineage_provenance(
        dict(lineage_provenance) if lineage_provenance is not None else None
    )
    reuse_root = Path(reuse_generation) if reuse_generation is not None else None
    reuse_manifest: dict[str, Any] | None = None
    if reuse_root is not None:
        reuse_manifest = validate_raw_generation(reuse_root, run_root=run_root)
    reuse_stats: dict[str, int] = {
        "reused_files": 0,
        "reused_bytes": 0,
        "new_files": 0,
        "new_bytes": 0,
    }
    normalized_sources = {key: Path(path) for key, path in source_artifacts.items()}
    missing = sorted(
        key
        for key in RAW_REQUIRED_ARTIFACTS
        if key != "sidecar_manifest"
        and (key not in normalized_sources or not normalized_sources[key].exists())
    )
    if missing:
        raise RawGenerationError(f"raw publication source artifacts are incomplete: {missing}")

    region_paths = {str(scope): Path(path) for scope, path in (region_artifacts or {}).items()}
    region_relatives = {
        scope: f"region_catalogs/{path.name}" for scope, path in sorted(region_paths.items())
    }
    artifact_paths = dict(RAW_GENERATION_ARTIFACT_PATHS)
    artifact_paths.update(
        {f"region:{scope}": relative for scope, relative in region_relatives.items()}
    )
    if "barcode_index" in normalized_sources and normalized_sources["barcode_index"].exists():
        artifact_paths["barcode_index"] = "barcode_index.parquet"

    def validate(staging: Path, final: Path, root: Path) -> None:
        validate_raw_generation(
            staging,
            expected_generation_id=staged.generation_id,
            final_dir=final,
            run_root=root,
        )

    def validate_published(_staging: Path, final: Path, root: Path) -> None:
        validate_raw_generation(
            final,
            expected_generation_id=staged.generation_id,
            run_root=root,
        )

    try:
        with staged_generation(
            raw_output_dir,
            run_root=run_root,
            validate=validate,
            generation_id=generation_id,
            manifest_checksum=_checksum,
            write_json=atomic_write_json,
            after_current=validate_published,
            select_current=select_current,
        ) as staged:
            generation_id = staged.generation_id
            staging_dir = staged.staging_dir
            final_dir = staged.final_dir
            for key, relative in artifact_paths.items():
                if key == "sidecar_manifest":
                    continue
                source = (
                    region_paths[key.removeprefix("region:")]
                    if key.startswith("region:")
                    else normalized_sources[key]
                )
                if reuse_root is None:
                    file_count, byte_count = _artifact_file_totals(source)
                    reuse_stats["new_files"] += file_count
                    reuse_stats["new_bytes"] += byte_count
                    _copy_artifact(source, staging_dir / relative)
                else:
                    _copy_artifact(
                        source,
                        staging_dir / relative,
                        reuse_source=reuse_root / relative,
                        reuse_stats=reuse_stats,
                    )
            _bind_generation_spine(
                staging_dir / "spine.h5ad",
                generation_id=generation_id,
                publication_dir=final_dir,
                run_root=run_root,
                region_artifacts=region_relatives,
            )
            _write_generation_sidecar_manifest(staging_dir, artifact_paths)
            artifacts = {
                key: _generation_artifact_record(staging_dir / relative, staging_dir)
                for key, relative in artifact_paths.items()
            }
            dependency_records = {
                str(key): artifact_record(Path(path), run_root, checksum=True)
                for key, path in sorted((dependencies or {}).items())
            }
            staged.record_manifest(
                {
                    "schema_version": RAW_GENERATION_SCHEMA_VERSION,
                    "status": "complete",
                    "generation_id": generation_id,
                    "config_hash": str(config_hash),
                    "input_artifact_ids": list(input_artifact_ids),
                    "region_artifacts": region_relatives,
                    "artifacts": artifacts,
                    "dependencies": dependency_records,
                    "lineage": dict(lineage) if lineage is not None else None,
                    "source_transition": dict(source_transition or {}),
                    "reuse": {
                        "generation_id": (
                            str(reuse_manifest.get("generation_id"))
                            if reuse_manifest is not None
                            else None
                        ),
                        **reuse_stats,
                    },
                }
            )
    except GenerationError as exc:
        raise RawGenerationError(str(exc)) from exc

    current_path = raw_output_dir / RAW_CURRENT_FILENAME

    outputs: dict[str, Path | str] = {
        key: final_dir / relative for key, relative in artifact_paths.items()
    }
    outputs.update(
        {
            "generation": final_dir,
            "generation_manifest": final_dir / RAW_GENERATION_MANIFEST,
            "current": current_path,
            "generation_id": generation_id,
        }
    )
    return outputs
