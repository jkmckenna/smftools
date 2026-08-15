"""Transactional renaming of a human-readable experiment identifier."""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..informatics.artifact_paths import resolve_artifact_path, serialize_artifact_path
from ..informatics.experiment_identity import resolve_experiment_id
from ..informatics.experiment_manifest import MANIFEST_FILENAME, read_experiment_manifest
from ..informatics.molecule_identity import validate_experiment_uid
from ..project.registry import REGISTRY_FILENAME, load_registry
from ..project.sample_store import PER_SAMPLE_DIRNAME, POINTER_FILENAME

STANDARD_CONFIG_FILENAME = "experiment_config.csv"
RENAME_JOURNAL_PREFIX = ".smftools-rename-"
RENAME_JOURNAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentRenameResult:
    """Paths and identities updated by :func:`rename_experiment_id`."""

    old_id: str
    new_id: str
    experiment_uid: str
    experiment_dir: Path
    config_path: Path | None
    project_dirs: tuple[Path, ...]
    query_sets_unchanged: tuple[str, ...]


@dataclass(frozen=True)
class _FileChange:
    path: Path
    before: bytes
    after: bytes


@dataclass(frozen=True)
class _DirectoryMove:
    source: Path
    destination: Path


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_directory_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("new experiment id cannot be empty")
    if (
        normalized in {".", ".."}
        or Path(normalized).name != normalized
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError("new experiment id must be one directory-safe path component")
    return normalized


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes atomically on the target filesystem."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _apply_transaction(
    file_changes: list[_FileChange],
    directory_moves: list[_DirectoryMove],
    *,
    journal_path: Path,
    old_root: Path,
    new_root: Path,
) -> None:
    journal = {
        "schema_version": RENAME_JOURNAL_SCHEMA_VERSION,
        "state": "prepared",
        "old_root": str(old_root),
        "new_root": str(new_root),
        "files": [
            {
                "path": str(change.path),
                "before_base64": base64.b64encode(change.before).decode("ascii"),
            }
            for change in file_changes
        ],
        "moves": [
            {"source": str(move.source), "destination": str(move.destination)}
            for move in directory_moves
        ],
    }
    _atomic_write_bytes(journal_path, _json_bytes(journal))
    changed: list[_FileChange] = []
    moved: list[_DirectoryMove] = []
    try:
        for change in file_changes:
            _atomic_write_bytes(change.path, change.after)
            changed.append(change)
        for move in directory_moves:
            os.replace(move.source, move.destination)
            moved.append(move)
        journal["state"] = "committed"
        _atomic_write_bytes(journal_path, _json_bytes(journal))
    except BaseException as error:
        rollback_errors: list[str] = []
        for move in reversed(moved):
            try:
                os.replace(move.destination, move.source)
            except BaseException as rollback_error:
                rollback_errors.append(
                    f"could not restore directory {move.source}: {rollback_error}"
                )
        for change in reversed(changed):
            try:
                _atomic_write_bytes(change.path, change.before)
            except BaseException as rollback_error:
                rollback_errors.append(f"could not restore file {change.path}: {rollback_error}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"experiment rename failed and rollback was incomplete: {details}"
            ) from error
        journal_path.unlink(missing_ok=True)
        raise
    try:
        journal_path.unlink()
    except OSError as error:
        warnings.warn(
            f"experiment rename committed, but its completed journal could not be removed: "
            f"{journal_path}: {error}",
            UserWarning,
            stacklevel=2,
        )


def _recover_journal(journal_path: Path, journal: dict[str, Any]) -> Path:
    old_root = Path(journal["old_root"])
    new_root = Path(journal["new_root"])
    state = journal.get("state")
    if state == "committed":
        if not new_root.is_dir() or old_root.exists():
            raise RuntimeError(
                f"committed rename journal has inconsistent directories: {journal_path}"
            )
        journal_path.unlink()
        return new_root
    if state != "prepared":
        raise RuntimeError(f"rename journal has unknown state {state!r}: {journal_path}")

    for record in reversed(journal.get("moves", [])):
        source = Path(record["source"])
        destination = Path(record["destination"])
        if destination.exists() and not source.exists():
            os.replace(destination, source)
        elif source.exists() and not destination.exists():
            continue
        else:
            raise RuntimeError(
                f"cannot recover rename move {source} -> {destination}: expected exactly one path"
            )
    for record in reversed(journal.get("files", [])):
        path = Path(record["path"])
        payload = base64.b64decode(record["before_base64"], validate=True)
        _atomic_write_bytes(path, payload)
    journal_path.unlink()
    return old_root


def _recover_matching_transaction(experiment_dir: Path) -> Path:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for journal_path in experiment_dir.parent.glob(f"{RENAME_JOURNAL_PREFIX}*.json"):
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            roots = {Path(journal["old_root"]), Path(journal["new_root"])}
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if experiment_dir in roots:
            matches.append((journal_path, journal))
    if len(matches) > 1:
        raise RuntimeError(f"multiple rename journals refer to {experiment_dir}")
    if not matches:
        return experiment_dir
    journal_path, journal = matches[0]
    if journal.get("schema_version") != RENAME_JOURNAL_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported rename journal schema {journal.get('schema_version')!r}: {journal_path}"
        )
    return _recover_journal(journal_path, journal)


def _config_after_rename(
    path: Path,
    old_id: str,
    new_id: str,
    old_root: Path,
    new_root: Path,
) -> bytes:
    before = path.read_bytes()
    text = before.decode("utf-8-sig")
    dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text, newline=""), dialect=dialect)
    if reader.fieldnames is None or not {"variable", "value"}.issubset(reader.fieldnames):
        raise ValueError(f"experiment config {path} must contain variable and value columns")
    rows = list(reader)
    by_variable: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_variable.setdefault(str(row.get("variable", "")).strip(), []).append(row)
    for variable in ("experiment_id", "experiment_name", "output_directory"):
        if len(by_variable.get(variable, [])) > 1:
            raise ValueError(f"experiment config {path} contains duplicate {variable!r} rows")

    declared_id = resolve_experiment_id(
        {
            "config experiment_id": (
                by_variable["experiment_id"][0].get("value")
                if by_variable.get("experiment_id")
                else None
            ),
            "config experiment_name": (
                by_variable["experiment_name"][0].get("value")
                if by_variable.get("experiment_name")
                else None
            ),
        }
    )
    if declared_id is not None and declared_id != old_id:
        raise ValueError(
            f"experiment config identity {declared_id!r} does not match manifest id {old_id!r}"
        )

    empty_row = {field: "" for field in reader.fieldnames}
    for variable in ("experiment_id", "experiment_name"):
        if by_variable.get(variable):
            by_variable[variable][0]["value"] = new_id
        else:
            row = dict(empty_row)
            row.update({"variable": variable, "value": new_id})
            rows.append(row)

    output_rows = by_variable.get("output_directory", [])
    if output_rows:
        output_value = str(output_rows[0].get("value", "")).strip()
        output_path = Path(output_value) if output_value else None
        if output_path is None:
            raise ValueError(f"experiment config {path} has an empty output_directory")
        if output_path.is_absolute():
            if output_path.resolve() != old_root:
                raise ValueError(
                    f"config output_directory {output_value!r} does not identify the experiment "
                    f"directory {old_root}"
                )
            rewritten_output = str(new_root)
        else:
            if output_path.name != old_id:
                raise ValueError(
                    f"relative config output_directory {output_value!r} does not end in {old_id!r}"
                )
            rewritten_output = str(output_path.with_name(new_id))
        output_rows[0]["value"] = rewritten_output
    else:
        row = dict(empty_row)
        row.update({"variable": "output_directory", "value": str(new_root)})
        rows.append(row)

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=reader.fieldnames, dialect=dialect, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    prefix = b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
    return prefix + output.getvalue().encode("utf-8")


def _remap_registry_path(value: str, project_dir: Path, old_root: Path, new_root: Path) -> str:
    resolved = resolve_artifact_path(value, project_dir)
    if resolved is None:
        raise ValueError("registry artifact path cannot be empty")
    try:
        relative = resolved.relative_to(old_root)
    except ValueError:
        return value
    return serialize_artifact_path(new_root / relative, project_dir)


def _registry_after_rename(
    project_dir: Path,
    registry: dict[str, Any],
    *,
    old_id: str,
    new_id: str,
    experiment_uid: str,
    old_root: Path,
    new_root: Path,
    renamed_at: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError(
            f"project registry {project_dir / REGISTRY_FILENAME} has no experiments map"
        )
    matching = [
        experiment_id
        for experiment_id, entry in experiments.items()
        if isinstance(entry, dict) and entry.get("experiment_uid") == experiment_uid
    ]
    if matching != [old_id]:
        raise ValueError(
            f"project {project_dir} must register UID {experiment_uid} exactly once as "
            f"{old_id!r}; found {matching or 'no matching entry'}"
        )
    if new_id in experiments:
        raise ValueError(f"project {project_dir} already contains experiment id {new_id!r}")

    updated = json.loads(json.dumps(registry))
    old_entry = updated["experiments"][old_id]
    resolved_entry_root = resolve_artifact_path(old_entry.get("path"), project_dir)
    if resolved_entry_root != old_root:
        raise ValueError(
            f"project {project_dir} records {old_id!r} at {resolved_entry_root}, not {old_root}"
        )
    old_entry["path"] = serialize_artifact_path(new_root, project_dir)
    if old_entry.get("name") == old_id:
        old_entry["name"] = new_id
    for collection in ("spines", "catalogs"):
        values = old_entry.get(collection, {})
        if not isinstance(values, dict):
            raise ValueError(f"registry entry {old_id!r} has invalid {collection!r}")
        old_entry[collection] = {
            key: _remap_registry_path(value, project_dir, old_root, new_root)
            for key, value in values.items()
        }
    updated["experiments"] = {
        (new_id if experiment_id == old_id else experiment_id): entry
        for experiment_id, entry in updated["experiments"].items()
    }

    query_sets: list[str] = []
    sets = updated.get("sets", {})
    if not isinstance(sets, dict):
        raise ValueError(f"project registry {project_dir / REGISTRY_FILENAME} has invalid sets")
    for set_name, definition in sets.items():
        if not isinstance(definition, dict):
            continue
        if definition.get("kind") == "list":
            members = definition.get("experiments", [])
            if not isinstance(members, list) or not all(
                isinstance(member, str) for member in members
            ):
                raise ValueError(
                    f"project {project_dir} set {set_name!r} has invalid list membership"
                )
            if old_id in members and new_id in members:
                raise ValueError(
                    f"project {project_dir} set {set_name!r} contains both {old_id!r} and "
                    f"{new_id!r}"
                )
            definition["experiments"] = [new_id if item == old_id else item for item in members]
        elif definition.get("kind") == "query":
            query_sets.append(str(set_name))
    updated["updated_at"] = renamed_at
    return updated, tuple(query_sets)


def rename_experiment_id(
    experiment_dir: str | Path,
    new_id: str,
    *,
    config_path: str | Path | None = None,
    project_dirs: tuple[str | Path, ...] = (),
) -> ExperimentRenameResult:
    """Rename one experiment and all explicitly discoverable mutable references.

    The durable ``experiment_uid`` remains unchanged. Project query definitions and
    experiment stage artifacts are historical records and are not rewritten.

    Args:
        experiment_dir: Current run-root directory.
        new_id: New human-readable identifier and directory name.
        config_path: Config CSV to update. When omitted, ``experiment_config.csv``
            inside the run root is used if it exists.
        project_dirs: Project directories whose registries and per-sample stores
            must be updated.

    Returns:
        A summary of the completed transaction.

    Raises:
        ValueError: Identity or destination preflight validation fails.
        OSError: A write or directory move fails after successful rollback.
    """
    old_root = _recover_matching_transaction(Path(experiment_dir).resolve())
    if not old_root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {old_root}")
    new_id = _validate_directory_id(new_id)
    manifest_path = old_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"experiment manifest does not exist: {manifest_path}")
    manifest = read_experiment_manifest(old_root)
    old_id = resolve_experiment_id(
        {
            "manifest experiment_id": manifest.get("experiment_id"),
            "manifest experiment": manifest.get("experiment"),
            "run directory": old_root.name,
        },
        required=True,
    )
    assert old_id is not None
    if new_id == old_id:
        raise ValueError(f"new experiment id is already {new_id!r}")
    new_root = old_root.with_name(new_id)
    if new_root.exists():
        raise FileExistsError(f"rename destination already exists: {new_root}")
    experiment_uid = validate_experiment_uid(manifest.get("experiment_uid"))
    renamed_at = _now()

    selected_config = Path(config_path).resolve() if config_path is not None else None
    if selected_config is None:
        standard_config = old_root / STANDARD_CONFIG_FILENAME
        selected_config = standard_config if standard_config.is_file() else None
    if selected_config is not None and not selected_config.is_file():
        raise FileNotFoundError(f"experiment config does not exist: {selected_config}")

    projects = tuple(Path(project).resolve() for project in project_dirs)
    if len(set(projects)) != len(projects):
        raise ValueError("each --project directory may be supplied only once")
    for project in projects:
        try:
            project.relative_to(old_root)
        except ValueError:
            continue
        raise ValueError(f"project directory cannot be inside the experiment directory: {project}")

    manifest_after = json.loads(json.dumps(manifest))
    manifest_after["experiment_id"] = new_id
    manifest_after["experiment"] = new_id
    history = manifest_after.setdefault("experiment_id_history", [])
    if not isinstance(history, list):
        raise ValueError("manifest experiment_id_history must be a list when present")
    history.append(
        {
            "previous_experiment_id": old_id,
            "experiment_id": new_id,
            "renamed_at": renamed_at,
        }
    )

    file_changes = [
        _FileChange(
            path=manifest_path,
            before=manifest_path.read_bytes(),
            after=_json_bytes(manifest_after),
        )
    ]
    if selected_config is not None:
        file_changes.append(
            _FileChange(
                path=selected_config,
                before=selected_config.read_bytes(),
                after=_config_after_rename(
                    selected_config,
                    old_id,
                    new_id,
                    old_root,
                    new_root,
                ),
            )
        )

    directory_moves: list[_DirectoryMove] = []
    query_sets: list[str] = []
    for project in projects:
        registry_path = project / REGISTRY_FILENAME
        registry = load_registry(project)
        updated_registry, unchanged = _registry_after_rename(
            project,
            registry,
            old_id=old_id,
            new_id=new_id,
            experiment_uid=experiment_uid,
            old_root=old_root,
            new_root=new_root,
            renamed_at=renamed_at,
        )
        file_changes.append(
            _FileChange(
                path=registry_path,
                before=registry_path.read_bytes(),
                after=_json_bytes(updated_registry),
            )
        )
        query_sets.extend(f"{project}:{name}" for name in unchanged)

        per_sample_parent = project / "project_outputs" / PER_SAMPLE_DIRNAME
        old_sample_root = per_sample_parent / old_id
        new_sample_root = per_sample_parent / new_id
        if new_sample_root.exists():
            raise FileExistsError(f"per-sample rename destination exists: {new_sample_root}")
        if old_sample_root.exists():
            if not old_sample_root.is_dir():
                raise ValueError(f"per-sample path is not a directory: {old_sample_root}")
            for pointer_path in sorted(old_sample_root.rglob(POINTER_FILENAME)):
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                if pointer.get("experiment_id") != old_id:
                    raise ValueError(
                        f"per-sample pointer {pointer_path} has experiment_id "
                        f"{pointer.get('experiment_id')!r}, expected {old_id!r}"
                    )
                pointer["experiment_id"] = new_id
                file_changes.append(
                    _FileChange(
                        path=pointer_path,
                        before=pointer_path.read_bytes(),
                        after=_json_bytes(pointer),
                    )
                )
            directory_moves.append(_DirectoryMove(old_sample_root, new_sample_root))

    directory_moves.append(_DirectoryMove(old_root, new_root))
    journal_path = old_root.parent / f"{RENAME_JOURNAL_PREFIX}{experiment_uid}.json"
    if journal_path.exists():
        raise FileExistsError(f"rename transaction journal already exists: {journal_path}")
    _apply_transaction(
        file_changes,
        directory_moves,
        journal_path=journal_path,
        old_root=old_root,
        new_root=new_root,
    )

    result_config = selected_config
    if result_config is not None:
        try:
            result_config = new_root / result_config.relative_to(old_root)
        except ValueError:
            pass
    return ExperimentRenameResult(
        old_id=old_id,
        new_id=new_id,
        experiment_uid=experiment_uid,
        experiment_dir=new_root,
        config_path=result_config,
        project_dirs=projects,
        query_sets_unchanged=tuple(query_sets),
    )
