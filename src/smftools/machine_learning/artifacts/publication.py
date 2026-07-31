"""Atomic publication and validation of immutable ML artifact bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from uuid import uuid4

from smftools.readwrite import atomic_write_json

from ..workspace import MLWorkspace
from .common import ArtifactReference
from .model import ModelManifest
from .run import RunManifest

RUN_MANIFEST_FILENAME = "run_manifest.json"
MODEL_MANIFEST_FILENAME = "model_manifest.json"
STAGING_DIRNAME = ".staging"
LOCKS_DIRNAME = ".locks"
_SAFE_ALIAS = re.compile(r"^[0-9A-Za-z._-]+$")
_RESERVED_ROOTS = frozenset({"datasets", "index", "models", "runs"})


class MLArtifactPublicationError(RuntimeError):
    """Base error for ML artifact publication and validation."""


class MLArtifactConflictError(MLArtifactPublicationError):
    """Raised when an immutable artifact identity already has different content."""


@dataclass(frozen=True)
class PublishedBundle:
    """One validated immutable bundle."""

    kind: str
    artifact_id: str
    path: Path
    manifest_sha256: str
    created: bool


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _assert_workspace_directory(workspace: MLWorkspace, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise MLArtifactPublicationError(f"{label} cannot be a symbolic link")
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace.root)
    except ValueError as exc:
        raise MLArtifactPublicationError(f"{label} escapes the active ML workspace") from exc
    return resolved


def _manifest_spec(
    workspace: MLWorkspace,
    manifest: RunManifest | ModelManifest,
) -> tuple[str, str, Path, str]:
    if manifest.workspace_id != workspace.workspace_id:
        raise MLArtifactPublicationError(
            "manifest workspace_id does not match the active ML workspace"
        )
    if isinstance(manifest, RunManifest):
        _assert_workspace_directory(workspace, workspace.runs_root, "runs root")
        return (
            "run",
            manifest.run_id,
            workspace.run_paths(manifest.run_id).root,
            RUN_MANIFEST_FILENAME,
        )
    if isinstance(manifest, ModelManifest):
        _assert_workspace_directory(workspace, workspace.models_root, "models root")
        target = workspace.model_dir(manifest.model_id)
        if target.is_symlink():
            raise MLArtifactPublicationError("model target cannot be a symbolic link")
        return (
            "model",
            manifest.model_id,
            target,
            MODEL_MANIFEST_FILENAME,
        )
    raise TypeError(f"unsupported manifest type: {type(manifest).__name__}")


def _references(manifest: RunManifest | ModelManifest) -> tuple[ArtifactReference, ...]:
    if isinstance(manifest, ModelManifest):
        return (manifest.artifact,)
    values = [manifest.resolved_plan, manifest.resolved_config, *manifest.artifacts]
    if manifest.failure is not None and manifest.failure.traceback_artifact is not None:
        values.append(manifest.failure.traceback_artifact)
    by_path: dict[str, ArtifactReference] = {}
    for reference in values:
        existing = by_path.get(reference.relative_path)
        if existing is not None and existing != reference:
            raise MLArtifactPublicationError(
                f"conflicting checksums for {reference.relative_path!r}"
            )
        by_path[reference.relative_path] = reference
    return tuple(by_path[path] for path in sorted(by_path))


def _bundle_relative_path(reference: str, *, target: Path, workspace: MLWorkspace) -> Path:
    portable = PurePosixPath(reference)
    target_prefix = PurePosixPath(workspace.portable_reference(target))
    if portable.parts[: len(target_prefix.parts)] == target_prefix.parts:
        portable = PurePosixPath(*portable.parts[len(target_prefix.parts) :])
    elif portable.parts and portable.parts[0] in _RESERVED_ROOTS:
        raise MLArtifactPublicationError(
            f"artifact reference {reference!r} targets a different workspace bundle"
        )
    if not portable.parts or portable.is_absolute() or ".." in portable.parts:
        raise MLArtifactPublicationError(
            f"artifact reference {reference!r} is not a bundle-contained portable path"
        )
    return Path(*portable.parts)


def _inventory(
    manifest: RunManifest | ModelManifest,
    *,
    target: Path,
    workspace: MLWorkspace,
) -> dict[str, tuple[Path, ArtifactReference]]:
    result: dict[str, tuple[Path, ArtifactReference]] = {}
    for reference in _references(manifest):
        relative = _bundle_relative_path(
            reference.relative_path,
            target=target,
            workspace=workspace,
        )
        key = relative.as_posix()
        if key in result:
            raise MLArtifactPublicationError(f"duplicate bundle path {key!r}")
        result[key] = (relative, reference)
    return result


def _load_manifest(path: Path, kind: str) -> RunManifest | ModelManifest:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise TypeError("manifest root is not an object")
        return RunManifest.from_dict(raw) if kind == "run" else ModelManifest.from_dict(raw)
    except Exception as exc:
        raise MLArtifactPublicationError(f"invalid {kind} manifest at {path}: {exc}") from exc


def _validate_payload(path: Path, reference: ArtifactReference) -> None:
    if not path.is_file() or path.is_symlink():
        raise MLArtifactPublicationError(
            f"artifact payload is missing or not a regular file: {path}"
        )
    size = path.stat().st_size
    if size != reference.size_bytes:
        raise MLArtifactPublicationError(
            f"artifact size mismatch for {reference.relative_path!r}: "
            f"expected {reference.size_bytes}, observed {size}"
        )
    observed = file_sha256(path)
    if observed != reference.sha256:
        raise MLArtifactPublicationError(
            f"artifact checksum mismatch for {reference.relative_path!r}"
        )


def validate_published_bundle(
    workspace: MLWorkspace,
    path: str | Path,
    *,
    kind: str,
    expected_id: str | None = None,
) -> PublishedBundle:
    """Validate manifest identity, location, complete inventory, and payload bytes."""
    if kind not in {"run", "model"}:
        raise ValueError("kind must be 'run' or 'model'")
    path = Path(path)
    if path.is_symlink():
        raise MLArtifactPublicationError(f"{kind} bundle cannot be a symbolic link")
    path = path.resolve()
    manifest_name = RUN_MANIFEST_FILENAME if kind == "run" else MODEL_MANIFEST_FILENAME
    manifest = _load_manifest(path / manifest_name, kind)
    found_kind, artifact_id, expected_path, _ = _manifest_spec(workspace, manifest)
    if found_kind != kind or path != expected_path.resolve():
        raise MLArtifactPublicationError(f"{kind} manifest is stored at the wrong workspace path")
    if expected_id is not None and artifact_id != expected_id:
        raise MLArtifactPublicationError(f"{kind} manifest identity does not match the expected ID")
    inventory = _inventory(manifest, target=path, workspace=workspace)
    expected_files = {manifest_name, *inventory}
    observed_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        unexpected = sorted(observed_files - expected_files)
        raise MLArtifactPublicationError(
            f"{kind} bundle inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    for relative, reference in inventory.values():
        _validate_payload(path / relative, reference)
    return PublishedBundle(
        kind=kind,
        artifact_id=artifact_id,
        path=path,
        manifest_sha256=file_sha256(path / manifest_name),
        created=False,
    )


def _safe_existing(
    workspace: MLWorkspace,
    *,
    final: Path,
    kind: str,
    artifact_id: str,
    manifest: RunManifest | ModelManifest,
) -> PublishedBundle:
    try:
        existing = validate_published_bundle(
            workspace,
            final,
            kind=kind,
            expected_id=artifact_id,
        )
    except MLArtifactPublicationError as exc:
        raise MLArtifactConflictError(
            f"{kind} ID {artifact_id} is already bound to invalid or different content"
        ) from exc
    stored = _load_manifest(
        final / (RUN_MANIFEST_FILENAME if kind == "run" else MODEL_MANIFEST_FILENAME),
        kind,
    )
    if stored != manifest:
        raise MLArtifactConflictError(
            f"{kind} ID {artifact_id} is already bound to a different manifest"
        )
    return existing


@contextmanager
def _publication_lock(root: Path, artifact_id: str, timeout: float) -> Iterator[None]:
    locks = root / LOCKS_DIRNAME
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"{artifact_id}.lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise MLArtifactPublicationError(
                    f"timed out waiting to publish artifact {artifact_id}"
                ) from None
            time.sleep(0.02)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _copy_payload(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise MLArtifactPublicationError(f"artifact source is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _remove_empty_staging_root(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def publish_bundle(
    workspace: MLWorkspace,
    manifest: RunManifest | ModelManifest,
    *,
    sources: Mapping[str, str | Path],
    lock_timeout: float = 10.0,
) -> PublishedBundle:
    """Stage, verify, and atomically publish one immutable run or model bundle.

    Source keys must exactly match the manifest's artifact reference strings.
    A reference may be relative to the bundle or may include its expected
    workspace-relative bundle prefix.
    """
    kind, artifact_id, final, manifest_name = _manifest_spec(workspace, manifest)
    inventory = _inventory(manifest, target=final, workspace=workspace)
    if set(sources) != {reference.relative_path for _, reference in inventory.values()}:
        expected = sorted(reference.relative_path for _, reference in inventory.values())
        raise MLArtifactPublicationError(
            f"sources must exactly match the manifest inventory: {expected}"
        )
    if lock_timeout < 0:
        raise ValueError("lock_timeout must be non-negative")
    category_root = final.parent
    category_root.mkdir(parents=True, exist_ok=True)
    with _publication_lock(category_root, artifact_id, lock_timeout):
        if final.exists():
            return _safe_existing(
                workspace,
                final=final,
                kind=kind,
                artifact_id=artifact_id,
                manifest=manifest,
            )
        transaction = category_root / STAGING_DIRNAME / uuid4().hex
        staged = transaction / artifact_id
        staged.mkdir(parents=True)
        try:
            for relative, reference in inventory.values():
                source = Path(sources[reference.relative_path]).resolve()
                _validate_payload(source, reference)
                _copy_payload(source, staged / relative)
                _validate_payload(staged / relative, reference)
            atomic_write_json(staged / manifest_name, manifest.to_dict())
            validate_published_bundle(
                _StagedWorkspace(workspace, final, staged),
                staged,
                kind=kind,
                expected_id=artifact_id,
            )
            if final.exists():
                existing = _safe_existing(
                    workspace,
                    final=final,
                    kind=kind,
                    artifact_id=artifact_id,
                    manifest=manifest,
                )
                shutil.rmtree(transaction)
                _remove_empty_staging_root(transaction.parent)
                return existing
            os.replace(staged, final)
            _fsync_directory(category_root)
            transaction.rmdir()
            _remove_empty_staging_root(transaction.parent)
            published = validate_published_bundle(
                workspace,
                final,
                kind=kind,
                expected_id=artifact_id,
            )
            return replace(published, created=True)
        except Exception:
            shutil.rmtree(transaction, ignore_errors=True)
            _remove_empty_staging_root(transaction.parent)
            raise


class _StagedWorkspace:
    """Resolve one manifest target to its staged location during validation."""

    def __init__(self, workspace: MLWorkspace, final: Path, staged: Path) -> None:
        self._workspace = workspace
        self._final = final.resolve()
        self._staged = staged.resolve()
        self.root = workspace.root
        self.runs_root = workspace.runs_root
        self.models_root = workspace.models_root
        self.workspace_id = workspace.workspace_id

    def run_paths(self, run_id: str):
        paths = self._workspace.run_paths(run_id)
        if paths.root.resolve() != self._final:
            return paths
        return _StagedRunPaths(self._staged)

    def model_dir(self, model_id: str) -> Path:
        path = self._workspace.model_dir(model_id)
        return self._staged if path.resolve() == self._final else path

    def portable_reference(self, path: str | Path) -> str:
        candidate = Path(path).resolve()
        if candidate == self._staged:
            candidate = self._final
        return self._workspace.portable_reference(candidate)


@dataclass(frozen=True)
class _StagedRunPaths:
    root: Path


def cleanup_abandoned_staging(
    workspace: MLWorkspace,
    *,
    older_than_seconds: float,
    now: float | None = None,
) -> tuple[Path, ...]:
    """Remove abandoned staging transactions and publication locks."""
    if older_than_seconds < 0:
        raise ValueError("older_than_seconds must be non-negative")
    cutoff = (time.time() if now is None else now) - older_than_seconds
    removed: list[Path] = []
    for category in (workspace.runs_root, workspace.models_root):
        _assert_workspace_directory(workspace, category, f"{category.name} root")
        staging = category / STAGING_DIRNAME
        if not staging.is_dir():
            continue
        for transaction in staging.iterdir():
            if transaction.is_dir() and not transaction.is_symlink():
                if transaction.stat().st_mtime <= cutoff:
                    shutil.rmtree(transaction)
                    removed.append(transaction)
        try:
            staging.rmdir()
        except OSError:
            pass
        locks = category / LOCKS_DIRNAME
        if locks.is_dir() and not locks.is_symlink():
            for lock in locks.glob("*.lock"):
                if lock.is_file() and not lock.is_symlink() and lock.stat().st_mtime <= cutoff:
                    lock.unlink()
                    removed.append(lock)
            try:
                locks.rmdir()
            except OSError:
                pass
    return tuple(sorted(removed))


def validate_alias_name(alias: str) -> str:
    """Validate one portable mutable alias component."""
    if (
        not isinstance(alias, str)
        or not alias
        or alias in {".", ".."}
        or _SAFE_ALIAS.fullmatch(alias) is None
    ):
        raise ValueError("alias must be one filesystem-safe path component")
    return alias
