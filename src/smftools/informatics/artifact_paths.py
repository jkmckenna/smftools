"""Portable serialization and resolution of dataset-relative artifact paths."""

from __future__ import annotations

import os
from pathlib import Path


def serialize_artifact_path(path: str | Path, anchor: str | Path) -> str:
    """Serialize an artifact relative to a dataset root when the platform allows it.

    Windows cannot express a relative path between different drive letters. In
    that case the absolute path is retained as an explicit compatibility
    fallback; same-volume datasets always use a relocatable POSIX-style path.
    """
    resolved_path = Path(path).resolve()
    resolved_anchor = Path(anchor).resolve()
    try:
        return Path(os.path.relpath(resolved_path, start=resolved_anchor)).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def resolve_artifact_path(value: object, anchor: str | Path | None) -> Path | None:
    """Resolve a relative or legacy absolute artifact pointer."""
    if not value:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    if anchor is None:
        return None
    return (Path(anchor) / candidate).resolve()
