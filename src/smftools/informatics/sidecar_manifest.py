from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..readwrite import atomic_write_json

MANIFEST_FILENAME = "sidecar_manifest.json"
SCHEMA_VERSION = 2


def sidecar_manifest_path(output_dir: str | Path) -> Path:
    """Return canonical sidecar manifest path for a load output directory."""
    return Path(output_dir) / MANIFEST_FILENAME


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": SCHEMA_VERSION, "sidecars": {}}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {"version": SCHEMA_VERSION, "sidecars": {}}
    data.setdefault("version", 1)
    sidecars = data.get("sidecars")
    if not isinstance(sidecars, dict):
        data["sidecars"] = {}
    return data


def register_sidecar(
    manifest_path: str | Path,
    key: str,
    sidecar_path: str | Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Register or update one sidecar entry in the manifest JSON file."""
    mpath = Path(manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(mpath)
    sidecar_path = Path(sidecar_path)
    try:
        stored_path = Path(
            os.path.relpath(sidecar_path.resolve(), start=mpath.parent.resolve())
        ).as_posix()
        path_kind = "relative"
        anchor = "manifest_parent"
    except ValueError:
        # Windows cannot express a relative path across drives.
        stored_path = str(sidecar_path.resolve())
        path_kind = "absolute"
        anchor = None
    payload: Dict[str, Any] = {
        "path": stored_path,
        "path_kind": path_kind,
        "anchor": anchor,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        reserved = {"path", "path_kind", "anchor", "updated_at_utc"}.intersection(metadata)
        if reserved:
            raise ValueError(f"sidecar metadata cannot replace reserved fields: {sorted(reserved)}")
        payload.update(metadata)
    manifest["sidecars"][key] = payload
    manifest["version"] = SCHEMA_VERSION
    atomic_write_json(mpath, manifest)
    return mpath


def resolve_sidecar(manifest_path: str | Path, key: str) -> Optional[Path]:
    """Resolve a sidecar path from manifest by key, returning existing path only."""
    mpath = Path(manifest_path)
    if not mpath.exists():
        return None
    manifest = _load_manifest(mpath)
    entry = manifest.get("sidecars", {}).get(key)
    if not isinstance(entry, dict):
        return None
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        return None
    p = Path(raw_path)
    path_kind = entry.get("path_kind")
    if path_kind == "relative":
        if entry.get("anchor") != "manifest_parent":
            return None
        p = mpath.parent / p
    elif path_kind != "absolute" and not p.is_absolute():
        # Backward-compatible read for version-1 relative entries.
        p = mpath.parent / p
    return p if p.exists() else None
