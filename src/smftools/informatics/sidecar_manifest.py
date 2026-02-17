from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_FILENAME = "sidecar_manifest.json"


def sidecar_manifest_path(output_dir: str | Path) -> Path:
    """Return canonical sidecar manifest path for a load output directory."""
    return Path(output_dir) / MANIFEST_FILENAME


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "sidecars": {}}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {"version": 1, "sidecars": {}}
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
    payload: Dict[str, Any] = {
        "path": str(Path(sidecar_path)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload.update(metadata)
    manifest["sidecars"][key] = payload
    with mpath.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
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
    return p if p.exists() else None
