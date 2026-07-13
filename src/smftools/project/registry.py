"""Project registry: register experiments by pointer (append-only) + named sets.

The registry (``registry.json``) is the project's source of truth for *which*
experiments belong to it and where they live. Experiments are never copied; each
entry is a pointer to a self-describing experiment directory (one containing a
``spine.h5ad``). Registration reads the spine's ``uns`` for modality + per-reference
``reference_uids`` so the project can harmonize references without opening matrices.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

REGISTRY_FILENAME = "registry.json"
SETS_SUBDIR = "sets"
SCHEMA_VERSION = 1
SPINE_FILENAME = "spine.h5ad"
RAW_SUBDIR = "raw"  # experiments may nest the spine under a raw/ dir
_CATALOG_NAMES = ("interval_catalog.parquet", "catalog.parquet")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_registry_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / REGISTRY_FILENAME


def init_project(project_dir: str | Path) -> Path:
    """Create a project directory + empty registry (idempotent)."""
    project_dir = Path(project_dir)
    (project_dir / SETS_SUBDIR).mkdir(parents=True, exist_ok=True)
    path = project_registry_path(project_dir)
    if not path.exists():
        save_registry(
            project_dir,
            {"schema_version": SCHEMA_VERSION, "created_at": _now(), "experiments": {}, "sets": {}},
        )
        logger.info("Initialized project at %s", project_dir)
    return path


def load_registry(project_dir: str | Path) -> dict:
    path = project_registry_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"no project registry at {path}; run 'project init' first")
    with path.open() as handle:
        return json.load(handle)


def save_registry(project_dir: str | Path, registry: dict) -> Path:
    path = project_registry_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = dict(registry)
    registry["updated_at"] = _now()
    with path.open("w") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
    return path


def _locate_spine(experiment_dir: Path) -> Path:
    for candidate in (experiment_dir / SPINE_FILENAME, experiment_dir / RAW_SUBDIR / SPINE_FILENAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {SPINE_FILENAME} found under {experiment_dir}")


def _relative_registry_path(path: Path, anchor: Path) -> str:
    """Return ``path`` as a POSIX-style string relative to ``anchor``, for ``registry.json``.

    Kept relative (not resolved to an absolute string) so the registry -- and the
    project directory it lives in -- survives being copied to a different machine
    or mount point, the same way the project's ``runs/`` symlinks already do.
    Pair with :func:`_resolve_registry_path` on the read side.
    """
    import os

    return Path(os.path.relpath(path.resolve(), start=anchor.resolve())).as_posix()


def _resolve_registry_path(value: str, anchor: Path) -> Path:
    """Resolve a ``registry.json`` path value written by :func:`_relative_registry_path`.

    Accepts both the new relative encoding (resolved against ``anchor``, the
    project dir) and the historical absolute-string encoding (used as-is, for
    registries written before this fix), so existing registries keep working
    without needing every experiment re-added.
    """
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (anchor / candidate).resolve()


def _discover_catalogs(experiment_dir: Path, project_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for base in (experiment_dir, experiment_dir / RAW_SUBDIR):
        for name in _CATALOG_NAMES:
            path = base / name
            if path.exists() and name not in found:
                found[name] = _relative_registry_path(path, project_dir)
    return found


def _read_spine_metadata(spine_path: Path) -> dict:
    from ..readwrite import safe_read_h5ad

    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    uns = spine.uns
    return {
        "modality": str(uns.get("modality", "unknown")),
        "reference_uids": {str(k): str(v) for k, v in dict(uns.get("reference_uids", {}) or {}).items()},
        "experiment": uns.get("experiment"),
        "schema_version": int(uns.get("raw_schema_version", 0) or 0),
        "n_reads": int(spine.n_obs),
    }


def add_experiment(
    project_dir: str | Path,
    experiment_dir: str | Path,
    *,
    experiment_id: str | None = None,
    name: str | None = None,
) -> tuple[str, dict]:
    """Register (or refresh) one experiment by pointer. Append-only; O(1)."""
    project_dir = Path(project_dir)
    experiment_dir = Path(experiment_dir).resolve()
    registry = load_registry(project_dir)

    spine_path = _locate_spine(experiment_dir)
    meta = _read_spine_metadata(spine_path)
    exp_id = str(experiment_id or meta.get("experiment") or experiment_dir.name)

    existing = registry["experiments"].get(exp_id)
    if existing and _resolve_registry_path(existing["path"], project_dir) != experiment_dir:
        raise ValueError(
            f"experiment id '{exp_id}' already registered at "
            f"{_resolve_registry_path(existing['path'], project_dir)}; pass a distinct --id"
        )

    spine_rel = (
        str(spine_path.relative_to(experiment_dir))
        if spine_path.is_relative_to(experiment_dir)
        else str(spine_path)
    )
    entry = {
        "path": _relative_registry_path(experiment_dir, project_dir),
        "name": name or exp_id,
        "modality": meta["modality"],
        "schema_version": meta["schema_version"],
        "spine": spine_rel,
        "references": meta["reference_uids"],
        "n_reads": meta["n_reads"],
        "date_added": existing["date_added"] if existing else _now(),
        "status": "active",
        "catalogs": _discover_catalogs(experiment_dir, project_dir),
    }
    registry["experiments"][exp_id] = entry
    save_registry(project_dir, registry)
    logger.info(
        "Registered experiment '%s' (%d reads, %d references)",
        exp_id,
        entry["n_reads"],
        len(entry["references"]),
    )
    return exp_id, entry


def remove_experiment(project_dir: str | Path, experiment_id: str) -> None:
    """Mark an experiment inactive (queries filter to active experiments)."""
    registry = load_registry(project_dir)
    if experiment_id in registry["experiments"]:
        registry["experiments"][experiment_id]["status"] = "inactive"
        save_registry(project_dir, registry)


def list_experiments(project_dir: str | Path, *, active_only: bool = True) -> list[dict]:
    """Return registered experiments with ``path``/``catalogs`` resolved to absolute paths.

    ``registry.json`` stores these relative to ``project_dir`` (see
    :func:`_relative_registry_path`); this is the single point where callers get
    back concrete, directly-usable paths, so the relative on-disk encoding stays
    an implementation detail.
    """
    project_dir = Path(project_dir)
    registry = load_registry(project_dir)
    results = []
    for exp_id, entry in registry["experiments"].items():
        if active_only and entry.get("status") != "active":
            continue
        resolved = dict(entry)
        resolved["path"] = str(_resolve_registry_path(entry["path"], project_dir))
        resolved["catalogs"] = {
            name: str(_resolve_registry_path(value, project_dir))
            for name, value in entry.get("catalogs", {}).items()
        }
        results.append({"id": exp_id, **resolved})
    return results


def add_set(
    project_dir: str | Path,
    name: str,
    *,
    experiments: list[str] | None = None,
    query: str | None = None,
) -> None:
    """Define a named set as an explicit experiment list OR a saved query."""
    if (experiments is None) == (query is None):
        raise ValueError("provide exactly one of experiments= or query=")
    registry = load_registry(project_dir)
    registry.setdefault("sets", {})[name] = (
        {"kind": "list", "experiments": [str(e) for e in experiments]}
        if experiments is not None
        else {"kind": "query", "sql": str(query)}
    )
    save_registry(project_dir, registry)


def resolve_set(project_dir: str | Path, name: str) -> dict:
    registry = load_registry(project_dir)
    the_set = registry.get("sets", {}).get(name)
    if the_set is None:
        raise KeyError(f"no set '{name}' in project")
    return the_set
