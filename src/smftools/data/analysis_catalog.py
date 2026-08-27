"""Analysis-location catalog: which volumes hold a copy of which run's analysis tree (`PSR-19`).

Distinct from `smftools.data.replica_catalog` (`PSR-10`), which tracks
*interchangeable* raw-dataset replicas keyed by content digest. Two copies of
a run's analysis tree are not interchangeable -- each may hold different
generations (`smftools.data.run_locality`, `PSR-17`) -- so this catalog only
records *where* known copies are; it never picks one as authoritative. That
judgment belongs to `compare_run_locations` at query time, not to anything
stored here.

Keyed by `experiment_uid` (`smftools.informatics.molecule_identity`), the
same durable, content-independent identity `PSR-17`'s duplicate detection
uses -- not a path or the human-chosen `experiment_id` label, neither of
which is stable across a rename or a machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..readwrite import atomic_write_json

CATALOG_FILENAME = "analysis_catalog.json"
SCHEMA_VERSION = 1


class AnalysisCatalogError(ValueError):
    """The catalog file exists but is not a valid analysis-location catalog."""


@dataclass(frozen=True)
class AnalysisLocation:
    """One volume's copy of a run's analysis tree."""

    volume_id: str
    #: Relative to the volume's own root -- never absolute, never
    #: mount-qualified (`PSR-08`/`PSR-09`).
    path: str
    scanned_at: str

    def to_dict(self) -> dict[str, str]:
        return {"volume_id": self.volume_id, "path": self.path, "scanned_at": self.scanned_at}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnalysisLocation:
        missing = [key for key in ("volume_id", "path", "scanned_at") if key not in payload]
        if missing:
            raise AnalysisCatalogError(f"malformed analysis location: missing field(s) {missing}")
        return cls(
            volume_id=str(payload["volume_id"]),
            path=str(payload["path"]),
            scanned_at=str(payload["scanned_at"]),
        )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_catalog_path() -> Path:
    """Where the machine-local analysis-location catalog lives, mirroring `roots.toml`."""
    from ..config.roots import user_roots_file

    return user_roots_file().parent / CATALOG_FILENAME


def load_catalog(path: str | Path | None = None) -> dict[str, list[AnalysisLocation]]:
    """Read the catalog at `path` (default `default_catalog_path()`).

    Returns an empty catalog, never an error, when the file does not exist.

    Raises:
        AnalysisCatalogError: The file exists but is not valid JSON, is on an
            unsupported schema version, or is structurally malformed.
    """
    catalog_path = Path(path) if path is not None else default_catalog_path()
    if not catalog_path.is_file():
        return {}
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisCatalogError(
            f"analysis catalog at {catalog_path} is not valid JSON: {exc}"
        ) from exc
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise AnalysisCatalogError(
            f"analysis catalog at {catalog_path} has an unsupported schema version."
        )
    runs = payload.get("runs", {})
    if not isinstance(runs, Mapping):
        raise AnalysisCatalogError(
            f"analysis catalog at {catalog_path} is malformed: 'runs' is not a table."
        )
    result: dict[str, list[AnalysisLocation]] = {}
    for experiment_uid, entry in runs.items():
        records = entry.get("locations") if isinstance(entry, Mapping) else None
        if not isinstance(records, list):
            raise AnalysisCatalogError(
                f"analysis catalog at {catalog_path} is malformed: "
                f"run {experiment_uid!r} has no locations list."
            )
        result[str(experiment_uid)] = [AnalysisLocation.from_dict(record) for record in records]
    return result


def save_catalog(
    catalog: Mapping[str, Sequence[AnalysisLocation]], *, path: str | Path | None = None
) -> Path:
    """Atomically publish `catalog` to `path` (default `default_catalog_path()`).

    A run with no locations left is dropped from the file rather than written
    as an empty list.
    """
    catalog_path = Path(path) if path is not None else default_catalog_path()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "runs": {
            experiment_uid: {"locations": [location.to_dict() for location in locations]}
            for experiment_uid, locations in catalog.items()
            if locations
        },
    }
    return atomic_write_json(catalog_path, payload)


def add_location(
    catalog: Mapping[str, Sequence[AnalysisLocation]],
    experiment_uid: str,
    *,
    volume_id: str,
    path: str,
    scanned_at: str | None = None,
) -> dict[str, list[AnalysisLocation]]:
    """Return a new catalog with one location added or refreshed.

    A location already recorded at the same `(volume_id, path)` has its
    `scanned_at` refreshed in place rather than being duplicated.
    """
    updated: dict[str, list[AnalysisLocation]] = {
        key: list(value) for key, value in catalog.items()
    }
    location = AnalysisLocation(volume_id=volume_id, path=path, scanned_at=scanned_at or _now())
    existing = updated.setdefault(experiment_uid, [])
    for index, current in enumerate(existing):
        if current.volume_id == volume_id and current.path == path:
            existing[index] = location
            break
    else:
        existing.append(location)
    return updated


def locations_for(
    catalog: Mapping[str, Sequence[AnalysisLocation]], experiment_uid: str
) -> list[AnalysisLocation]:
    """Every catalogued analysis location for `experiment_uid`."""
    return list(catalog.get(experiment_uid, ()))
