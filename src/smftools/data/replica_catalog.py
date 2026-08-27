"""Replica catalog: which volumes hold a copy of which dataset (`PSR-10`).

A **dataset** is identified by its input-manifest digest
(`smftools.informatics.input_manifest.input_manifest_digest`) -- reused rather
than inventing a second identity scheme, per the design in
`dev/plans/completed/portable_storage_roots_implementation_plan.md`.
`PSR-08`'s stamp identifies a *volume*; this identifies *content*. One dataset
maps to however many replicas exist -- an original archive drive, a backup
drive, a working copy -- each recorded as `(volume_id, path relative to that
volume's root)` plus the digest it carried and when that was last confirmed.

The catalog is a plain JSON file, syncable between the user's machines by
copying it, and never required: with no catalog a dataset is simply
unlocatable except by hand, degrading to a bare path in the experiment config
(Layer 2). `PSR-11`'s `data scan` populates it from attached volumes; `PSR-19`
updates it in band as smftools itself publishes and reads data. Neither exists
yet -- this module is schema and API only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

from ..readwrite import atomic_write_json

if TYPE_CHECKING:
    from .volume_discovery import DiscoveredVolume

CATALOG_FILENAME = "replica_catalog.json"
SCHEMA_VERSION = 1

#: Preference order for `resolve_replica` when more than one replica of a
#: dataset is attached at once -- a working copy before an archive before a
#: backup, per the design's "working SSD before archive HDD".
DEFAULT_KIND_PREFERENCE: tuple[str, ...] = ("working", "archive", "backup")


class ReplicaCatalogError(ValueError):
    """The catalog file exists but is not a valid replica catalog."""


@dataclass(frozen=True)
class Replica:
    """One volume's copy of a dataset."""

    volume_id: str
    #: Relative to the volume's own root -- never absolute, never
    #: mount-qualified, so the record stays valid however the volume is
    #: mounted (`PSR-08`/`PSR-09`).
    path: str
    #: The dataset digest this replica carried as of `verified_at`. Recorded
    #: per-replica, not assumed to equal the catalog key, so a replica that
    #: has silently drifted (partial copy, bit rot) is a mismatch `data
    #: verify` (`PSR-11`) can detect rather than something taken on faith.
    digest: str
    verified_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "volume_id": self.volume_id,
            "path": self.path,
            "digest": self.digest,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Replica:
        missing = [
            key for key in ("volume_id", "path", "digest", "verified_at") if key not in payload
        ]
        if missing:
            raise ReplicaCatalogError(f"malformed replica record: missing field(s) {missing}")
        return cls(
            volume_id=str(payload["volume_id"]),
            path=str(payload["path"]),
            digest=str(payload["digest"]),
            verified_at=str(payload["verified_at"]),
        )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_catalog_path() -> Path:
    """Where the machine-local replica catalog lives, mirroring `roots.toml`.

    Shares `SMFTOOLS_CONFIG_DIR` (default `~/.config/smftools`) with the named
    roots file rather than introducing a second override, since both are
    machine-local configuration about where this machine's data lives.
    """
    from ..config.roots import user_roots_file

    return user_roots_file().parent / CATALOG_FILENAME


def load_catalog(path: str | Path | None = None) -> dict[str, list[Replica]]:
    """Read the catalog at `path` (default `default_catalog_path()`).

    Returns an empty catalog, never an error, when the file does not exist --
    a missing catalog degrades gracefully rather than blocking anything that
    reads it.

    Raises:
        ReplicaCatalogError: The file exists but is not valid JSON, is on an
            unsupported schema version, or is structurally malformed.
    """
    catalog_path = Path(path) if path is not None else default_catalog_path()
    if not catalog_path.is_file():
        return {}
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplicaCatalogError(
            f"replica catalog at {catalog_path} is not valid JSON: {exc}"
        ) from exc
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ReplicaCatalogError(
            f"replica catalog at {catalog_path} has an unsupported schema version."
        )
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, Mapping):
        raise ReplicaCatalogError(
            f"replica catalog at {catalog_path} is malformed: 'datasets' is not a table."
        )
    result: dict[str, list[Replica]] = {}
    for digest, entry in datasets.items():
        records = entry.get("replicas") if isinstance(entry, Mapping) else None
        if not isinstance(records, list):
            raise ReplicaCatalogError(
                f"replica catalog at {catalog_path} is malformed: "
                f"dataset {digest!r} has no replica list."
            )
        result[str(digest)] = [Replica.from_dict(record) for record in records]
    return result


def save_catalog(
    catalog: Mapping[str, Sequence[Replica]], *, path: str | Path | None = None
) -> Path:
    """Atomically publish `catalog` to `path` (default `default_catalog_path()`).

    A dataset with no replicas left is dropped from the file rather than
    written as an empty list, keeping the catalog free of dead entries.
    """
    catalog_path = Path(path) if path is not None else default_catalog_path()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "datasets": {
            digest: {"replicas": [replica.to_dict() for replica in replicas]}
            for digest, replicas in catalog.items()
            if replicas
        },
    }
    return atomic_write_json(catalog_path, payload)


def add_replica(
    catalog: Mapping[str, Sequence[Replica]],
    dataset_digest: str,
    *,
    volume_id: str,
    path: str,
    digest: str | None = None,
    verified_at: str | None = None,
) -> dict[str, list[Replica]]:
    """Return a new catalog with one replica added or refreshed.

    A replica already recorded at the same `(volume_id, path)` is updated in
    place -- its `digest`/`verified_at` refreshed -- rather than duplicated,
    since re-scanning a known location is a confirmation, not a new replica.

    Args:
        catalog: The catalog to update. Not mutated; a new dict is returned.
        dataset_digest: The catalog key -- the dataset's input-manifest
            digest.
        volume_id: The volume this replica lives on (`PSR-08`).
        path: The dataset's location relative to that volume's root.
        digest: The digest actually observed at this replica. Defaults to
            `dataset_digest`, i.e. "matches", for the common case of
            registering a replica that was just verified against its key.
        verified_at: ISO-8601 timestamp. Defaults to now.
    """
    updated: dict[str, list[Replica]] = {key: list(value) for key, value in catalog.items()}
    replica = Replica(
        volume_id=volume_id,
        path=path,
        digest=digest if digest is not None else dataset_digest,
        verified_at=verified_at if verified_at is not None else _now(),
    )
    existing = updated.setdefault(dataset_digest, [])
    for index, current in enumerate(existing):
        if current.volume_id == volume_id and current.path == path:
            existing[index] = replica
            break
    else:
        existing.append(replica)
    return updated


def replicas_for(catalog: Mapping[str, Sequence[Replica]], dataset_digest: str) -> list[Replica]:
    """Every replica recorded for `dataset_digest`, attached or not."""
    return list(catalog.get(dataset_digest, ()))


@dataclass(frozen=True)
class ResolvedReplica:
    """A catalog replica that is currently reachable, and where."""

    replica: Replica
    mount_path: Path

    @property
    def resolved_path(self) -> Path:
        """The replica's dataset path on this machine, right now."""
        return self.mount_path / self.replica.path


def resolve_replica(
    catalog: Mapping[str, Sequence[Replica]],
    dataset_digest: str,
    *,
    attached: Iterable["DiscoveredVolume"],
    preference: Sequence[str] = DEFAULT_KIND_PREFERENCE,
) -> Optional[ResolvedReplica]:
    """The first attached replica of `dataset_digest`, in preference order.

    Args:
        catalog: The catalog to search.
        dataset_digest: Which dataset.
        attached: Currently-attached volumes, normally a
            `smftools.data.volume_discovery.discover_volumes()` result.
        preference: Volume kinds, most preferred first. A volume kind not
            named here sorts after every named kind (in encounter order)
            rather than being dropped -- an unrecognized kind is not "no
            replica exists here".

    Returns:
        ResolvedReplica or None: None when no recorded replica is currently
        attached.
    """
    attached_by_id = {found.stamp.volume_id: found for found in attached}
    candidates = [
        (replica, attached_by_id[replica.volume_id])
        for replica in replicas_for(catalog, dataset_digest)
        if replica.volume_id in attached_by_id
    ]
    if not candidates:
        return None

    def _rank(candidate: tuple[Replica, "DiscoveredVolume"]) -> int:
        kind = candidate[1].stamp.kind
        return preference.index(kind) if kind in preference else len(preference)

    replica, found = min(candidates, key=_rank)
    return ResolvedReplica(replica=replica, mount_path=found.mount_path)
