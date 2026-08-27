"""Volume stamps: a permanent identity for a drive, independent of mount point or label.

A stamp file travels with the drive so plugging it into any machine identifies
it with no per-machine configuration -- see `PSR-08` in
`dev/plans/completed/portable_storage_roots_implementation_plan.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..readwrite import atomic_write_json

STAMP_FILENAME = ".smftools-volume.json"

VALID_VOLUME_KINDS = ("working", "archive", "backup")


@dataclass(frozen=True)
class VolumeStamp:
    """Identity stamped onto a drive by `smftools data init-volume`.

    `volume_id` is the only field volume discovery may rely on for identity.
    `label` and `kind` are user-facing metadata recorded at stamp time; they
    may drift from reality (a drive renamed at the OS level) without
    smftools ever knowing or caring, since nothing derives `volume_id` from
    them.
    """

    volume_id: str
    label: str
    kind: str
    created: str

    def to_dict(self) -> dict[str, str]:
        return {
            "volume_id": self.volume_id,
            "label": self.label,
            "kind": self.kind,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VolumeStamp:
        missing = [key for key in ("volume_id", "label", "kind", "created") if key not in payload]
        if missing:
            raise ValueError(f"malformed volume stamp: missing field(s) {missing}")
        return cls(
            volume_id=str(payload["volume_id"]),
            label=str(payload["label"]),
            kind=str(payload["kind"]),
            created=str(payload["created"]),
        )


def _stamp_path(mount: str | Path) -> Path:
    return Path(mount) / STAMP_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_volume_stamp(mount: str | Path) -> VolumeStamp | None:
    """Read the stamp at `mount`, or None if it has never been stamped.

    Raises ValueError if a stamp file exists but is not valid JSON or is
    missing a required field. A corrupt stamp must never be silently treated
    as "unstamped" -- that would let `init_volume` mint a second, conflicting
    `volume_id` for a drive that already has one.
    """
    path = _stamp_path(mount)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"volume stamp at {path} is not valid JSON: {exc}") from exc
    return VolumeStamp.from_dict(payload)


def init_volume(
    mount: str | Path, *, label: str, kind: str = "archive"
) -> tuple[VolumeStamp, bool]:
    """Stamp `mount` with a new, permanent volume identity.

    Returns `(stamp, created)`; `created` is False when `mount` was already
    stamped, in which case the existing stamp is returned untouched and
    `label`/`kind` are ignored. The stamp is written once and never
    rewritten (`PSR-08`), so a drive keeps its `volume_id` even if it is
    later relabeled at the OS level or reattached under a different mount
    name -- discovery only ever reads `volume_id` back out of the stamp
    file, never anything derived from where or how the drive is mounted.
    """
    mount_path = Path(mount)
    if not mount_path.is_dir():
        raise FileNotFoundError(
            f"volume mount point does not exist or is not a directory: {mount_path}"
        )
    if kind not in VALID_VOLUME_KINDS:
        raise ValueError(f"unknown volume kind {kind!r}; expected one of {VALID_VOLUME_KINDS}")

    existing = read_volume_stamp(mount_path)
    if existing is not None:
        return existing, False

    stamp = VolumeStamp(volume_id=uuid4().hex, label=label, kind=kind, created=_now())
    atomic_write_json(_stamp_path(mount_path), stamp.to_dict())
    return stamp, True
