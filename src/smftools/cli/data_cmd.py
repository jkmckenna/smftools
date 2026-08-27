"""CLI logic for `smftools data`: machine- and volume-scoped storage operations."""

from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def data_init_volume(mount: str | Path, *, label: str, kind: str) -> tuple[dict, bool, list[str]]:
    """Stamp `mount` with a permanent volume identity.

    Returns `(stamp_dict, created, warnings)`. `created` is False when the
    volume already carried a stamp, in which case `warnings` names any
    requested `label`/`kind` that differs from what is already stamped --
    the stamp is never rewritten (`PSR-08`), so the request is honored by
    reporting the mismatch rather than by changing the file.
    """
    from ..data.volume_stamp import init_volume

    stamp, created = init_volume(mount, label=label, kind=kind)
    warnings: list[str] = []
    if not created:
        if stamp.label != label:
            warnings.append(
                f"requested label {label!r} ignored; volume is already labeled {stamp.label!r}"
            )
        if stamp.kind != kind:
            warnings.append(
                f"requested kind {kind!r} ignored; volume is already kind {stamp.kind!r}"
            )
    return stamp.to_dict(), created, warnings


def data_list_volumes(*, config_dir: str | Path | None = None) -> list[dict]:
    """Every stamped volume currently attached to this machine, as plain dicts.

    Scans platform mount roots plus any configured `extra_search_paths`
    (`PSR-09`). Reports only what is attached right now -- a volume that is
    stamped but not currently reachable is invisible here until `PSR-10`'s
    catalog exists to remember it.
    """
    from ..data.volume_discovery import discover_volumes

    found = discover_volumes(config_dir=Path(config_dir) if config_dir is not None else None)
    return [{**item.stamp.to_dict(), "mount_path": str(item.mount_path)} for item in found]
