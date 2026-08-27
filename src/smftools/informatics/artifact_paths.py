"""Portable serialization and resolution of dataset-relative artifact paths."""

from __future__ import annotations

import os
from pathlib import Path


def serialize_artifact_path(path: str | Path, anchor: str | Path) -> str:
    """Serialize an artifact portably, preferring the most stable encoding available.

    Three encodings, in order of preference:

    1. **Plain relative** when the artifact lies inside ``anchor``. Nothing more
       stable exists: the pair moves as a unit.
    2. **Root-qualified** ``${root}/relative`` when a bound storage root contains
       it but ``anchor`` does not. A relative walk out of the anchor and back
       down would otherwise encode the mount name and the anchor's own depth --
       the two things a named root exists to stop encoding (`PSR-07`).
    3. **Relative walk**, then absolute. Windows cannot express a relative path
       between drive letters, so the absolute form remains the last resort.
    """
    resolved_path = Path(path).resolve()
    resolved_anchor = Path(anchor).resolve()
    if resolved_path == resolved_anchor or resolved_anchor in resolved_path.parents:
        try:
            return Path(os.path.relpath(resolved_path, start=resolved_anchor)).as_posix()
        except ValueError:
            return resolved_path.as_posix()

    from smftools.config.roots import qualify_with_root

    qualified = qualify_with_root(resolved_path, config_dir=resolved_anchor)
    if qualified is not None:
        return qualified

    try:
        return Path(os.path.relpath(resolved_path, start=resolved_anchor)).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def resolve_artifact_path(value: object, anchor: str | Path | None) -> Path | None:
    """Resolve any of the three artifact-pointer encodings.

    Accepts plain relative, root-qualified ``${root}/relative``, and legacy
    absolute strings, so pointers written by older versions keep resolving
    without anything being rewritten (`PSR-07`).
    """
    if not value:
        return None
    text = str(value)
    if "${" in text:
        from smftools.config.roots import expand_roots

        text = expand_roots(
            text, config_dir=Path(anchor) if anchor is not None else None, field="artifact pointer"
        )
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if anchor is None:
        return None
    return (Path(anchor) / candidate).resolve()
