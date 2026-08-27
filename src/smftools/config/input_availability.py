"""Classify whether an experiment's raw input is present, offline, or missing.

Raw sequencing input is routinely archived to removable storage once an
experiment's pipeline outputs exist. Every stage used to fail the moment that
volume was detached, because config loading discovered input files eagerly and
`discover_input_files` raises on an absent path -- so `hmm`, `spatial`, and
`export-bundle`, none of which read a byte of raw input, failed alongside `raw`.

Three states replace that single failure:

``present``
    The path resolves and can be discovered, exactly as before.
``offline``
    The path lies on a volume that is not currently attached. Expected, and not
    an error: only the stages that consume raw input object.
``missing``
    The volume is attached (or the path was never on removable storage) and the
    path is simply absent. A real error, reported as early as it ever was.

The volume test here is structural -- a path under a platform mount root whose
mount directory is absent. `PSR-12` replaces it with an exact answer once
volumes carry a durable identity; until then this is deliberately conservative,
because misreading a deleted path as "offline" would silently skip ingestion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

INPUT_PRESENT = "present"
INPUT_OFFLINE = "offline"
INPUT_MISSING = "missing"

#: Mount roots by platform, mapped to the depth at which the *volume* directory
#: sits beneath them. macOS mounts at ``/Volumes/<label>``; Linux uses
#: ``/mnt/<label>``, ``/media/<user>/<label>``, or ``/run/media/<user>/<label>``.
#: Public (not underscore-prefixed) because ``smftools.data.volume_discovery``
#: (`PSR-09`) reuses it as the single source of truth for platform mount
#: conventions rather than duplicating this table.
MOUNT_ROOTS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("/", "Volumes"), 1),
    (("/", "mnt"), 1),
    (("/", "media"), 2),
    (("/", "run", "media"), 2),
)


@dataclass(frozen=True)
class InputAvailability:
    """The resolved state of one configured input path."""

    state: str
    path: Optional[Path] = None
    volume: Optional[Path] = None
    detail: str = ""

    @property
    def is_present(self) -> bool:
        """Whether the input can be read right now."""
        return self.state == INPUT_PRESENT


def detached_volume_for(path: Path) -> Optional[Path]:
    """Return the mount directory ``path`` sits on when that directory is absent.

    Args:
        path: An absolute path that does not currently resolve.

    Returns:
        Path or None: The volume directory (e.g. ``/Volumes/<label>``) when
        ``path`` lies under a known mount root and that volume directory does
        not exist, otherwise ``None``.
    """
    parts = Path(path).parts
    for root, depth in MOUNT_ROOTS:
        if parts[: len(root)] != root:
            continue
        volume_parts = parts[: len(root) + depth]
        if len(volume_parts) < len(root) + depth:
            # The path names the mount root itself, not a volume beneath it.
            continue
        volume = Path(*volume_parts)
        if not volume.exists():
            return volume
    return None


def resolve_input_availability(path: str | Path | None) -> InputAvailability:
    """Classify one configured input path as present, offline, or missing.

    Args:
        path: The configured ``input_data_path`` or manifest path. ``None``
            resolves to ``present``, since an unset path is not a failure here;
            the caller decides whether one was required.

    Returns:
        InputAvailability: The state, plus the detached volume and a
        human-readable sentence when the input cannot be read.
    """
    if path is None:
        return InputAvailability(state=INPUT_PRESENT)
    resolved = Path(path).expanduser()
    if resolved.exists():
        return InputAvailability(state=INPUT_PRESENT, path=resolved)

    volume = detached_volume_for(resolved)
    if volume is not None:
        return InputAvailability(
            state=INPUT_OFFLINE,
            path=resolved,
            volume=volume,
            detail=(
                f"input path {resolved} is on volume {volume}, which is not attached. "
                "Stages that do not read raw input continue normally; attach the volume "
                "to run ingestion."
            ),
        )
    return InputAvailability(
        state=INPUT_MISSING,
        path=resolved,
        detail=f"input path does not exist: {resolved}",
    )


def require_input_available(availability: InputAvailability, *, stage: str) -> None:
    """Raise unless the input can be read, naming what to attach.

    Args:
        availability: The state resolved at config load.
        stage: The stage requiring the input, named in the error.

    Raises:
        FileNotFoundError: If the input is offline or missing.
    """
    if availability.is_present:
        return
    if availability.state == INPUT_OFFLINE:
        raise FileNotFoundError(
            f"stage {stage!r} reads raw input, which is currently offline: "
            f"{availability.path} is on volume {availability.volume}, which is not "
            "attached. Attach the volume and re-run. Stages that do not read raw "
            "input (preprocess, spatial, hmm, latent, export-bundle) run without it."
        )
    raise FileNotFoundError(
        f"stage {stage!r} reads raw input, which is missing: {availability.path}. "
        "The path's volume is attached, so this is an absent or mistyped path rather "
        "than archived data."
    )


def restore_offline_input_identity(
    output_directory: str | Path | None, *, bam_suffix: str
) -> Optional[tuple[str, list[Path]]]:
    """Recover the input type and file list a completed run already recorded.

    A stage decides whether it is already complete by comparing config hashes, and
    ``input_type``/``input_files`` feed that hash. Discovery cannot produce them
    while the input volume is detached, so without this every stage would see a
    changed hash, judge a finished raw generation incompatible, and try to
    re-ingest data it cannot reach (`PSR-01`).

    The values come from the run's resolved input manifest -- the sources raw
    actually ingested -- and **not** from the manifest's config snapshot, which
    records ``input_type`` after the FASTQ-to-BAM rewrite has already replaced it
    and so would not reproduce the hash.

    Args:
        output_directory: The run root, or ``None`` when unset.
        bam_suffix: The configured BAM suffix, for classification.

    Returns:
        tuple or None: ``(input_type, input_files)`` when a resolved input
        manifest is readable, otherwise ``None``.
    """
    if output_directory is None:
        return None
    manifest = (
        Path(output_directory) / "raw_outputs" / "input_manifest" / "resolved_input_manifest.json"
    )
    if not manifest.is_file():
        return None
    try:
        with manifest.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    base = payload.get("base_directory")
    sources = payload.get("sources") or []
    paths: list[Path] = []
    for source in sources:
        raw_path = source.get("path")
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute() and base:
            candidate = Path(base) / candidate
        paths.append(candidate)
    if not paths:
        return None

    from .discover_input_files import input_kind_for_path

    kinds = {input_kind_for_path(path, bam_suffix=bam_suffix) for path in paths}
    if len(kinds) != 1:
        # Mixed recorded types would have been refused at ingestion, so this means
        # the record is not one this restoration understands. Fall back rather
        # than assert an identity that could differ from the hashed one.
        return None
    return kinds.pop(), paths
