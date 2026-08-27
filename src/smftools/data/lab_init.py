"""`data init`: scaffold a new lab directory tree (`PSR-14`).

Mirrors `smftools project init`'s scaffold pattern, one level up: `data/` for
immutable raw instrument output and `analyses/{runs,projects}/` for
regenerable pipeline output, per
`docs/source/tutorials/directory_organization.md`. Idempotent and additive
only, like every other scaffold in this plan (`project init`,
`data init-volume`): re-running it on an existing lab root only fills in
whatever directories are still missing.
"""

from __future__ import annotations

from pathlib import Path

#: Relative to the lab root, in the order documented in
#: `directory_organization.md`'s "Recommended layout".
_SCAFFOLD_DIRS: tuple[Path, ...] = (
    Path("data"),
    Path("analyses") / "runs",
    Path("analyses") / "projects",
)


def scaffold_lab_root(lab_root: str | Path) -> list[Path]:
    """Create `data/` and `analyses/{runs,projects}/` under `lab_root`.

    Skips any directory that already exists, so re-running this on an
    existing lab root only fills in whatever is still missing -- it never
    touches anything already there, including any data already collected
    under `data/`.

    Returns:
        list[Path]: The directories actually created (empty if the lab root
        was already fully scaffolded).
    """
    lab_root = Path(lab_root)
    lab_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for relative in _SCAFFOLD_DIRS:
        directory = lab_root / relative
        if not directory.exists():
            directory.mkdir(parents=True)
            created.append(directory)
    return created
