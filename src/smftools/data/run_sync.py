"""`data sync`: copy missing generations between two locations of a run (`PSR-20`).

Generations are content-addressed and never edited after publication
(`smftools.informatics.generation_listing`), so copying one a destination
lacks cannot corrupt anything and can resume after an interruption. Sync is
therefore purely additive: for a stage `compare_run_locations` (`PSR-17`)
finds one location `ahead` of the other, the missing generation directories
are copied across, in either direction, and `current.json` is never touched.

`diverged` and `pointer_conflict` stages are reported, never resolved --
there is no flag that picks a side by timestamp. A diverged stage means two
people analysed independently and both results are real; a pointer is a
decision, not a copy, and advancing one is a separate, explicit act this
module never performs on its own.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from ..informatics.generation_listing import GENERATIONS_SUBDIR, STAGE_GENERATION_DIRS
from .run_locality import (
    STATE_AHEAD,
    STATE_BEHIND,
    STATE_DIVERGED,
    STATE_POINTER_CONFLICT,
    compare_run_locations,
)

_DIVERGED_REASON = (
    "diverged: each location holds a generation the other lacks; sync is additive-only "
    "and will not pick a side -- resolve manually."
)
_POINTER_CONFLICT_REASON = (
    "pointer conflict: same generations, different current.json; advancing a pointer is a "
    "separate, explicit act, not something sync performs."
)


@dataclass(frozen=True)
class StageSyncResult:
    """What sync did (or refused to do) for one stage."""

    kind: str
    state: str
    #: Generation ids copied from `location_a` into `location_b`.
    copied_a_to_b: tuple[str, ...]
    #: Generation ids copied from `location_b` into `location_a`.
    copied_b_to_a: tuple[str, ...]
    #: Set (and nothing copied) for `diverged`/`pointer_conflict`.
    skipped_reason: Optional[str]


@dataclass(frozen=True)
class SyncResult:
    """Every stage's sync outcome between two locations of one run."""

    location_a: Path
    location_b: Path
    stages: tuple[StageSyncResult, ...]

    @property
    def any_copied(self) -> bool:
        return any(stage.copied_a_to_b or stage.copied_b_to_a for stage in self.stages)

    @property
    def unresolved_stages(self) -> tuple[str, ...]:
        return tuple(stage.kind for stage in self.stages if stage.skipped_reason is not None)


def _generation_dir(run_root: Path, stage_dir: str, generation_id: str) -> Path:
    return run_root / stage_dir / GENERATIONS_SUBDIR / generation_id


def _copy_generation(
    source_root: Path, dest_root: Path, stage_dir: str, generation_id: str
) -> None:
    """Copy one generation directory, publishing it via stage-then-rename.

    A destination that already exists is left untouched -- generations are
    content-addressed, so nothing to reconcile -- which is what makes a
    re-run after an interrupted copy safe: the destination only ever appears
    once fully copied, so an interrupted attempt is retried whole rather than
    silently completed with a partial directory.
    """
    source = _generation_dir(source_root, stage_dir, generation_id)
    dest = _generation_dir(dest_root, stage_dir, generation_id)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{generation_id}.syncing-{uuid4().hex}"
    shutil.copytree(source, staging)
    staging.rename(dest)


def sync_run_locations(
    location_a: str | Path, location_b: str | Path, *, dry_run: bool = False
) -> SyncResult:
    """Additively sync every stage between two locations of the same run.

    Does not check that `location_a`/`location_b` are actually the same run
    -- see `smftools.data.run_locality.are_duplicates` for that; callers
    combine the two rather than this function silently refusing to run.

    Args:
        location_a: One location's run root.
        location_b: The other location's run root.
        dry_run: Classify and report without copying anything.

    Returns:
        SyncResult: Per-stage outcome. `identical` stages copy nothing;
        `ahead`/`behind` stages copy the missing generations in the
        direction that fills the gap; `diverged`/`pointer_conflict` stages
        copy nothing and carry a `skipped_reason`.
    """
    location_a = Path(location_a)
    location_b = Path(location_b)
    comparison = compare_run_locations(location_a, location_b)

    results: list[StageSyncResult] = []
    for stage in comparison.stages:
        stage_dir = STAGE_GENERATION_DIRS[stage.kind]

        if stage.state == STATE_DIVERGED:
            results.append(
                StageSyncResult(
                    kind=stage.kind,
                    state=stage.state,
                    copied_a_to_b=(),
                    copied_b_to_a=(),
                    skipped_reason=_DIVERGED_REASON,
                )
            )
            continue
        if stage.state == STATE_POINTER_CONFLICT:
            results.append(
                StageSyncResult(
                    kind=stage.kind,
                    state=stage.state,
                    copied_a_to_b=(),
                    copied_b_to_a=(),
                    skipped_reason=_POINTER_CONFLICT_REASON,
                )
            )
            continue

        copied_a_to_b: tuple[str, ...] = ()
        copied_b_to_a: tuple[str, ...] = ()
        if stage.state == STATE_AHEAD:
            if not dry_run:
                for generation_id in stage.a_only:
                    _copy_generation(location_a, location_b, stage_dir, generation_id)
            copied_a_to_b = stage.a_only
        elif stage.state == STATE_BEHIND:
            if not dry_run:
                for generation_id in stage.b_only:
                    _copy_generation(location_b, location_a, stage_dir, generation_id)
            copied_b_to_a = stage.b_only

        results.append(
            StageSyncResult(
                kind=stage.kind,
                state=stage.state,
                copied_a_to_b=copied_a_to_b,
                copied_b_to_a=copied_b_to_a,
                skipped_reason=None,
            )
        )

    return SyncResult(location_a=location_a, location_b=location_b, stages=tuple(results))
