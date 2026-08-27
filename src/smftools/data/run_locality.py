"""Per-run analysis locality: same run or not, and which copy leads (`PSR-17`).

Two on-disk copies of a run's *analysis* tree are not raw replicas
(interchangeable by checksum, `PSR-10`); each may legitimately hold different
generations, since analysis can happen independently at each location. "Any
attached one will do" is the wrong model here -- the analysis side needs one
authoritative location per run, with duplicates detected and their
relationship classified rather than silently picked between.

Two questions, answered separately:

- **Are these the same run at all?** (`are_duplicates`) -- decided by the
  durable `experiment_uid` persisted in `experiment_manifest.json` at raw
  ingestion (`smftools.informatics.molecule_identity.new_experiment_uid`),
  never a path or a human-chosen label, neither of which is stable across
  machines or reliable across a rename.
- **If so, what is their relationship, per stage?** (`compare_run_locations`)
  -- decided from the published `generations/` set and `current.json`
  pointer at each location, never modification time: `cp` doesn't preserve
  it, exFAT rounds to two seconds, and clocks drift between machines, so an
  mtime rule is least reliable in exactly the two-machine case that produces
  a real divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..informatics.experiment_manifest import read_experiment_manifest
from ..informatics.generation_listing import STATE_OK, list_experiment_generations
from ..informatics.molecule_identity import EXPERIMENT_UID_COLUMN, validate_experiment_uid

STATE_IDENTICAL = "identical"
STATE_AHEAD = "ahead"
STATE_BEHIND = "behind"
STATE_DIVERGED = "diverged"
STATE_POINTER_CONFLICT = "pointer_conflict"


def run_identity(run_root: str | Path) -> Optional[str]:
    """The durable `experiment_uid` recorded for `run_root`, or None.

    None covers both "no manifest yet" and "manifest predates the identity
    system" -- either way, nothing here can prove two locations are the same
    run, so callers must not guess.
    """
    raw = read_experiment_manifest(run_root).get(EXPERIMENT_UID_COLUMN)
    if raw is None:
        return None
    try:
        return validate_experiment_uid(raw)
    except ValueError:
        return None


def are_duplicates(location_a: str | Path, location_b: str | Path) -> bool:
    """Whether `location_a` and `location_b` are copies of the same run.

    False when either location's identity is unknown, not just when they
    disagree -- "no proof" and "proof they differ" are both reasons not to
    treat two locations as the same run, but only the second is worth
    distinguishing in a message, which `run_identity` lets a caller do
    directly.
    """
    identity_a = run_identity(location_a)
    identity_b = run_identity(location_b)
    return identity_a is not None and identity_a == identity_b


@dataclass(frozen=True)
class StageLocality:
    """One stage's generation-set relationship between two locations."""

    kind: str
    state: str
    #: Generation ids published at `location_a` but not `location_b`.
    a_only: tuple[str, ...]
    #: Generation ids published at `location_b` but not `location_a`.
    b_only: tuple[str, ...]
    a_current: Optional[str]
    b_current: Optional[str]


@dataclass(frozen=True)
class RunLocalityComparison:
    """Every stage's locality state between two copies of one run."""

    location_a: Path
    location_b: Path
    stages: tuple[StageLocality, ...]

    @property
    def diverged_stages(self) -> tuple[str, ...]:
        return tuple(stage.kind for stage in self.stages if stage.state == STATE_DIVERGED)

    @property
    def pointer_conflicts(self) -> tuple[str, ...]:
        return tuple(stage.kind for stage in self.stages if stage.state == STATE_POINTER_CONFLICT)

    @property
    def identical(self) -> bool:
        """Whether every stage present at either location matches exactly."""
        return all(stage.state == STATE_IDENTICAL for stage in self.stages)


def compare_run_locations(location_a: str | Path, location_b: str | Path) -> RunLocalityComparison:
    """Classify `location_a` vs. `location_b`, stage by stage.

    Does not check whether the two locations are actually the same run --
    see `are_duplicates` for that. Comparing two unrelated runs' generation
    sets is meaningless but harmless; callers combine the two rather than
    this function silently refusing to run.

    A stage neither location has published anything for contributes no
    `StageLocality` -- there is nothing to compare, not a fourth kind of
    disagreement.
    """
    records_a = list_experiment_generations(location_a)
    records_b = list_experiment_generations(location_b)
    kinds = sorted({record.kind for record in records_a} | {record.kind for record in records_b})

    stages = []
    for kind in kinds:
        ids_a = {r.generation_id for r in records_a if r.kind == kind and r.state == STATE_OK}
        ids_b = {r.generation_id for r in records_b if r.kind == kind and r.state == STATE_OK}
        if not ids_a and not ids_b:
            continue
        current_a = next(
            (r.generation_id for r in records_a if r.kind == kind and r.is_current), None
        )
        current_b = next(
            (r.generation_id for r in records_b if r.kind == kind and r.is_current), None
        )
        a_only = tuple(sorted(ids_a - ids_b))
        b_only = tuple(sorted(ids_b - ids_a))

        if a_only and b_only:
            state = STATE_DIVERGED
        elif a_only:
            state = STATE_AHEAD
        elif b_only:
            state = STATE_BEHIND
        elif current_a != current_b:
            state = STATE_POINTER_CONFLICT
        else:
            state = STATE_IDENTICAL

        stages.append(
            StageLocality(
                kind=kind,
                state=state,
                a_only=a_only,
                b_only=b_only,
                a_current=current_a,
                b_current=current_b,
            )
        )
    return RunLocalityComparison(
        location_a=Path(location_a), location_b=Path(location_b), stages=tuple(stages)
    )
