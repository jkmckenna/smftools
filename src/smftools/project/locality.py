"""Whether a registered experiment can actually be read right now.

A project references experiments; it never copies them. Those run directories
routinely live somewhere other than the project — an external SSD, a second
machine's disk — so "registered" and "reachable" are different questions, and
until `PSR-18` nothing asked the second one.

The consequences were two-shaped. `ProjectCatalog`'s union methods
(``interval_catalog``, ``region_catalog``, ``reference_interval_map``) skip a
path that does not exist and return the union of whatever remains, so a detached
volume silently produces a smaller answer. Pooling fails instead, but late:
``iter_set_parts`` materializes each member in turn, so the failure arrives
mid-stream from deep inside a file open, after earlier parts have been yielded
and possibly written.

Both are the same defect as `F-PSR-01` at a different altitude, and take the same
fix: classify, then refuse the ambiguous case rather than guessing past it. A
selection that quietly drops experiments reports fewer molecules, which reads as
a biological result rather than a defect.

`offline` and `missing` mean here exactly what they mean for raw input, and share
the mount-root detection in :mod:`smftools.config.input_availability` so the two
layers cannot disagree about whether a volume is attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from smftools.config.input_availability import (
    INPUT_MISSING,
    INPUT_OFFLINE,
    INPUT_PRESENT,
    detached_volume_for,
)
from smftools.logging_utils import get_logger

logger = get_logger(__name__)

REACHABLE = INPUT_PRESENT
OFFLINE = INPUT_OFFLINE
MISSING = INPUT_MISSING


@dataclass(frozen=True)
class ExperimentLocality:
    """Where one registered experiment is, and whether it answers."""

    experiment: str
    path: Path
    state: str
    volume: Optional[Path] = None

    @property
    def is_reachable(self) -> bool:
        """Whether the experiment's files can be read right now."""
        return self.state == REACHABLE

    def describe(self) -> str:
        """One human-readable clause naming why this experiment does not answer."""
        if self.is_reachable:
            return f"{self.experiment}: reachable"
        if self.state == OFFLINE:
            return f"{self.experiment}: on detached volume {self.volume}"
        return f"{self.experiment}: path missing ({self.path})"


def resolve_experiment_locality(experiment: str, path: str | Path) -> ExperimentLocality:
    """Classify one registered experiment as reachable, offline, or missing.

    Args:
        experiment: The experiment id, used in messages.
        path: The experiment's resolved run directory.

    Returns:
        ExperimentLocality: The state, plus the detached volume when there is one.
    """
    resolved = Path(path)
    if resolved.exists():
        return ExperimentLocality(experiment=str(experiment), path=resolved, state=REACHABLE)
    volume = detached_volume_for(resolved)
    if volume is not None:
        return ExperimentLocality(
            experiment=str(experiment), path=resolved, state=OFFLINE, volume=volume
        )
    return ExperimentLocality(experiment=str(experiment), path=resolved, state=MISSING)


def locality_for_entries(entries: Iterable[dict]) -> dict[str, ExperimentLocality]:
    """Classify every entry from :func:`list_experiments` by id."""
    return {
        str(entry["id"]): resolve_experiment_locality(entry["id"], entry["path"])
        for entry in entries
    }


class UnreachableExperimentsError(RuntimeError):
    """A selection covers experiments whose files cannot be read.

    Raised rather than warned because the alternative is a pooled or unioned
    answer that is quietly short several experiments, which is indistinguishable
    from a real result.
    """

    def __init__(self, localities: list[ExperimentLocality], *, operation: str):
        self.localities = list(localities)
        detail = "; ".join(item.describe() for item in self.localities)
        volumes = sorted({str(item.volume) for item in self.localities if item.volume})
        remedy = (
            f" Attach {', '.join(volumes)} and retry."
            if volumes
            else " Re-register or restore these experiments."
        )
        super().__init__(
            f"{operation} covers {len(self.localities)} unreachable experiment(s): "
            f"{detail}.{remedy} Pass allow_unreachable=True to proceed with a "
            "selection that is explicitly recorded as partial."
        )


def require_reachable(
    entries: Iterable[dict],
    *,
    operation: str,
    allow_unreachable: bool = False,
) -> list[ExperimentLocality]:
    """Refuse an operation whose selection includes unreachable experiments.

    Args:
        entries: Entries from :func:`list_experiments` that the operation covers.
        operation: Operation name, used in the error.
        allow_unreachable: Proceed anyway, returning what was excluded so the
            caller can record the answer as partial. The caller is then
            responsible for labelling it; silently dropping is what this exists
            to prevent.

    Returns:
        list[ExperimentLocality]: The unreachable members. Empty when all answer.

    Raises:
        UnreachableExperimentsError: If any member is unreachable and
            ``allow_unreachable`` is False.
    """
    unreachable = [item for item in locality_for_entries(entries).values() if not item.is_reachable]
    if not unreachable:
        return []
    if not allow_unreachable:
        raise UnreachableExperimentsError(unreachable, operation=operation)
    logger.warning(
        "%s proceeding without %d unreachable experiment(s): %s. The result is partial.",
        operation,
        len(unreachable),
        "; ".join(item.describe() for item in unreachable),
    )
    return unreachable
