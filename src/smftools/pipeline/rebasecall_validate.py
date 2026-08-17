"""Validate a published lineage, and promote only what validates.

Registry status is a *claim*; this module is the *check*. A lineage recorded as
complete may still be missing a stage generation, carry a transition report that
does not reconcile, or reference a basecall whose bytes have changed. Promotion
runs the check first, so a user cannot make an incomplete lineage the answer to
their project by accident.

Replayability is reported separately from completeness. A lineage can be
complete and still not replayable -- its filtered signal was never materialized
and its source POD5s have moved on -- and conflating the two would either block
ordinary promotion or quietly overstate what can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..project.registry import LineageSelectionError, set_active_lineage
from .rebasecall_lineage import (
    PublishedRebasecallLineage,
    RebasecallLineageError,
    read_published_rebasecall_lineage,
    write_lineage_validation,
)

REBASECALL_VALIDATION_SCHEMA_VERSION = 1

_STAGE_OUTPUT_DIRS = {
    "raw": "raw_outputs",
    "preprocess": "preprocess_adata_outputs",
    "spatial": "spatial_adata_outputs",
    "hmm": "hmm_adata_outputs",
    "latent": "latent_adata_outputs",
}


@dataclass(frozen=True)
class LineageCheck:
    """One named validation result."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class LineageValidationReport:
    """The machine-readable summary written beside a validated lineage."""

    lineage_id: str
    complete: bool
    replayable: bool | None
    replay_required: bool = False
    checks: tuple[LineageCheck, ...] = field(default_factory=tuple)
    schema_version: int = REBASECALL_VALIDATION_SCHEMA_VERSION

    @property
    def failures(self) -> tuple[LineageCheck, ...]:
        """Checks that count against completeness.

        A lineage that is complete but not replayable has no failures: the
        replay check is informational unless it was required. Reporting it as a
        failure anyway would make a passing report contradict itself, and would
        put ``replayable`` in a refusal message that was never about replay.
        """
        return tuple(
            check
            for check in self.checks
            if not check.passed and (check.name != "replayable" or self.replay_required)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lineage_id": self.lineage_id,
            "complete": self.complete,
            "replayable": self.replayable,
            "replay_required": self.replay_required,
            "checks": [check.to_dict() for check in self.checks],
            "failures": [check.name for check in self.failures],
        }


def _check(name: str, passed: bool, detail: str) -> LineageCheck:
    return LineageCheck(name=name, passed=passed, detail=detail)


def _validate_stage_generations(
    lineage: PublishedRebasecallLineage,
    run_root: Path,
) -> list[LineageCheck]:
    """Every generation the lineage claims must exist and validate on its own."""
    from ..informatics.generation import GenerationError, resolve_stage_generation

    checks: list[LineageCheck] = []
    for stage, generation_id in sorted(lineage.stage_generations.items()):
        directory = _STAGE_OUTPUT_DIRS.get(stage)
        if directory is None:
            checks.append(_check(f"stage:{stage}", False, f"unknown lineage stage {stage!r}"))
            continue
        try:
            resolved = resolve_stage_generation(run_root / directory, generation_id)
        except GenerationError as exc:
            checks.append(_check(f"stage:{stage}", False, str(exc)))
            continue
        if resolved is None:
            checks.append(
                _check(f"stage:{stage}", False, f"generation {generation_id!r} did not resolve")
            )
            continue
        checks.append(
            _check(f"stage:{stage}", True, f"generation {generation_id} resolves and validates")
        )
    return checks


def _validate_transition(lineage: PublishedRebasecallLineage) -> list[LineageCheck]:
    from .rebasecall_transition import read_qc_transition, reconcile_qc_transition

    try:
        frame, summary = read_qc_transition(lineage)
    except RebasecallLineageError as exc:
        return [_check("qc_transition", False, str(exc))]
    report = reconcile_qc_transition(frame, summary)
    if not report["reconciled"]:
        return [
            _check(
                "qc_transition",
                False,
                f"published counts do not reconcile: {sorted(report['disagreements'])}",
            )
        ]
    return [
        _check(
            "qc_transition",
            True,
            f"{len(frame)} selected molecule(s) reconciled against published counts",
        )
    ]


def _validate_basecall(
    lineage: PublishedRebasecallLineage,
    basecall_root: Path | None,
) -> list[LineageCheck]:
    if basecall_root is None:
        return [
            _check(
                "basecall",
                False,
                "no basecall root supplied, so the lineage's basecall could not be revalidated",
            )
        ]
    from .rebasecall_basecall import read_published_rebasecall_basecall

    directory = Path(basecall_root) / lineage.basecall_id
    try:
        published = read_published_rebasecall_basecall(
            directory,
            expected_basecall_id=lineage.basecall_id,
        )
    except Exception as exc:
        return [_check("basecall", False, f"{type(exc).__name__}: {exc}")]
    return [
        _check(
            "basecall",
            True,
            f"basecall {published.basecall_id} revalidates ({published.generation_kind})",
        )
    ]


def _validate_replayable(
    lineage: PublishedRebasecallLineage,
    basecall_root: Path | None,
    signal_root: Path | None,
) -> tuple[bool, LineageCheck]:
    """Replayable means the signal this lineage was built from is still provable.

    Filtered signal artifacts are self-contained, so they settle it. Original
    POD5s are only evidence when a resolution recorded their checksums, which
    lives with the basecall rather than the lineage.
    """
    signal_id = None
    if basecall_root is not None:
        manifest_path = Path(basecall_root) / lineage.basecall_id / "basecall_manifest.json"
        if manifest_path.is_file():
            import json

            try:
                signal_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("signal_id")
            except (OSError, ValueError):
                signal_id = None
    if signal_id is None:
        return False, _check(
            "replayable",
            False,
            "no filtered signal artifact was materialized, so replay depends on source POD5s "
            "that this lineage cannot vouch for",
        )
    if signal_root is None:
        return False, _check(
            "replayable",
            False,
            f"signal {signal_id} was materialized but no signal root was supplied to check it",
        )
    from .rebasecall_signal import read_materialized_rebasecall_signal

    try:
        read_materialized_rebasecall_signal(
            Path(signal_root) / str(signal_id),
            expected_signal_id=str(signal_id),
        )
    except Exception as exc:
        return False, _check("replayable", False, f"{type(exc).__name__}: {exc}")
    return True, _check(
        "replayable",
        True,
        f"filtered signal {signal_id} revalidates, so this lineage can be replayed",
    )


def validate_rebasecall_lineage(
    lineage_dir: str | Path,
    run_root: str | Path,
    *,
    basecall_root: str | Path | None = None,
    signal_root: str | Path | None = None,
    require_replayable: bool = False,
    write_report: bool = True,
) -> LineageValidationReport:
    """Check a published lineage end to end and record the result beside it.

    ``require_replayable`` folds replayability into completeness, for the case
    where a lineage is being kept as the reproducible record of a publication
    rather than merely as a result.
    """
    lineage_dir = Path(lineage_dir)
    run_root = Path(run_root)
    try:
        lineage = read_published_rebasecall_lineage(lineage_dir)
    except RebasecallLineageError as exc:
        return LineageValidationReport(
            lineage_id=lineage_dir.name,
            complete=False,
            replayable=None,
            checks=(_check("manifest", False, str(exc)),),
        )

    checks: list[LineageCheck] = [_check("manifest", True, "lineage manifest validates")]
    checks.extend(_validate_stage_generations(lineage, run_root))
    checks.extend(_validate_transition(lineage))
    checks.extend(
        _validate_basecall(
            lineage,
            None if basecall_root is None else Path(basecall_root),
        )
    )
    replayable, replay_check = _validate_replayable(
        lineage,
        None if basecall_root is None else Path(basecall_root),
        None if signal_root is None else Path(signal_root),
    )
    checks.append(replay_check)

    # Replayability is reported always and required only on request, so an
    # ordinary lineage is not blocked for lacking materialized signal.
    complete = all(
        check.passed for check in checks if check.name != "replayable" or require_replayable
    )
    report = LineageValidationReport(
        lineage_id=lineage.lineage_id,
        complete=complete,
        replayable=replayable,
        replay_required=require_replayable,
        checks=tuple(checks),
    )
    if write_report:
        write_lineage_validation(lineage, report.to_dict())
    return report


def promote_rebasecall_lineage(
    project_dir: str | Path,
    experiment_id: str,
    lineage_id: str,
    *,
    lineage_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    basecall_root: str | Path | None = None,
    signal_root: str | Path | None = None,
    require_replayable: bool = False,
    validator: Callable[..., LineageValidationReport] | None = None,
) -> Mapping[str, Any]:
    """Validate a lineage, then make it the experiment's answer.

    Validation comes first and its failure is the refusal, so a user cannot
    activate an incomplete lineage by accident. Rollback needs no separate
    machinery: promoting a prior complete lineage -- including ``original`` --
    is the same operation.
    """
    from ..project.registry import ORIGINAL_LINEAGE

    lineage_id = str(lineage_id)
    report: LineageValidationReport | None = None
    if lineage_id != ORIGINAL_LINEAGE:
        if lineage_dir is None or run_root is None:
            raise RebasecallLineageError(
                "promotion_unverifiable",
                "promoting a descendant requires its lineage directory and run root so the "
                "lineage can be validated before it becomes the answer",
            )
        validator = validator or validate_rebasecall_lineage
        report = validator(
            lineage_dir,
            run_root,
            basecall_root=basecall_root,
            signal_root=signal_root,
            require_replayable=require_replayable,
        )
        if not report.complete:
            raise RebasecallLineageError(
                "lineage_incomplete",
                "refusing to promote an incomplete lineage; failed checks: "
                f"{[check.name for check in report.failures]}",
            )
    try:
        active = set_active_lineage(project_dir, experiment_id, lineage_id)
    except LineageSelectionError as exc:
        raise RebasecallLineageError("promotion_refused", str(exc)) from exc
    return {
        "experiment_id": str(experiment_id),
        "active_lineage": active,
        "validation": None if report is None else report.to_dict(),
    }
