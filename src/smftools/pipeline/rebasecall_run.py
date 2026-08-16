"""Run a descendant raw stage beneath a staged re-basecalling lineage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .rebasecall_basecall import BASECALL_BAM_FILENAME, PublishedRebasecallBasecall
from .rebasecall_lineage import (
    PublishedRebasecallLineage,
    RebasecallLineageError,
    build_lineage_identity,
    descendant_raw_provenance,
    read_published_rebasecall_lineage,
    staged_lineage,
)
from .rebasecall_plan import RebasecallPlan
from .rebasecall_selection import FrozenRebasecallSelection

DESCENDANT_CONFIG_FILENAME = "descendant_config.csv"

# Fields the descendant overrides. Everything else is inherited verbatim: a
# lineage re-runs the *same* experiment against new calls, so silently changing
# a reference or a threshold here would make the comparison meaningless.
_OVERRIDDEN_CONFIG_FIELDS = ("input_data_path", "input_manifest_path", "alignment_mode")


@dataclass(frozen=True)
class LineageRawStageResult:
    """The workflow-contract result of running one lineage's raw stage."""

    lineage: PublishedRebasecallLineage
    raw_generation_id: str
    descendant_config_path: Path
    run_root: Path

    def to_dict(self) -> dict[str, Any]:
        """Return the engine-facing result payload."""
        return {
            "lineage_id": self.lineage.lineage_id,
            "basecall_id": self.lineage.basecall_id,
            "stage_generations": self.lineage.stage_generations,
            "raw_generation_id": self.raw_generation_id,
            "run_root": str(self.run_root),
            "descendant_config_path": str(self.descendant_config_path),
        }


def _read_config_rows(config_path: Path) -> list[list[str]]:
    try:
        with config_path.open(newline="", encoding="utf-8-sig") as handle:
            return [row for row in csv.reader(handle) if row]
    except OSError as exc:
        raise RebasecallLineageError(
            "lineage_config_unreadable",
            f"the parent experiment config {config_path} could not be read",
        ) from exc


def derive_descendant_config(
    parent_config_path: str | Path,
    basecall: PublishedRebasecallBasecall,
    destination: str | Path,
) -> Path:
    """Write the descendant's config: the parent's, reading the new calls.

    ``output_directory`` is deliberately *not* overridden. The descendant
    publishes into the parent experiment's ordinary stage directories, beside
    the parent's generation, because a lineage is a map ``stage -> generation
    id`` rather than a second nested run tree.
    """
    parent_config_path = Path(parent_config_path)
    destination = Path(destination)
    rows = _read_config_rows(parent_config_path)
    if not rows:
        raise RebasecallLineageError(
            "lineage_config_unreadable",
            f"the parent experiment config {parent_config_path} is empty",
        )
    calls_path = basecall.directory / BASECALL_BAM_FILENAME
    if not calls_path.is_file():
        raise RebasecallLineageError(
            "lineage_basecall_missing",
            "the published basecall has no calls BAM to ingest",
        )

    header, *body = rows
    width = len(header)
    kept = [row for row in body if row and row[0] not in _OVERRIDDEN_CONFIG_FIELDS]

    def _row(variable: str, value: str) -> list[str]:
        row = [variable, value]
        row.extend([""] * max(0, width - len(row)))
        return row

    kept.append(_row("input_data_path", str(calls_path)))
    kept.append(_row("alignment_mode", "align"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(kept)
    return destination


def run_lineage_raw_stage(
    plan: RebasecallPlan,
    selection: FrozenRebasecallSelection,
    basecall: PublishedRebasecallBasecall,
    rebasecall_root: str | Path,
    *,
    accepted_plan_id: str,
    parent_config_path: str | Path,
    raw_stage_runner: Callable[..., Any] | None = None,
    identity_map: str | None = None,
) -> LineageRawStageResult:
    """Publish one lineage whose descendant raw generation was actually built.

    The raw stage runs inside the lineage transaction, so a stage killed partway
    leaves the parent run and every prior complete lineage untouched: the
    descendant generation is never selected, and the lineage never appears.
    """
    if raw_stage_runner is None:
        from ..cli.raw_adata import raw_adata as raw_stage_runner  # noqa: PLC0415

    rebasecall_root = Path(rebasecall_root)
    identity = build_lineage_identity(plan, selection, basecall)
    run_root = Path(plan.run_root)

    with staged_lineage(
        plan,
        selection,
        basecall,
        rebasecall_root,
        accepted_plan_id=accepted_plan_id,
    ) as staged:
        descendant_config = derive_descendant_config(
            parent_config_path,
            basecall,
            staged.staging_dir / DESCENDANT_CONFIG_FILENAME,
        )
        provenance = descendant_raw_provenance(
            staged.lineage_id,
            identity,
            basecall,
            identity_map=identity_map,
        )
        result = raw_stage_runner(str(descendant_config), lineage_provenance=provenance)
        generation_id = _descendant_generation_id(result)
        staged.record_stage_generation("raw", generation_id)
        final_dir = staged.final_dir

    lineage = read_published_rebasecall_lineage(final_dir, expected_lineage_id=staged.lineage_id)
    return LineageRawStageResult(
        lineage=lineage,
        raw_generation_id=generation_id,
        descendant_config_path=lineage.directory / DESCENDANT_CONFIG_FILENAME,
        run_root=run_root,
    )


def _descendant_generation_id(result: Any) -> str:
    """Recover the published generation id from a raw-stage result."""
    if isinstance(result, Mapping) and "generation_id" in result:
        return str(result["generation_id"])
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        generation_spine = Path(str(result[1]))
        # ``<raw_outputs>/generations/<id>/spine.h5ad``
        if generation_spine.name == "spine.h5ad" and generation_spine.parent.parent.name == (
            "generations"
        ):
            return generation_spine.parent.name
    raise RebasecallLineageError(
        "lineage_raw_stage_unrecognized",
        "the raw stage did not report a published generation",
    )
