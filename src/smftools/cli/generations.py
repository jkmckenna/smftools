"""CLI rendering for read-only generation inventories.

Presentation only; discovery lives in
:mod:`smftools.informatics.generation_listing`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..informatics.generation_listing import (
    STATE_OK,
    GenerationRecord,
    list_experiment_generations,
    list_project_generations,
)

_COLUMNS = ("", "KIND", "GENERATION", "STATE", "MODIFIED", "ARTIFACTS", "SIZE", "CONTAINER")


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}T"


def _short_timestamp(record: GenerationRecord) -> str:
    stamp = record.created_at or record.modified_at
    return stamp[:19] if stamp else "-"


def _rows(records: Iterable[GenerationRecord]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for record in records:
        rows.append(
            (
                "*" if record.is_current else " ",
                record.kind,
                record.generation_id,
                record.state,
                _short_timestamp(record),
                "-" if record.artifact_count is None else str(record.artifact_count),
                _human_bytes(record.size_bytes),
                record.container,
            )
        )
    return rows


def render_table(records: list[GenerationRecord]) -> str:
    """Render an aligned table plus a footer for any defects found."""
    if not records:
        return "No published generations found."

    rows = [_COLUMNS] + _rows(records)
    widths = [max(len(row[i]) for row in rows) for i in range(len(_COLUMNS))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows
    ]

    flagged = [record for record in records if record.issues]
    if flagged:
        lines.append("")
        lines.append(f"{len(flagged)} generation(s) with issues:")
        for record in flagged:
            for issue in record.issues:
                lines.append(f"  {record.kind}/{record.generation_id}: {issue}")

    current = sum(1 for record in records if record.is_current)
    healthy = sum(1 for record in records if record.state == STATE_OK)
    lines.append("")
    lines.append(
        f"{len(records)} generation(s); {current} current, "
        f"{healthy} readable, {len(records) - healthy} unreadable or missing."
    )
    return "\n".join(lines)


def render_json(records: list[GenerationRecord]) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generations": [record.to_dict() for record in records],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), indent=2)


def experiment_generations(
    output_root: str | Path,
    *,
    include_size: bool = False,
) -> list[GenerationRecord]:
    """Inventory one experiment output root."""
    return list_experiment_generations(output_root, include_size=include_size)


def project_generations(
    project_dir: str | Path,
    *,
    include_size: bool = False,
    include_experiments: bool = True,
) -> list[GenerationRecord]:
    """Inventory project-owned generations, optionally fanning out to experiments.

    An experiment whose registered path is unreachable (an unmounted volume, a
    moved tree) is skipped rather than raising: a partial inventory is the
    useful answer, and the missing rows are visible by comparison with
    ``project list``.
    """
    records = list_project_generations(project_dir, include_size=include_size)
    if not include_experiments:
        return records

    from .project_cmd import project_list

    experiments, _ = project_list(project_dir)
    for entry in experiments:
        path = Path(str(entry.get("path", "")))
        if not path.is_dir():
            continue
        records.extend(list_experiment_generations(path, include_size=include_size))
    return records
