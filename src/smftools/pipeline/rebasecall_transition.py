"""Reconcile every selected origin molecule against its re-basecalled result.

Parent selection and refreshed QC are separate facts. No parent QC result is
copied forward: the descendant lineage recomputes QC and dedup against the new
calls, and this report is where the two are compared without either being
mistaken for the other.

One row per selected origin molecule, so "every selected molecule is accounted
for" is literally checkable: the row count equals the frozen selection's row
count, and each terminal status explains where that molecule ended up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .rebasecall_basecall import BASECALL_ORIGIN_FILENAME, PublishedRebasecallBasecall
from .rebasecall_lineage import PublishedRebasecallLineage, RebasecallLineageError
from .rebasecall_selection import FrozenRebasecallSelection

REBASECALL_TRANSITION_SCHEMA_VERSION = 1
QC_TRANSITION_FILENAME = "qc_transition.parquet"
QC_TRANSITION_SUMMARY_FILENAME = "qc_transition_summary.json"

TRANSITION_COLUMNS = (
    "molecule_uid",
    "pod5_read_id",
    "pod5_source_id",
    "selected_by_parent",
    "source_signal_resolved",
    "basecall_read_ids",
    "basecall_output_count",
    "new_molecule_uids",
    "new_molecule_count",
    "passes_read_qc",
    "passes_modification_qc",
    "passes_variant_qc",
    "passes_qc",
    "is_duplicate",
    "passes_dedup",
    "terminal_status",
    "terminal_reason",
)

# Terminal statuses are ordered by how far a molecule got, so a reader can see
# where a cohort was lost without joining anything back together.
TERMINAL_STATUSES = (
    "no_signal",
    "no_call",
    "dropped_in_raw",
    "failed_qc",
    "duplicate",
    "passed",
    "qc_not_run",
)

_QC_COLUMNS = (
    "passes_read_qc",
    "passes_modification_qc",
    "passes_variant_qc",
    "passes_qc",
    "is_duplicate",
    "passes_dedup",
)


@dataclass(frozen=True)
class QcTransitionSummary:
    """Reconciled counts for one lineage, reproducible from the table itself."""

    schema_version: int
    selected_molecule_count: int
    signal_resolved_count: int
    basecalled_molecule_count: int
    basecall_output_count: int
    new_molecule_count: int
    passes_qc_count: int
    duplicate_count: int
    passes_dedup_count: int
    terminal_status_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_molecule_count": self.selected_molecule_count,
            "signal_resolved_count": self.signal_resolved_count,
            "basecalled_molecule_count": self.basecalled_molecule_count,
            "basecall_output_count": self.basecall_output_count,
            "new_molecule_count": self.new_molecule_count,
            "passes_qc_count": self.passes_qc_count,
            "duplicate_count": self.duplicate_count,
            "passes_dedup_count": self.passes_dedup_count,
            "terminal_status_counts": dict(self.terminal_status_counts),
        }


def _read_parquet(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    except Exception as exc:
        raise RebasecallLineageError(
            "transition_source_unreadable",
            f"transition input {path.name!r} could not be read: {type(exc).__name__}: {exc}",
        ) from exc


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RebasecallLineageError(
            "transition_source_unreadable",
            f"transition input {path.name!r} could not be read: {type(exc).__name__}: {exc}",
        ) from exc


def _optional_flag(row: Mapping[str, Any], column: str) -> bool | None:
    value = row.get(column)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)


def _joined(values: list[str]) -> str:
    return ",".join(sorted(set(values)))


def _true_count(frame: pd.DataFrame, column: str) -> int:
    """Count exactly-true values, leaving ``None`` as "not known" rather than false."""
    return 0 if not len(frame) else int(frame[column].eq(True).sum())


def _counts_from_table(frame: pd.DataFrame) -> dict[str, int]:
    """Derive every published count from the table alone.

    Build and reconcile share this, so the exit-gate check cannot pass by
    recomputing counts a different way than the writer did.
    """
    return {
        "selected_molecule_count": len(frame),
        "signal_resolved_count": _true_count(frame, "source_signal_resolved"),
        "basecalled_molecule_count": (
            0 if not len(frame) else int((frame["basecall_output_count"] > 0).sum())
        ),
        "basecall_output_count": (
            0 if not len(frame) else int(frame["basecall_output_count"].sum())
        ),
        "new_molecule_count": 0 if not len(frame) else int(frame["new_molecule_count"].sum()),
        "passes_qc_count": _true_count(frame, "passes_qc"),
        "duplicate_count": _true_count(frame, "is_duplicate"),
        "passes_dedup_count": _true_count(frame, "passes_dedup"),
    }


def _status_counts_from_table(frame: pd.DataFrame) -> dict[str, int]:
    return {
        status: int((frame["terminal_status"] == status).sum())
        for status in TERMINAL_STATUSES
        if len(frame) and int((frame["terminal_status"] == status).sum())
    }


def _terminal(
    *,
    resolved: bool,
    basecall_count: int,
    molecule_count: int,
    qc: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if not resolved:
        return "no_signal", "the selected POD5 read was not resolved to source signal"
    if basecall_count == 0:
        return "no_call", "the basecaller produced no output for this read"
    if molecule_count == 0:
        return "dropped_in_raw", "the new calls produced no descendant raw molecule"
    if qc is None:
        return "qc_not_run", "the lineage stopped before preprocess, so QC was not recomputed"
    if _optional_flag(qc, "passes_qc") is False:
        return "failed_qc", "the descendant molecule did not pass recomputed QC"
    if _optional_flag(qc, "is_duplicate") is True:
        return "duplicate", "the descendant molecule was marked a duplicate"
    if _optional_flag(qc, "passes_dedup") is False:
        return "failed_qc", "the descendant molecule did not pass recomputed dedup"
    return "passed", "the descendant molecule passed recomputed QC and dedup"


def build_qc_transition(
    selection: FrozenRebasecallSelection,
    basecall: PublishedRebasecallBasecall,
    raw_generation_dir: str | Path,
    preprocess_generation_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, QcTransitionSummary]:
    """Reconcile the frozen selection against the descendant's published artifacts.

    Every selected molecule yields exactly one row, whether or not it survived,
    which is what makes the terminal-status counts a reconciliation rather than a
    summary of the survivors.
    """
    selected = _read_parquet(
        Path(selection.rows_path),
        ("molecule_uid", "pod5_read_id", "pod5_source_id"),
    )
    origin = _read_csv(Path(basecall.directory) / BASECALL_ORIGIN_FILENAME)
    raw_obs = _read_parquet(Path(raw_generation_dir) / "obs.parquet")

    calls_by_pod5: dict[str, list[str]] = {}
    for pod5_read_id, read_id in zip(
        origin.get("pod5_read_id", pd.Series(dtype=str)),
        origin.get("read_id", pd.Series(dtype=str)),
        strict=False,
    ):
        calls_by_pod5.setdefault(str(pod5_read_id), []).append(str(read_id))

    molecules_by_read: dict[str, list[str]] = {}
    if "read_id" in raw_obs.columns:
        molecule_column = "molecule_uid" if "molecule_uid" in raw_obs.columns else "read_id"
        for read_id, molecule_uid in zip(
            raw_obs["read_id"], raw_obs[molecule_column], strict=False
        ):
            molecules_by_read.setdefault(str(read_id), []).append(str(molecule_uid))

    qc_by_read: dict[str, dict[str, Any]] = {}
    if preprocess_generation_dir is not None:
        stage_obs = _read_parquet(Path(preprocess_generation_dir) / "stage_obs.parquet")
        available = [column for column in _QC_COLUMNS if column in stage_obs.columns]
        if "read_id" in stage_obs.columns:
            for record in stage_obs[["read_id", *available]].to_dict("records"):
                qc_by_read[str(record["read_id"])] = record

    rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        pod5_read_id = str(record["pod5_read_id"])
        resolved = bool(pod5_read_id) and pod5_read_id.lower() != "nan"
        call_ids = calls_by_pod5.get(pod5_read_id, [])
        molecule_ids = [
            molecule_uid
            for call_id in call_ids
            for molecule_uid in molecules_by_read.get(call_id, [])
        ]
        qc = next((qc_by_read[call_id] for call_id in call_ids if call_id in qc_by_read), None)
        status, reason = _terminal(
            resolved=resolved,
            basecall_count=len(call_ids),
            molecule_count=len(molecule_ids),
            qc=qc if preprocess_generation_dir is not None else None,
        )
        rows.append(
            {
                "molecule_uid": record.get("molecule_uid"),
                "pod5_read_id": pod5_read_id,
                "pod5_source_id": record.get("pod5_source_id"),
                "selected_by_parent": True,
                "source_signal_resolved": resolved,
                "basecall_read_ids": _joined(call_ids),
                "basecall_output_count": len(call_ids),
                "new_molecule_uids": _joined(molecule_ids),
                "new_molecule_count": len(molecule_ids),
                **{
                    column: (None if qc is None else _optional_flag(qc, column))
                    for column in _QC_COLUMNS
                },
                "terminal_status": status,
                "terminal_reason": reason,
            }
        )

    frame = pd.DataFrame(rows, columns=list(TRANSITION_COLUMNS))
    counts = _counts_from_table(frame)
    summary = QcTransitionSummary(
        schema_version=REBASECALL_TRANSITION_SCHEMA_VERSION,
        terminal_status_counts=_status_counts_from_table(frame),
        **counts,
    )
    return frame, summary


def write_qc_transition(
    lineage: PublishedRebasecallLineage,
    frame: pd.DataFrame,
    summary: QcTransitionSummary,
) -> tuple[Path, Path]:
    """Write the transition table beside a published lineage.

    The report is recorded after publication and outside lineage identity:
    recomputing it must not change what the lineage *is*.
    """
    table_path = lineage.directory / QC_TRANSITION_FILENAME
    summary_path = lineage.directory / QC_TRANSITION_SUMMARY_FILENAME
    frame.to_parquet(table_path, index=False)
    summary_path.write_text(
        json.dumps(summary.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return table_path, summary_path


def read_qc_transition(
    lineage: PublishedRebasecallLineage,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a lineage's transition table and its recorded summary."""
    table_path = lineage.directory / QC_TRANSITION_FILENAME
    summary_path = lineage.directory / QC_TRANSITION_SUMMARY_FILENAME
    if not table_path.is_file() or not summary_path.is_file():
        raise RebasecallLineageError(
            "transition_report_missing",
            "the lineage has no published QC transition report",
        )
    frame = _read_parquet(table_path)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebasecallLineageError(
            "transition_report_invalid",
            "the lineage QC transition summary is unreadable",
        ) from exc
    return frame, summary


def reconcile_qc_transition(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every summary count from the table and report disagreement.

    This is the exit-gate check: the published counts must be reproducible from
    the published table alone, with no access to the run that produced them.
    """
    recomputed = _counts_from_table(frame)
    disagreements = {
        key: {"published": summary.get(key), "recomputed": value}
        for key, value in recomputed.items()
        if summary.get(key) != value
    }
    status_counts = _status_counts_from_table(frame)
    if dict(summary.get("terminal_status_counts") or {}) != status_counts:
        disagreements["terminal_status_counts"] = {
            "published": dict(summary.get("terminal_status_counts") or {}),
            "recomputed": status_counts,
        }
    # Every selected molecule must land in exactly one terminal status, or the
    # table is a summary of survivors rather than a reconciliation.
    if len(frame) and sum(status_counts.values()) != len(frame):
        disagreements["terminal_status_total"] = {
            "published": sum(status_counts.values()),
            "recomputed": len(frame),
        }
    return {
        "reconciled": not disagreements,
        "recomputed": recomputed,
        "terminal_status_counts": status_counts,
        "disagreements": disagreements,
    }
