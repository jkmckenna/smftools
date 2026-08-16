from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.pipeline.test_rebasecall_basecall import _case, _execute, _FakeDorado

from smftools.pipeline.rebasecall_basecall import BASECALL_ORIGIN_FILENAME
from smftools.pipeline.rebasecall_lineage import (
    RebasecallLineageError,
    read_published_rebasecall_lineage,
    staged_lineage,
)
from smftools.pipeline.rebasecall_transition import (
    QC_TRANSITION_FILENAME,
    QC_TRANSITION_SUMMARY_FILENAME,
    TRANSITION_COLUMNS,
    build_qc_transition,
    read_qc_transition,
    reconcile_qc_transition,
    write_qc_transition,
)

pytestmark = pytest.mark.unit


def _lineage_case(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    basecall = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())
    return plan, frozen, basecall


def _write_raw_generation(tmp_path, rows):
    """Write a descendant raw generation's obs with the given read/molecule rows."""
    directory = tmp_path / "raw_outputs" / "generations" / "descendant-a"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["read_id", "molecule_uid"]).to_parquet(
        directory / "obs.parquet", index=False
    )
    return directory


def _write_preprocess_generation(tmp_path, rows):
    directory = tmp_path / "preprocess_adata_outputs" / "generations" / "descendant-p"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "stage_obs.parquet", index=False)
    return directory


def _selected_pod5_ids(frozen):
    return sorted(pd.read_parquet(frozen.rows_path)["pod5_read_id"].astype(str))


def test_every_selected_molecule_gets_exactly_one_row(tmp_path, monkeypatch):
    _, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    raw_dir = _write_raw_generation(
        tmp_path,
        [(read_id, f"m-{read_id}") for read_id in selected],
    )

    frame, summary = build_qc_transition(frozen, basecall, raw_dir)

    assert list(frame.columns) == list(TRANSITION_COLUMNS)
    assert len(frame) == len(selected)
    assert sorted(frame["pod5_read_id"]) == selected
    assert summary.selected_molecule_count == len(selected)
    assert summary.basecalled_molecule_count == len(selected)
    # No preprocess generation was supplied, so QC is explicitly not-run rather
    # than silently reported as passing.
    assert set(frame["terminal_status"]) == {"qc_not_run"}
    assert frame["passes_qc"].isna().all()


def test_qc_and_dedup_outcomes_are_reported_per_origin(tmp_path, monkeypatch):
    _, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    raw_dir = _write_raw_generation(
        tmp_path,
        [(read_id, f"m-{read_id}") for read_id in selected],
    )
    preprocess_dir = _write_preprocess_generation(
        tmp_path,
        {
            "read_id": selected,
            "passes_read_qc": [True, True, False],
            "passes_qc": [True, True, False],
            "is_duplicate": [False, True, False],
            "passes_dedup": [True, False, False],
        },
    )

    frame, summary = build_qc_transition(frozen, basecall, raw_dir, preprocess_dir)

    statuses = dict(zip(frame["pod5_read_id"], frame["terminal_status"], strict=True))
    assert statuses[selected[0]] == "passed"
    assert statuses[selected[1]] == "duplicate"
    assert statuses[selected[2]] == "failed_qc"
    assert summary.passes_qc_count == 2
    assert summary.duplicate_count == 1
    assert summary.passes_dedup_count == 1


def test_a_selected_read_with_no_call_is_reconciled_not_dropped(tmp_path, monkeypatch):
    _, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    # Remove one read's calls from the basecall origin map.
    origin_path = Path(basecall.directory) / BASECALL_ORIGIN_FILENAME
    origin = pd.read_csv(origin_path)
    origin[origin["pod5_read_id"] != selected[0]].to_csv(origin_path, index=False)
    raw_dir = _write_raw_generation(
        tmp_path,
        [(read_id, f"m-{read_id}") for read_id in selected[1:]],
    )

    frame, summary = build_qc_transition(frozen, basecall, raw_dir)

    row = frame[frame["pod5_read_id"] == selected[0]].iloc[0]
    assert row["terminal_status"] == "no_call"
    assert row["basecall_output_count"] == 0
    assert summary.selected_molecule_count == len(selected)
    assert summary.basecalled_molecule_count == len(selected) - 1


def test_a_call_that_produced_no_raw_molecule_is_reported(tmp_path, monkeypatch):
    _, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    raw_dir = _write_raw_generation(
        tmp_path,
        [(read_id, f"m-{read_id}") for read_id in selected[:-1]],
    )

    frame, _ = build_qc_transition(frozen, basecall, raw_dir)

    row = frame[frame["pod5_read_id"] == selected[-1]].iloc[0]
    assert row["terminal_status"] == "dropped_in_raw"
    assert row["basecall_output_count"] == 1
    assert row["new_molecule_count"] == 0


def test_split_children_are_aggregated_onto_their_origin(tmp_path, monkeypatch):
    plan, frozen, _ = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    split = _execute(
        plan, frozen, tmp_path / "split-basecalls", _FakeDorado(splits={selected[0]: 3})
    )
    raw_rows = []
    for read_id in selected:
        if read_id == selected[0]:
            raw_rows.extend(
                (f"{read_id}-{ordinal}", f"m-{read_id}-{ordinal}") for ordinal in range(3)
            )
        else:
            raw_rows.append((read_id, f"m-{read_id}"))
    raw_dir = _write_raw_generation(tmp_path, raw_rows)

    frame, summary = build_qc_transition(frozen, split, raw_dir)

    row = frame[frame["pod5_read_id"] == selected[0]].iloc[0]
    assert row["basecall_output_count"] == 3
    assert row["new_molecule_count"] == 3
    # One origin, three descendants: the origin is still a single reconciled row.
    assert len(frame) == len(selected)
    assert summary.new_molecule_count == len(selected) + 2


def test_published_counts_are_reproducible_from_the_table_alone(tmp_path, monkeypatch):
    """The exit gate: the report reconciles without the run that produced it."""
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    raw_dir = _write_raw_generation(
        tmp_path,
        [(read_id, f"m-{read_id}") for read_id in selected],
    )
    preprocess_dir = _write_preprocess_generation(
        tmp_path,
        {
            "read_id": selected,
            "passes_qc": [True, False, True],
            "is_duplicate": [False, False, True],
            "passes_dedup": [True, False, False],
        },
    )
    frame, summary = build_qc_transition(frozen, basecall, raw_dir, preprocess_dir)
    with staged_lineage(
        plan, frozen, basecall, tmp_path / "rebasecall_outputs", accepted_plan_id=plan.plan_id
    ) as staged:
        staged.record_stage_generation("raw", "descendant-a")
        final_dir = staged.final_dir
    lineage = read_published_rebasecall_lineage(final_dir)

    write_qc_transition(lineage, frame, summary)
    published_frame, published_summary = read_qc_transition(lineage)
    report = reconcile_qc_transition(published_frame, published_summary)

    assert report["reconciled"] is True
    assert report["disagreements"] == {}
    assert sum(report["terminal_status_counts"].values()) == len(selected)
    assert (lineage.directory / QC_TRANSITION_FILENAME).is_file()
    assert (lineage.directory / QC_TRANSITION_SUMMARY_FILENAME).is_file()


def test_a_tampered_summary_fails_reconciliation(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    selected = _selected_pod5_ids(frozen)
    raw_dir = _write_raw_generation(
        tmp_path,
        [(read_id, f"m-{read_id}") for read_id in selected],
    )
    frame, summary = build_qc_transition(frozen, basecall, raw_dir)
    with staged_lineage(
        plan, frozen, basecall, tmp_path / "rebasecall_outputs", accepted_plan_id=plan.plan_id
    ) as staged:
        staged.record_stage_generation("raw", "descendant-a")
        final_dir = staged.final_dir
    lineage = read_published_rebasecall_lineage(final_dir)
    write_qc_transition(lineage, frame, summary)

    summary_path = lineage.directory / QC_TRANSITION_SUMMARY_FILENAME
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["basecalled_molecule_count"] += 5
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    published_frame, published_summary = read_qc_transition(lineage)
    report = reconcile_qc_transition(published_frame, published_summary)

    assert report["reconciled"] is False
    assert "basecalled_molecule_count" in report["disagreements"]


def test_a_lineage_without_a_report_says_so(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    with staged_lineage(
        plan, frozen, basecall, tmp_path / "rebasecall_outputs", accepted_plan_id=plan.plan_id
    ) as staged:
        staged.record_stage_generation("raw", "descendant-a")
        final_dir = staged.final_dir
    lineage = read_published_rebasecall_lineage(final_dir)

    with pytest.raises(RebasecallLineageError) as error:
        read_qc_transition(lineage)

    assert error.value.code == "transition_report_missing"


def test_deeper_targets_are_refused_rather_than_stopping_short(tmp_path, monkeypatch):
    from smftools.pipeline.rebasecall_run import run_lineage_raw_stage

    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    deep = replace(plan, request=replace(plan.request, downstream_target="hmm"))

    with pytest.raises(RebasecallLineageError) as error:
        run_lineage_raw_stage(
            deep,
            frozen,
            basecall,
            tmp_path / "rebasecall_outputs",
            accepted_plan_id=deep.plan_id,
            parent_config_path=tmp_path / "missing.csv",
        )

    assert error.value.code == "lineage_target_unsupported"
