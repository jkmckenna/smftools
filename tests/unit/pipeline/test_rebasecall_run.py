from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.pipeline.test_rebasecall_basecall import _case, _execute, _FakeDorado

from smftools.pipeline.rebasecall_basecall import BASECALL_BAM_FILENAME
from smftools.pipeline.rebasecall_lineage import (
    LINEAGE_STAGING_SUBDIR,
    LINEAGES_SUBDIR,
    RebasecallLineageError,
    list_published_rebasecall_lineages,
)
from smftools.pipeline.rebasecall_run import (
    DESCENDANT_CONFIG_FILENAME,
    derive_descendant_config,
    run_lineage_raw_stage,
)

pytestmark = pytest.mark.unit

_PARENT_CONFIG_ROWS = (
    ("variable", "value", "help", "options", "type"),
    ("smf_modality", "direct", "Modality of SMF.", "", "str"),
    ("input_data_path", "/parent/reads.pod5", "Input path", "", "str"),
    ("alignment_mode", "existing", "Alignment mode", "", "str"),
    ("fasta", "/refs/genome.fa", "Reference", "", "str"),
    ("output_directory", "/runs/experiment-a", "Output root", "", "str"),
    ("hmm_n_states", "3", "States", "", "int"),
)


def _write_parent_config(tmp_path: Path) -> Path:
    path = tmp_path / "experiment_config.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(_PARENT_CONFIG_ROWS)
    return path


def _config_values(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row]
    return {row[0]: row[1] for row in rows[1:]}


def _selected_reads(frozen):
    return sorted(pd.read_parquet(frozen.rows_path)["pod5_read_id"].astype(str))


def _lineage_case(tmp_path, monkeypatch, *, mode="all-parent-molecules", target="preprocess"):
    _, _, plan, frozen = _case(tmp_path, monkeypatch, mode=mode)
    # The shared fixture requests "full"; lineage execution currently supports
    # raw and preprocess, and refuses deeper targets rather than stopping short.
    plan = replace(plan, request=replace(plan.request, downstream_target=target))
    basecall = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())
    return plan, frozen, basecall


def _spine(tmp_path, stage, generation_id):
    return tmp_path / stage / "generations" / generation_id / "spine.h5ad"


def _stage_runner(tmp_path, stage, generation_id, *, record=None, read_ids=()):
    """A stage double that publishes the artifacts the transition report reads."""

    def runner(config_path, **kwargs):
        if record is not None:
            # Read the config here: it lives in the lineage staging tree, which
            # is gone by the time the caller inspects the result.
            record.append({"config": _config_values(Path(config_path)), **kwargs})
        spine = _spine(tmp_path, stage, generation_id)
        spine.parent.mkdir(parents=True, exist_ok=True)
        rows = list(read_ids)
        if stage == "raw_outputs":
            pd.DataFrame(
                {"read_id": rows, "molecule_uid": [f"m-{value}" for value in rows]}
            ).to_parquet(spine.parent / "obs.parquet", index=False)
        else:
            pd.DataFrame({"read_id": rows, "passes_qc": [True] * len(rows)}).to_parquet(
                spine.parent / "stage_obs.parquet", index=False
            )
        return (spine, None)

    return runner


def test_descendant_config_reads_the_new_calls_and_inherits_everything_else(tmp_path, monkeypatch):
    _, _, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)

    derived = derive_descendant_config(parent, basecall, tmp_path / "derived.csv")

    values = _config_values(derived)
    assert values["input_data_path"] == str(basecall.directory / BASECALL_BAM_FILENAME)
    assert values["alignment_mode"] == "align"
    # The descendant publishes beside the parent, so the run root is inherited.
    assert values["output_directory"] == "/runs/experiment-a"
    assert values["smf_modality"] == "direct"
    assert values["fasta"] == "/refs/genome.fa"
    assert values["hmm_n_states"] == "3"


def test_a_basecall_without_calls_cannot_derive_a_config(tmp_path, monkeypatch):
    _, _, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    (basecall.directory / BASECALL_BAM_FILENAME).unlink()

    with pytest.raises(RebasecallLineageError) as error:
        derive_descendant_config(parent, basecall, tmp_path / "derived.csv")

    assert error.value.code == "lineage_basecall_missing"


def test_the_stages_run_inside_the_lineage_and_are_recorded(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    root = tmp_path / "rebasecall_outputs"
    raw_seen: list[dict[str, object]] = []
    preprocess_seen: list[dict[str, object]] = []

    result = run_lineage_raw_stage(
        plan,
        frozen,
        basecall,
        root,
        accepted_plan_id=plan.plan_id,
        parent_config_path=parent,
        raw_stage_runner=_stage_runner(
            tmp_path,
            "raw_outputs",
            "descendant-a",
            record=raw_seen,
            read_ids=_selected_reads(frozen),
        ),
        preprocess_stage_runner=_stage_runner(
            tmp_path,
            "preprocess_adata_outputs",
            "descendant-p",
            record=preprocess_seen,
            read_ids=_selected_reads(frozen),
        ),
    )

    assert result.raw_generation_id == "descendant-a"
    assert result.lineage.stage_generations == {
        "raw": "descendant-a",
        "preprocess": "descendant-p",
    }
    assert len(raw_seen) == 1 and len(preprocess_seen) == 1
    assert raw_seen[0]["config"]["input_data_path"] == str(
        basecall.directory / BASECALL_BAM_FILENAME
    )
    provenance = dict(raw_seen[0]["lineage_provenance"])
    # Per D2 the descendant derives its kind from the basecall it was built from.
    assert provenance["generation_kind"] == basecall.generation_kind
    assert provenance["basecall_id"] == basecall.basecall_id
    assert provenance["lineage_id"] == result.lineage.lineage_id
    # Preprocess must read the descendant raw generation, not whatever the
    # parent currently selects.
    assert preprocess_seen[0]["lineage_generations"] == {"raw": "descendant-a"}
    assert (result.lineage.directory / DESCENDANT_CONFIG_FILENAME).is_file()
    assert result.descendant_config_path.is_file()


def test_a_raw_only_target_stops_after_raw(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch, target="raw")
    parent = _write_parent_config(tmp_path)
    preprocess_calls: list[dict[str, object]] = []

    result = run_lineage_raw_stage(
        plan,
        frozen,
        basecall,
        tmp_path / "rebasecall_outputs",
        accepted_plan_id=plan.plan_id,
        parent_config_path=parent,
        raw_stage_runner=_stage_runner(
            tmp_path, "raw_outputs", "descendant-a", read_ids=_selected_reads(frozen)
        ),
        preprocess_stage_runner=_stage_runner(
            tmp_path,
            "preprocess_adata_outputs",
            "descendant-p",
            record=preprocess_calls,
        ),
    )

    assert result.lineage.stage_generations == {"raw": "descendant-a"}
    assert preprocess_calls == []


def test_the_result_payload_follows_the_workflow_contract(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)

    result = run_lineage_raw_stage(
        plan,
        frozen,
        basecall,
        tmp_path / "rebasecall_outputs",
        accepted_plan_id=plan.plan_id,
        parent_config_path=parent,
        raw_stage_runner=_stage_runner(
            tmp_path, "raw_outputs", "descendant-a", read_ids=_selected_reads(frozen)
        ),
        preprocess_stage_runner=_stage_runner(
            tmp_path, "preprocess_adata_outputs", "descendant-p", read_ids=_selected_reads(frozen)
        ),
    )

    payload = result.to_dict()

    assert payload["lineage_id"] == result.lineage.lineage_id
    assert payload["basecall_id"] == basecall.basecall_id
    assert payload["stage_generations"] == {
        "raw": "descendant-a",
        "preprocess": "descendant-p",
    }
    assert payload["raw_generation_id"] == "descendant-a"
    assert Path(payload["run_root"]) == Path(plan.run_root)


def test_a_killed_raw_stage_publishes_no_lineage(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    root = tmp_path / "rebasecall_outputs"

    def killed(config_path, **_kwargs):
        raise RuntimeError("raw stage was killed")

    with pytest.raises(RuntimeError, match="raw stage was killed"):
        run_lineage_raw_stage(
            plan,
            frozen,
            basecall,
            root,
            accepted_plan_id=plan.plan_id,
            parent_config_path=parent,
            raw_stage_runner=killed,
        )

    assert list_published_rebasecall_lineages(root) == ()
    assert not any((root / LINEAGE_STAGING_SUBDIR).iterdir())
    assert not (root / LINEAGES_SUBDIR).exists() or not any((root / LINEAGES_SUBDIR).iterdir())


def test_a_raw_stage_that_reports_no_generation_is_an_error(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    root = tmp_path / "rebasecall_outputs"

    with pytest.raises(RebasecallLineageError) as error:
        run_lineage_raw_stage(
            plan,
            frozen,
            basecall,
            root,
            accepted_plan_id=plan.plan_id,
            parent_config_path=parent,
            raw_stage_runner=lambda config_path, **_kwargs: None,
        )

    assert error.value.code == "lineage_raw_stage_unrecognized"
    assert list_published_rebasecall_lineages(root) == ()


def test_a_relocated_lineage_still_validates(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    root = tmp_path / "rebasecall_outputs"
    result = run_lineage_raw_stage(
        plan,
        frozen,
        basecall,
        root,
        accepted_plan_id=plan.plan_id,
        parent_config_path=parent,
        raw_stage_runner=_stage_runner(
            tmp_path, "raw_outputs", "descendant-a", read_ids=_selected_reads(frozen)
        ),
        preprocess_stage_runner=_stage_runner(
            tmp_path, "preprocess_adata_outputs", "descendant-p", read_ids=_selected_reads(frozen)
        ),
    )

    relocated = tmp_path / "moved_outputs"
    root.rename(relocated)
    moved = list_published_rebasecall_lineages(relocated)

    # Lineage identity is path-neutral, so moving the container preserves it.
    assert [item.lineage_id for item in moved] == [result.lineage.lineage_id]
    assert moved[0].stage_generations == {"raw": "descendant-a", "preprocess": "descendant-p"}


def test_the_lineage_publishes_a_reconcilable_transition_report(tmp_path, monkeypatch):
    """The exit gate, end to end: run the lineage, then reconcile what it wrote."""
    from smftools.pipeline.rebasecall_transition import (
        read_qc_transition,
        reconcile_qc_transition,
    )

    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    selected = _selected_reads(frozen)

    result = run_lineage_raw_stage(
        plan,
        frozen,
        basecall,
        tmp_path / "rebasecall_outputs",
        accepted_plan_id=plan.plan_id,
        parent_config_path=parent,
        raw_stage_runner=_stage_runner(tmp_path, "raw_outputs", "descendant-a", read_ids=selected),
        preprocess_stage_runner=_stage_runner(
            tmp_path, "preprocess_adata_outputs", "descendant-p", read_ids=selected
        ),
    )

    frame, summary = read_qc_transition(result.lineage)
    report = reconcile_qc_transition(frame, summary)

    assert report["reconciled"] is True
    assert len(frame) == len(selected)
    assert summary["selected_molecule_count"] == len(selected)
    assert result.to_dict()["qc_transition"]["passes_qc_count"] == len(selected)
