from __future__ import annotations

import csv
from pathlib import Path

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


def _lineage_case(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    basecall = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())
    return plan, frozen, basecall


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


def test_the_raw_stage_runs_inside_the_lineage_and_is_recorded(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    root = tmp_path / "rebasecall_outputs"
    seen: list[dict[str, object]] = []

    def runner(config_path, *, lineage_provenance):
        seen.append(
            {
                "config": _config_values(Path(config_path)),
                "provenance": dict(lineage_provenance),
            }
        )
        return (
            None,
            tmp_path / "raw_outputs" / "generations" / "descendant-a" / "spine.h5ad",
            None,
        )

    result = run_lineage_raw_stage(
        plan,
        frozen,
        basecall,
        root,
        accepted_plan_id=plan.plan_id,
        parent_config_path=parent,
        raw_stage_runner=runner,
    )

    assert result.raw_generation_id == "descendant-a"
    assert result.lineage.stage_generations == {"raw": "descendant-a"}
    assert len(seen) == 1
    assert seen[0]["config"]["input_data_path"] == str(basecall.directory / BASECALL_BAM_FILENAME)
    # Per D2 the descendant derives its kind from the basecall it was built from.
    assert seen[0]["provenance"]["generation_kind"] == basecall.generation_kind
    assert seen[0]["provenance"]["basecall_id"] == basecall.basecall_id
    assert seen[0]["provenance"]["lineage_id"] == result.lineage.lineage_id
    assert (result.lineage.directory / DESCENDANT_CONFIG_FILENAME).is_file()
    assert result.descendant_config_path.is_file()


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
        raw_stage_runner=lambda config_path, *, lineage_provenance: (
            None,
            tmp_path / "raw_outputs" / "generations" / "descendant-a" / "spine.h5ad",
            None,
        ),
    )

    payload = result.to_dict()

    assert payload["lineage_id"] == result.lineage.lineage_id
    assert payload["basecall_id"] == basecall.basecall_id
    assert payload["stage_generations"] == {"raw": "descendant-a"}
    assert payload["raw_generation_id"] == "descendant-a"
    assert Path(payload["run_root"]) == Path(plan.run_root)


def test_a_killed_raw_stage_publishes_no_lineage(tmp_path, monkeypatch):
    plan, frozen, basecall = _lineage_case(tmp_path, monkeypatch)
    parent = _write_parent_config(tmp_path)
    root = tmp_path / "rebasecall_outputs"

    def killed(config_path, *, lineage_provenance):
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
            raw_stage_runner=lambda config_path, *, lineage_provenance: None,
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
        raw_stage_runner=lambda config_path, *, lineage_provenance: (
            None,
            tmp_path / "raw_outputs" / "generations" / "descendant-a" / "spine.h5ad",
            None,
        ),
    )

    relocated = tmp_path / "moved_outputs"
    root.rename(relocated)
    moved = list_published_rebasecall_lineages(relocated)

    # Lineage identity is path-neutral, so moving the container preserves it.
    assert [item.lineage_id for item in moved] == [result.lineage.lineage_id]
    assert moved[0].stage_generations == {"raw": "descendant-a"}
