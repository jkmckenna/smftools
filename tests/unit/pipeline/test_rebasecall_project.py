from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.pipeline.test_rebasecall_plan import _request

from smftools.pipeline.rebasecall_lineage import RebasecallLineageError
from smftools.pipeline.rebasecall_project import (
    format_project_rebasecall_plan,
    plan_project_rebasecall,
    run_project_rebasecall,
)
from smftools.project import registry as reg

pytestmark = pytest.mark.unit


@dataclass
class _FakeLineage:
    lineage_id: str
    basecall_id: str
    stage_generations: dict


@dataclass
class _FakeResult:
    lineage: _FakeLineage
    run_root: Path


class _FakeCatalog:
    """Stands in for ProjectCatalog: the fan-out only asks it for experiments."""

    def __init__(self, entries):
        self._entries = entries

    def experiments(self, **_kwargs):
        return list(self._entries)


def _write_config(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "experiment_config.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(
            [
                ("variable", "value", "help", "options", "type"),
                ("output_directory", str(run_root), "", "", "str"),
            ]
        )
    return path


def _project(tmp_path, experiment_ids=("exp-a", "exp-b"), *, with_config=True):
    project_dir = tmp_path / "project"
    reg.init_project(project_dir)
    entries = []
    for exp_id in experiment_ids:
        run_root = tmp_path / exp_id
        if with_config:
            _write_config(run_root)
        else:
            run_root.mkdir(parents=True, exist_ok=True)
        entries.append(
            {
                "id": exp_id,
                "experiment_uid": f"uid-{exp_id}",
                "path": str(run_root),
                "lineage": "original",
            }
        )
        # Register a minimal entry so lineage registration has something to
        # attach to; the fan-out reads paths from the catalog, not the registry.
        spine = run_root / "spine.h5ad"
        spine.touch()
        registry = reg.load_registry(project_dir)
        registry["experiments"][exp_id] = {
            "path": str(run_root),
            "name": exp_id,
            "experiment_uid": f"uid-{exp_id}",
            "status": "active",
            "spines": {"raw": str(spine)},
            "catalogs": {},
        }
        reg.save_registry(project_dir, registry)
    return project_dir, entries


def _planner(status="ready", blockers=(), model_name="chem_hac@v1.0.0"):
    def planner(cfg, _request):
        return SimpleNamespace(
            status=status,
            blockers=tuple(SimpleNamespace(code=code) for code in blockers),
            plan_id=f"plan-{Path(cfg.output_directory).name}",
            model=SimpleNamespace(
                to_dict=lambda: {"simplex_model": {"name": model_name}},
            ),
        )

    return planner


def test_planning_covers_every_selected_experiment_and_writes_nothing(tmp_path):
    project_dir, entries = _project(tmp_path)
    before = sorted(path.name for path in tmp_path.rglob("*"))

    plan = plan_project_rebasecall(
        project_dir,
        _request(),
        planner=_planner(),
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
    )

    assert [member.experiment_id for member in plan.members] == ["exp-a", "exp-b"]
    assert len(plan.ready) == 2
    assert plan.to_dict()["ready_count"] == 2
    assert sorted(path.name for path in tmp_path.rglob("*")) == before


def test_each_experiment_resolves_its_own_model(tmp_path):
    project_dir, entries = _project(tmp_path)
    models = {"exp-a": "chem_hac@v1.0.0", "exp-b": "chem_sup@v2.0.0"}

    def planner(cfg, _request):
        experiment = Path(cfg.output_directory).name
        return SimpleNamespace(
            status="ready",
            blockers=(),
            plan_id=f"plan-{experiment}",
            model=SimpleNamespace(to_dict=lambda: {"simplex_model": {"name": models[experiment]}}),
        )

    plan = plan_project_rebasecall(
        project_dir,
        _request(),
        planner=planner,
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
    )

    resolved = {
        member.experiment_id: member.model["simplex_model"]["name"] for member in plan.members
    }
    assert resolved == models


def test_one_blocked_experiment_does_not_block_the_others(tmp_path):
    project_dir, entries = _project(tmp_path)

    def planner(cfg, _request):
        experiment = Path(cfg.output_directory).name
        if experiment == "exp-a":
            raise RuntimeError("this experiment cannot be planned")
        return _planner()(cfg, _request)

    plan = plan_project_rebasecall(
        project_dir,
        _request(),
        planner=planner,
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
    )

    assert [member.experiment_id for member in plan.blocked] == ["exp-a"]
    assert [member.experiment_id for member in plan.ready] == ["exp-b"]
    assert "this experiment cannot be planned" in plan.blocked[0].blockers[0]


def test_an_experiment_without_a_config_is_reported_not_skipped(tmp_path):
    project_dir, entries = _project(tmp_path, with_config=False)

    plan = plan_project_rebasecall(
        project_dir,
        _request(),
        planner=_planner(),
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
    )

    assert {member.status for member in plan.members} == {"blocked"}
    assert plan.members[0].blockers == ("experiment_config_missing",)


def test_an_unknown_experiment_selection_is_refused(tmp_path):
    project_dir, entries = _project(tmp_path)

    with pytest.raises(RebasecallLineageError) as error:
        plan_project_rebasecall(
            project_dir,
            _request(),
            experiments=["exp-a", "exp-missing"],
            planner=_planner(),
            catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
        )

    assert error.value.code == "project_experiment_unknown"


def test_running_registers_each_lineage_without_making_it_active(tmp_path):
    project_dir, entries = _project(tmp_path)
    request = _request()

    def runner(*, member, request, root):
        run_root = member.run_root
        generation = f"gen-{member.experiment_id}"
        spine = run_root / "raw_outputs" / "generations" / generation / "spine.h5ad"
        spine.parent.mkdir(parents=True, exist_ok=True)
        spine.touch()
        return _FakeResult(
            lineage=_FakeLineage(
                lineage_id=f"lineage-{member.experiment_id}",
                basecall_id="c" * 64,
                stage_generations={"raw": generation},
            ),
            run_root=run_root,
        )

    result = run_project_rebasecall(
        project_dir,
        request,
        tmp_path / "fanout",
        accepted_request_id=request.request_id,
        planner=_planner(),
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
        lineage_runner=runner,
    )

    assert {member.status for member in result.members} == {"published"}
    resolved = {entry["id"]: entry for entry in reg.list_experiments(project_dir)}
    for exp_id in ("exp-a", "exp-b"):
        # Registered and queryable by name...
        assert f"lineage-{exp_id}" in resolved[exp_id]["available_lineages"]
        # ...but the project still answers with the original.
        assert resolved[exp_id]["lineage"] == "original"
    named = reg.list_experiments(project_dir, lineage={"exp-a": "lineage-exp-a"})
    assert {entry["lineage"] for entry in named} == {"lineage-exp-a", "original"}


def test_one_failed_experiment_does_not_abort_the_project(tmp_path):
    project_dir, entries = _project(tmp_path)
    request = _request()

    def runner(*, member, request, root):
        if member.experiment_id == "exp-a":
            raise RuntimeError("basecalling died")
        generation = f"gen-{member.experiment_id}"
        spine = member.run_root / "raw_outputs" / "generations" / generation / "spine.h5ad"
        spine.parent.mkdir(parents=True, exist_ok=True)
        spine.touch()
        return _FakeResult(
            lineage=_FakeLineage(
                lineage_id=f"lineage-{member.experiment_id}",
                basecall_id="c" * 64,
                stage_generations={"raw": generation},
            ),
            run_root=member.run_root,
        )

    result = run_project_rebasecall(
        project_dir,
        request,
        tmp_path / "fanout",
        accepted_request_id=request.request_id,
        planner=_planner(),
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
        lineage_runner=runner,
    )

    statuses = {member.experiment_id: member.status for member in result.members}
    assert statuses == {"exp-a": "failed", "exp-b": "published"}
    resolved = {entry["id"]: entry for entry in reg.list_experiments(project_dir)}
    assert resolved["exp-a"]["available_lineages"] == ["original"]
    assert "lineage-exp-b" in resolved["exp-b"]["available_lineages"]


def test_a_stale_accepted_request_never_runs(tmp_path):
    project_dir, entries = _project(tmp_path)

    with pytest.raises(RebasecallLineageError) as error:
        run_project_rebasecall(
            project_dir,
            _request(),
            tmp_path / "fanout",
            accepted_request_id="not-the-request",
            planner=_planner(),
            catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
            lineage_runner=lambda **_kwargs: pytest.fail("a stale request must not run"),
        )

    assert error.value.code == "accepted_request_mismatch"


def test_the_human_report_names_each_experiment_and_its_state(tmp_path):
    project_dir, entries = _project(tmp_path)

    plan = plan_project_rebasecall(
        project_dir,
        _request(),
        planner=_planner(),
        catalog_opener=lambda *_a, **_k: _FakeCatalog(entries),
    )
    report = format_project_rebasecall_plan(plan)

    assert "exp-a: ready" in report
    assert "chem_hac@v1.0.0" in report
    assert "until an explicit promotion" in report


def test_the_project_plan_command_reports_without_writing(tmp_path, monkeypatch):
    """The CLI the tutorial documents must exist and stay read-only."""
    import json

    from click.testing import CliRunner

    from smftools import cli_entry

    project_dir, entries = _project(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "cli-request",
                "source": {"raw_generation": "raw-a"},
                "selection": {"mode": "all-parent-molecules"},
                "basecall": {"model": "hac@latest"},
                "signal": {"materialize": False},
                "downstream": {"target": "preprocess"},
                "promotion": {"activate": False},
            }
        ),
        encoding="utf-8",
    )
    before = sorted(path.name for path in tmp_path.rglob("*"))
    # Capture the real planner before patching, or the stub would call itself.
    real_plan = plan_project_rebasecall

    def stub(*args, **kwargs):
        return real_plan(
            *args,
            **{
                **kwargs,
                "planner": _planner(),
                "catalog_opener": lambda *_a, **_k: _FakeCatalog(entries),
            },
        )

    monkeypatch.setattr("smftools.pipeline.rebasecall_project.plan_project_rebasecall", stub)

    result = CliRunner().invoke(
        cli_entry.cli,
        ["project", "rebasecall", "plan", str(project_dir), str(request_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    # CliRunner folds stderr into output, and config loading warns here.
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["ready_count"] == 2
    assert {member["experiment_id"] for member in payload["members"]} == {"exp-a", "exp-b"}
    assert sorted(path.name for path in tmp_path.rglob("*")) == before
