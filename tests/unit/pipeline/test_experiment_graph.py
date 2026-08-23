from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import helpers
from smftools.informatics.experiment_manifest import (
    StageLifecycle,
    read_experiment_manifest,
    record_stage_state,
)
from smftools.pipeline import experiment_graph
from smftools.pipeline.semantic_graph import PlanState

pytestmark = pytest.mark.unit

_STAGE_SCHEMAS = {
    "raw": 3,
    "preprocess": 2,
    "spatial": 3,
    "hmm": 2,
    "latent": 2,
}


def _cfg(tmp_path, *, run_latent: bool = True, **overrides):
    values = {
        "output_directory": tmp_path,
        "experiment_name": "experiment",
        "smf_modality": "direct",
        "full_run_latent": run_latent,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _paths(tmp_path):
    return SimpleNamespace(
        raw=tmp_path / "legacy-raw.h5ad.gz",
        pp=tmp_path / "legacy-preprocess.h5ad.gz",
        pp_dedup=tmp_path / "legacy-preprocess.h5ad.gz",
        spatial=tmp_path / "legacy-spatial.h5ad.gz",
        hmm=tmp_path / "legacy-hmm.h5ad.gz",
        latent=tmp_path / "legacy-latent.h5ad.gz",
        variant=tmp_path / "legacy-variant.h5ad.gz",
        chimeric=tmp_path / "legacy-chimeric.h5ad.gz",
        spine=tmp_path / "load_adata_outputs" / "spine.h5ad",
        raw_spine=tmp_path / "raw_outputs" / "spine.h5ad",
        preprocess_spine=tmp_path / "preprocess_adata_outputs" / "spine.h5ad",
        spatial_spine=tmp_path / "spatial_adata_outputs" / "spine.h5ad",
        hmm_spine=tmp_path / "hmm_adata_outputs" / "spine.h5ad",
        latent_spine=tmp_path / "latent_adata_outputs" / "spine.h5ad",
    )


def _record_stage(cfg, stage: str, *, config_value: str | None = None):
    record_stage_state(
        cfg.output_directory,
        stage,
        "complete",
        config_hash=config_value or helpers.stage_config_hash(cfg, stage),
        input_artifact_ids=[],
        schema_versions={
            stage: _STAGE_SCHEMAS[stage],
        },
    )


def _raw_identity_cfg(tmp_path, *, manifest=False):
    source = tmp_path / "reads.fastq"
    source.write_bytes(b"reads-v1")
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    values = {
        "smf_modality": "conversion",
        "fasta": fasta,
        "input_files": [source],
        "input_manifest_path": None,
        "alignment_mode": "align",
        "fastq_barcode_map": None,
        "fastq_auto_pairing": True,
        "conversion_types": [],
        "strands": [],
    }
    if manifest:
        manifest_path = tmp_path / "inputs.csv"
        manifest_path.write_text("path\nreads.fastq\n", encoding="utf-8")
        values.update(input_manifest_path=manifest_path, input_files=[source])
    return _cfg(tmp_path, **values), source, fasta


def _record_raw_identity(cfg):
    record_stage_state(
        cfg.output_directory,
        "raw",
        "complete",
        config_hash=helpers.stage_config_hash(cfg, "raw"),
        input_artifact_ids=helpers.raw_input_artifact_ids(cfg),
        schema_versions={"raw": 3},
    )


def _states(plan):
    return {decision.analysis_id: decision.state for decision in plan.decisions}


def _trust_current_preprocess(monkeypatch, tmp_path):
    from smftools.preprocessing import preprocess_generation

    monkeypatch.setattr(
        preprocess_generation,
        "resolve_current_preprocess_generation",
        lambda _output: (tmp_path / "generation", {"generation_id": None}),
    )


def test_experiment_graph_is_linear_and_variant_aliases_preprocess(tmp_path):
    specs = {spec.analysis_id: spec for spec in experiment_graph.experiment_node_specs()}

    assert set(experiment_graph.EXPERIMENT_NODE_IDS.values()).issubset(specs)
    assert set(experiment_graph.LEGACY_EXPERIMENT_NODE_IDS.values()).issubset(specs)
    assert specs[experiment_graph.EXPERIMENT_NODE_IDS["raw"]].dependencies == ()
    assert specs[experiment_graph.EXPERIMENT_NODE_IDS["latent"]].dependencies == (
        experiment_graph.EXPERIMENT_NODE_IDS["hmm"],
    )
    assert "variant" not in experiment_graph.LEGACY_EXPERIMENT_NODE_IDS
    assert (
        experiment_graph.resolve_experiment_target(_cfg(tmp_path), "variant")
        == (experiment_graph.EXPERIMENT_NODE_IDS["preprocess"])
    )


def test_full_target_resolves_latent_by_default_and_hmm_when_disabled(tmp_path):
    paths = _paths(tmp_path)
    latent_cfg = _cfg(tmp_path)
    hmm_cfg = _cfg(tmp_path, run_latent=False)

    latent_plan = experiment_graph.build_experiment_plan(
        latent_cfg,
        "full",
        paths=paths,
    )
    hmm_plan = experiment_graph.build_experiment_plan(
        hmm_cfg,
        "full",
        paths=paths,
    )

    assert latent_plan.requested_target == experiment_graph.EXPERIMENT_NODE_IDS["latent"]
    assert latent_plan.topological_order == tuple(
        experiment_graph.EXPERIMENT_NODE_IDS[stage] for stage in experiment_graph.EXPERIMENT_STAGES
    )
    assert hmm_plan.requested_target == experiment_graph.EXPERIMENT_NODE_IDS["hmm"]
    assert experiment_graph.EXPERIMENT_NODE_IDS["latent"] not in hmm_plan.topological_order


def test_partial_plan_reuses_compatible_dependencies_and_stops_at_target(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    _record_stage(cfg, "raw")
    _record_stage(cfg, "preprocess")
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)
    _trust_current_preprocess(monkeypatch, tmp_path)

    plan = experiment_graph.build_experiment_plan(cfg, "hmm", paths=paths)

    states = _states(plan)
    assert plan.topological_order == tuple(
        experiment_graph.EXPERIMENT_NODE_IDS[stage]
        for stage in ("raw", "preprocess", "spatial", "hmm")
    )
    assert states[experiment_graph.EXPERIMENT_NODE_IDS["raw"]] is PlanState.COMPATIBLE
    assert states[experiment_graph.EXPERIMENT_NODE_IDS["preprocess"]] is PlanState.COMPATIBLE
    assert states[experiment_graph.EXPERIMENT_NODE_IDS["spatial"]] is PlanState.MISSING
    assert states[experiment_graph.EXPERIMENT_NODE_IDS["hmm"]] is PlanState.MISSING


def test_legacy_preprocess_completion_without_current_generation_is_invalid(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    _record_stage(cfg, "raw")
    _record_stage(cfg, "preprocess")
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    plan = experiment_graph.build_experiment_plan(cfg, "preprocess", paths=paths)

    assert plan.decisions[-1].state is PlanState.INVALID_ARTIFACT
    assert plan.decisions[-1].reason_code == "stage_artifact_validation_failed"


def test_stale_config_and_invalid_artifacts_have_explicit_plan_reasons(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    _record_stage(cfg, "raw", config_value="old-config")
    _record_stage(cfg, "preprocess")

    def validate(_root, stage, **_kwargs):
        return stage != "preprocess"

    monkeypatch.setattr(experiment_graph, "stage_is_complete", validate)
    plan = experiment_graph.build_experiment_plan(cfg, "spatial", paths=paths)
    decisions = {decision.analysis_id: decision for decision in plan.decisions}

    raw = decisions[experiment_graph.EXPERIMENT_NODE_IDS["raw"]]
    preprocess = decisions[experiment_graph.EXPERIMENT_NODE_IDS["preprocess"]]
    spatial = decisions[experiment_graph.EXPERIMENT_NODE_IDS["spatial"]]
    assert raw.state is PlanState.STALE_CONFIG
    assert raw.reason_code == "semantic_config_changed"
    assert preprocess.state is PlanState.INVALID_ARTIFACT
    assert preprocess.reason_code == "stage_artifact_validation_failed"
    assert spatial.state is PlanState.MISSING


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    [
        ("semantic_algorithm_version", "0", "algorithm_version_changed"),
        ("semantic_output_schema_version", 1, "output_schema_version_changed"),
    ],
)
def test_stored_semantic_versions_drive_stage_compatibility(
    tmp_path,
    monkeypatch,
    field,
    value,
    reason_code,
):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    record_stage_state(
        cfg.output_directory,
        "raw",
        "complete",
        config_hash=helpers.stage_config_hash(cfg, "raw"),
        input_artifact_ids=[],
        schema_versions={"raw": 3},
        **{field: value},
    )
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    plan = experiment_graph.build_experiment_plan(cfg, "raw", paths=paths)

    assert plan.decisions[0].state is PlanState.STALE_ALGORITHM
    assert plan.decisions[0].reason_code == reason_code


def test_changed_source_artifact_is_reported_as_stale_input(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    paths.raw_spine.parent.mkdir(parents=True)
    paths.raw_spine.write_bytes(b"raw-source-v2")
    _record_stage(cfg, "raw")
    record_stage_state(
        cfg.output_directory,
        "preprocess",
        "complete",
        config_hash=helpers.stage_config_hash(cfg, "preprocess"),
        input_artifact_ids=["old-raw-source"],
        schema_versions={"preprocess": 2},
    )
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    plan = experiment_graph.build_experiment_plan(cfg, "preprocess", paths=paths)
    decision = plan.decisions[-1]

    assert decision.state is PlanState.STALE_INPUT
    assert decision.reason_code == "input_artifacts_changed"


def test_raw_planning_uses_manifest_sources_and_reference_identity(tmp_path, monkeypatch):
    cfg, source, fasta = _raw_identity_cfg(tmp_path)
    paths = _paths(tmp_path)
    _record_raw_identity(cfg)
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    compatible = experiment_graph.build_experiment_plan(cfg, "raw", paths=paths)
    assert compatible.decisions[0].state is PlanState.COMPATIBLE
    expected = experiment_graph._expected_stage_inputs(
        cfg,
        paths,
        "raw",
        scope_identity=f"experiment_name:{cfg.experiment_name}",
    )
    assert len(expected.input_artifacts) == 3
    assert expected.input_artifacts[0].artifact_id == "input-manifest"
    assert expected.input_artifacts[1].artifact_id.startswith("source:")
    assert expected.input_artifacts[2].artifact_id == "alignment-reference-bundle"

    source.write_bytes(b"reads-v2")
    changed_source = experiment_graph.build_experiment_plan(cfg, "raw", paths=paths)
    assert changed_source.decisions[0].state is PlanState.STALE_INPUT

    source.write_bytes(b"reads-v1")
    fasta.write_text(">ref\nTGCA\n", encoding="utf-8")
    changed_reference = experiment_graph.build_experiment_plan(cfg, "raw", paths=paths)
    assert changed_reference.decisions[0].state is PlanState.STALE_INPUT


def test_raw_manifest_row_reorder_is_compatible_but_membership_change_is_stale(
    tmp_path, monkeypatch
):
    cfg, first, _fasta = _raw_identity_cfg(tmp_path, manifest=True)
    second = tmp_path / "other.fastq"
    second.write_bytes(b"other")
    cfg.input_manifest_path.write_text("path\nreads.fastq\nother.fastq\n", encoding="utf-8")
    cfg.input_files = [first, second]
    _record_raw_identity(cfg)
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    cfg.input_manifest_path.write_text("path\nother.fastq\nreads.fastq\n", encoding="utf-8")
    reordered = experiment_graph.build_experiment_plan(cfg, "raw", paths=_paths(tmp_path))
    assert reordered.decisions[0].state is PlanState.COMPATIBLE

    cfg.input_manifest_path.write_text("path\nreads.fastq\n", encoding="utf-8")
    cfg.input_files = [first]
    removed = experiment_graph.build_experiment_plan(cfg, "raw", paths=_paths(tmp_path))
    assert removed.decisions[0].state is PlanState.STALE_INPUT


def test_added_raw_source_invalidates_raw_channel_and_dependent_preprocess(tmp_path, monkeypatch):
    cfg, first, _fasta = _raw_identity_cfg(tmp_path, manifest=True)
    _record_raw_identity(cfg)
    _record_stage(cfg, "preprocess")
    second = tmp_path / "added.fastq"
    second.write_bytes(b"added")
    cfg.input_manifest_path.write_text("path\nreads.fastq\nadded.fastq\n", encoding="utf-8")
    cfg.input_files = [first, second]
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)
    _trust_current_preprocess(monkeypatch, tmp_path)

    plan = experiment_graph.build_experiment_plan(cfg, "preprocess", paths=_paths(tmp_path))
    states = _states(plan)

    assert states[experiment_graph.EXPERIMENT_NODE_IDS["raw"]] is PlanState.STALE_INPUT
    assert (
        states[experiment_graph.EXPERIMENT_NODE_IDS["preprocess"]] is PlanState.DEPENDENT_RECOMPUTE
    )


def test_force_flag_recomputes_target_and_dependents(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, force_redo_load_adata=True)
    paths = _paths(tmp_path)
    for stage in ("raw", "preprocess", "spatial", "hmm"):
        _record_stage(cfg, stage)
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    plan = experiment_graph.build_experiment_plan(cfg, "hmm", paths=paths)
    states = _states(plan)

    assert states[experiment_graph.EXPERIMENT_NODE_IDS["raw"]] is PlanState.MISSING
    assert (
        states[experiment_graph.EXPERIMENT_NODE_IDS["preprocess"]] is PlanState.DEPENDENT_RECOMPUTE
    )
    assert states[experiment_graph.EXPERIMENT_NODE_IDS["spatial"]] is PlanState.DEPENDENT_RECOMPUTE
    assert states[experiment_graph.EXPERIMENT_NODE_IDS["hmm"]] is PlanState.DEPENDENT_RECOMPUTE


def test_execution_invokes_only_missing_or_incompatible_stage_wrappers(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    _record_stage(cfg, "raw")
    _record_stage(cfg, "preprocess")
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)
    _trust_current_preprocess(monkeypatch, tmp_path)
    calls = []
    runners = {
        stage: lambda _config_path, stage=stage: calls.append(stage) or f"{stage}-result"
        for stage in experiment_graph.EXPERIMENT_STAGES
    }

    result = experiment_graph.execute_experiment_target(
        "experiment.csv",
        "hmm",
        cfg=cfg,
        paths=paths,
        stage_runners=runners,
    )

    assert calls == ["spatial", "hmm"]
    assert result.final_result == "hmm-result"
    assert [stage for stage, _value in result.stage_results] == [
        "raw",
        "preprocess",
        "spatial",
        "hmm",
    ]


def test_planning_is_read_only_and_json_is_deterministic(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    _record_stage(cfg, "raw")
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    first = experiment_graph.build_experiment_plan(cfg, "latent", paths=paths)
    second = experiment_graph.build_experiment_plan(cfg, "latent", paths=paths)
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert first.to_json() == second.to_json()
    assert before == after


def test_plan_cli_supports_human_and_machine_readable_output(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path / "run")
    config_path = tmp_path / "experiment.csv"
    config_path.touch()
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)

    human = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "plan", str(config_path), "--target", "hmm"],
    )
    machine = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "plan", str(config_path), "--target", "hmm", "--json"],
    )
    impact_human = CliRunner().invoke(
        cli_entry.cli,
        [
            "experiment",
            "plan",
            str(config_path),
            "--target",
            "hmm",
            "--upgrade-impact",
        ],
    )
    impact_machine = CliRunner().invoke(
        cli_entry.cli,
        [
            "experiment",
            "plan",
            str(config_path),
            "--target",
            "hmm",
            "--upgrade-impact",
            "--json",
        ],
    )

    assert human.exit_code == 0, human.output
    assert "Experiment target: experiment.hmm.complete" in human.output
    assert "missing" in human.output
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    assert payload["requested_target"] == experiment_graph.EXPERIMENT_NODE_IDS["hmm"]
    assert payload["topological_order"][-1] == experiment_graph.EXPERIMENT_NODE_IDS["hmm"]
    assert impact_human.exit_code == 0, impact_human.output
    assert "Estimated recompute cost: unknown" in impact_human.output
    assert "Plan states:" in impact_human.output
    assert impact_machine.exit_code == 0, impact_machine.output
    impact_payload = json.loads(impact_machine.output)
    assert impact_payload["schema_version"] == 1
    assert impact_payload["scope"] == "experiment"
    assert impact_payload["requested_target"] == experiment_graph.EXPERIMENT_NODE_IDS["hmm"]


def test_experiment_upgrade_impact_uses_prior_stage_elapsed_time(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    record_stage_state(
        cfg.output_directory,
        "raw",
        "complete",
        config_hash=helpers.stage_config_hash(cfg, "raw"),
        input_artifact_ids=[],
        schema_versions={"raw": 3},
        semantic_algorithm_version="0",
        timings={"elapsed_seconds": 12.5},
    )
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)
    before = (tmp_path / "experiment_manifest.json").read_bytes()

    report = experiment_graph.build_experiment_upgrade_impact(cfg, "raw", paths=paths)

    assert report.plan.decisions[0].state is PlanState.STALE_ALGORITHM
    assert report.to_dict()["recompute_cost"] == {
        "basis": "historical_elapsed_seconds",
        "estimated_seconds": 12.5,
        "complete": True,
        "recompute_node_count": 1,
        "known_node_count": 1,
        "known_nodes": [experiment_graph.EXPERIMENT_NODE_IDS["raw"]],
        "unknown_nodes": [],
    }
    assert (tmp_path / "experiment_manifest.json").read_bytes() == before


@pytest.mark.parametrize("target", ["raw", "preprocess", "spatial", "hmm", "latent"])
def test_direct_stage_cli_commands_submit_semantic_target_requests(
    tmp_path,
    monkeypatch,
    target,
):
    from smftools.cli import recipes

    config_path = tmp_path / "experiment.csv"
    config_path.touch()
    requests = []
    monkeypatch.setattr(
        recipes,
        "run_experiment_target",
        lambda path, requested: requests.append((path, requested)),
    )

    result = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", target, str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert requests == [(str(config_path), target)]


def test_stage_result_metadata_is_deterministic_and_versioned():
    kwargs = {
        "stage_config_hash": "config-1",
        "input_artifact_ids": ["raw:one"],
        "artifacts": {"spine": {"path": "raw_outputs/spine.h5ad", "size_bytes": 10}},
        "schema_versions": {"raw": 3},
    }

    first = experiment_graph.experiment_stage_result_metadata("raw", **kwargs)
    second = experiment_graph.experiment_stage_result_metadata("raw", **kwargs)

    assert first == second
    assert first["semantic_analysis_id"] == experiment_graph.EXPERIMENT_NODE_IDS["raw"]
    assert first["semantic_output_schema_version"] == 3
    assert first["semantic_result_id"].startswith("raw:")
    assert experiment_graph.experiment_stage_result_metadata("full", **kwargs) == {}


def test_stage_publication_records_semantic_result_identity(tmp_path):
    artifact = tmp_path / "raw_outputs" / "summary.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")

    with StageLifecycle(tmp_path, "raw", config_hash="config-1") as lifecycle:
        helpers.publish_stage_outputs(
            lifecycle,
            {"summary": artifact},
            required=("summary",),
            task_catalog_key=None,
            checksum_keys=("summary",),
            schema_versions={"raw": 3},
        )

    entry = read_experiment_manifest(tmp_path)["stages"]["raw"]
    assert entry["semantic_analysis_id"] == experiment_graph.EXPERIMENT_NODE_IDS["raw"]
    assert entry["semantic_algorithm_version"] == experiment_graph._STAGE_ALGORITHM_VERSIONS["raw"]
    assert entry["semantic_output_schema_version"] == 3
    assert entry["semantic_result_id"].startswith("raw:")
    assert len(entry["semantic_channel_fingerprint"]) == 64


def test_batch_surfaces_target_planning_failures(tmp_path, monkeypatch):
    from smftools.cli import recipes

    config = tmp_path / "experiment.csv"
    config.touch()
    config_table = tmp_path / "configs.txt"
    config_table.write_text(f"{config}\n", encoding="utf-8")

    def fail_plan(_config_path, _target):
        raise ValueError("simulated target planning failure")

    monkeypatch.setattr(recipes, "run_experiment_target", fail_plan)
    result = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "batch", "hmm", str(config_table)],
    )

    assert result.exit_code != 0
    summary = json.loads((tmp_path / "configs.hmm.batch-summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["results"][0]["exception"] == {
        "type": "ValueError",
        "message": "simulated target planning failure",
    }


def test_every_stage_declares_an_explicit_algorithm_version():
    """Each stage owns a version that reaches its spec.

    The dict was once ``{stage: "1" for stage in EXPERIMENT_STAGES}``, which made
    ``algorithm_version`` inert: no behavioural fix could ever mark a stored
    generation stale.
    """
    versions = experiment_graph._STAGE_ALGORITHM_VERSIONS
    assert set(versions) == set(experiment_graph.EXPERIMENT_STAGES)
    assert all(isinstance(value, str) and value for value in versions.values())

    specs = {spec.analysis_id: spec for spec in experiment_graph.experiment_node_specs()}
    for stage in experiment_graph.EXPERIMENT_STAGES:
        spec = specs[experiment_graph.EXPERIMENT_NODE_IDS[stage]]
        assert spec.algorithm_version == versions[stage]


def test_generation_from_before_the_raw_algorithm_bump_is_not_compatible(tmp_path, monkeypatch):
    """A pre-F31 raw generation must not be served as compatible.

    Raw generations written before the demux-status fix were recorded at
    algorithm version "1" and lack the demux obs columns, yet the planner
    reported them ``compatible`` because the version never moved.
    """
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    record_stage_state(
        cfg.output_directory,
        "raw",
        "complete",
        config_hash=helpers.stage_config_hash(cfg, "raw"),
        input_artifact_ids=[],
        schema_versions={"raw": 3},
        semantic_algorithm_version="1",
    )
    monkeypatch.setattr(experiment_graph, "stage_is_complete", lambda *_args, **_kwargs: True)

    plan = experiment_graph.build_experiment_plan(cfg, "raw", paths=paths)

    assert plan.decisions[0].state is PlanState.STALE_ALGORITHM
    assert plan.decisions[0].reason_code == "algorithm_version_changed"
