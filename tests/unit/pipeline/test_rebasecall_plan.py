from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.informatics.dorado_model import (
    DoradoBasecallResolution,
    DoradoModelArtifact,
    DoradoModelError,
    DoradoRunCondition,
)
from smftools.informatics.input_manifest import InputManifestRow, ResolvedInputManifest
from smftools.informatics.pod5_identity import Pod5DatasetIndex
from smftools.pipeline import rebasecall_plan
from smftools.pipeline.rebasecall_plan import ParentGeneration
from smftools.pipeline.rebasecall_request import rebasecall_request_from_dict

pytestmark = pytest.mark.unit


def _pod5_index(*read_ids):
    return Pod5DatasetIndex({read_id: ("source-a",) for read_id in read_ids})


def _resolved_dorado(options, tmp_path):
    simplex = DoradoModelArtifact("chem_hac@v1.0.0", "a" * 64, 2, 20)
    return DoradoBasecallResolution(
        selector=options.model,
        dorado_version="1.3.1+test",
        chemistry="chem",
        run_conditions=(DoradoRunCondition("FLOW", "KIT", 5000),),
        simplex_model=simplex,
        modification_models=(),
        model_bundle_digest="b" * 64,
        supported_flags=("--read-ids",),
        capability_digest="c" * 64,
        options=options,
        normalized_argv=("dorado", "basecaller", "chem_hac@v1.0.0"),
        executable_path=tmp_path / "dorado",
        model_directory=tmp_path / "models",
        simplex_path=tmp_path / "models" / simplex.name,
        modification_paths=(),
    )


def _request(mode="qc", *, signal=None, **selection_overrides):
    if mode == "qc":
        selection = {
            "mode": "qc",
            "predicate": {
                "all": [
                    {"column": "passes_read_qc", "op": "eq", "value": True},
                    {"column": "passes_dedup", "op": "eq", "value": True},
                ]
            },
        }
    elif mode == "ids":
        selection = {"mode": "ids", "id_kind": "read_id", "ids": ["r1", "missing"]}
    else:
        selection = {"mode": mode}
    selection.update(selection_overrides)
    source = {"raw_generation": "raw-a"}
    if mode == "qc":
        source["preprocess_generation"] = "pre-a"
    return rebasecall_request_from_dict(
        {
            "schema_version": 1,
            "name": "test-request",
            "source": source,
            "selection": selection,
            "basecall": {"model": "hac@latest"},
            "signal": signal or {"materialize": False},
            "downstream": {"target": "full"},
            "promotion": {"activate": False},
        }
    )


def _install_parent_fixtures(tmp_path, monkeypatch, *, preprocess_source="raw-a"):
    raw_dir = tmp_path / "raw_outputs" / "generations" / "raw-a"
    preprocess_dir = tmp_path / "preprocess_adata_outputs" / "generations" / "pre-a"
    raw_dir.mkdir(parents=True)
    preprocess_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "read_id": ["r1", "r2", "r3"],
            "molecule_uid": ["m1", "m2", "m3"],
        }
    ).to_parquet(raw_dir / "obs.parquet", index=False)
    pd.DataFrame(
        {
            "read_id": ["r1", "r2", "r3"],
            "passes_read_qc": [True, True, False],
            "passes_dedup": [True, False, False],
        }
    ).to_parquet(preprocess_dir / "stage_obs.parquet", index=False)
    signal = tmp_path / "reads.pod5"
    signal.write_bytes(b"pod5")
    source_manifest = ResolvedInputManifest(
        rows=(
            InputManifestRow(
                source_id="source-a",
                path=str(signal),
                sha256=hashlib.sha256(signal.read_bytes()).hexdigest(),
                size_bytes=signal.stat().st_size,
                source_kind="pod5",
                source_role="raw_signal",
            ),
        ),
        digest="manifest-a",
        resolution_method="published",
        base_directory=str(tmp_path),
    )
    raw_parent = ParentGeneration(
        stage="raw",
        selector="raw-a",
        generation_id="raw-a",
        generation_dir=raw_dir,
        manifest={"generation_id": "raw-a"},
    )
    preprocess_parent = ParentGeneration(
        stage="preprocess",
        selector="pre-a",
        generation_id="pre-a",
        generation_dir=preprocess_dir,
        manifest={
            "generation_id": "pre-a",
            "source": {"generation_id": preprocess_source},
        },
    )
    monkeypatch.setattr(rebasecall_plan, "_resolve_raw_parent", lambda *_args: raw_parent)
    monkeypatch.setattr(
        rebasecall_plan, "_resolve_preprocess_parent", lambda *_args: preprocess_parent
    )
    monkeypatch.setattr(rebasecall_plan, "_read_input_manifest", lambda _parent: source_manifest)
    monkeypatch.setattr(
        rebasecall_plan,
        "read_experiment_manifest",
        lambda _root: {"experiment_uid": "uid-a", "experiment_id": "experiment-a"},
    )
    monkeypatch.setattr(
        rebasecall_plan,
        "resolve_dorado_basecall",
        lambda options, *_args: _resolved_dorado(options, tmp_path),
    )
    return SimpleNamespace(
        output_directory=tmp_path,
        experiment_id="experiment-a",
        experiment_name="experiment-a",
        raw_parent=raw_parent,
        preprocess_parent=preprocess_parent,
        source_manifest=source_manifest,
        model_dir=tmp_path / "models",
        device="auto",
    )


def test_qc_plan_resolves_exact_parents_counts_selection_and_writes_nothing(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("qc"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    assert plan.status == "ready"
    assert plan.selection_status == "ready"
    assert plan.raw_parent is not None
    assert plan.raw_parent.generation_id == "raw-a"
    assert plan.raw_parent.molecule_count == 3
    assert plan.preprocess_parent is not None
    assert plan.preprocess_parent.generation_id == "pre-a"
    assert plan.selection.universe_count == 3
    assert plan.selection.selected_count == 1
    assert plan.selection.consumed_columns == ("passes_dedup", "passes_read_qc")
    assert plan.sources.signal_read_count == 5
    assert plan.identity.status == "resolved"
    assert plan.identity.evidence_counts == {"read_id": 1}
    assert plan.to_dict()["execution_status"] == "not_implemented"
    assert plan.to_dict()["selection_freezing"] == {
        "status": "available_during_run_preparation",
        "accepted_plan_id": plan.plan_id,
    }
    assert plan.model.status == "resolved"
    assert plan.model.simplex_model["name"] == "chem_hac@v1.0.0"
    assert plan.to_dict()["requested_model"]["model_bundle_digest"] == "b" * 64
    assert (
        "dorado_and_model_bundle_resolution:srb-04" not in plan.to_dict()["deferred_capabilities"]
    )
    assert (
        "dorado_basecall_execution_and_validation:srb-04b"
        in plan.to_dict()["deferred_capabilities"]
    )
    assert plan.to_json() == plan.to_json()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_all_signal_keeps_signal_and_parent_universes_distinct(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-signal"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    assert plan.status == "ready"
    assert plan.raw_parent is not None
    assert plan.raw_parent.molecule_count == 3
    assert plan.selection.universe_count == 5
    assert plan.selection.selected_count == 5
    assert plan.identity.mode == "signal_inventory"
    assert plan.identity.unique_pod5_read_count == 5
    assert [warning.code for warning in plan.warnings] == ["full_signal_scope"]


def test_exact_model_bundle_changes_accepted_plan_identity(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    def first_resolver(options, *_args):
        return _resolved_dorado(options, tmp_path)

    def changed_resolver(options, *_args):
        original = _resolved_dorado(options, tmp_path)
        return replace(
            original,
            simplex_model=replace(original.simplex_model, sha256="d" * 64),
            model_bundle_digest="e" * 64,
        )

    first = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3"),
        dorado_resolver=first_resolver,
    )
    changed = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3"),
        dorado_resolver=changed_resolver,
    )

    assert first.request.request_id == changed.request.request_id
    assert first.selection.to_dict() == changed.selection.to_dict()
    assert first.plan_id != changed.plan_id


def test_model_resolution_failure_blocks_execution_but_not_selection(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    def fail_resolution(*_args):
        raise DoradoModelError("dorado_model_not_installed", "model bytes are absent")

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3"),
        dorado_resolver=fail_resolution,
    )

    assert plan.status == "blocked"
    assert plan.selection_status == "ready"
    assert plan.model.status == "blocked"
    assert plan.model.failure_code == "dorado_model_not_installed"
    assert "dorado_model_not_installed" in {reason.code for reason in plan.blockers}


def test_signal_inventory_failure_preserves_source_plan_and_qc_selection(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    def fail_inventory(_sources):
        raise ValueError("unreadable POD5")

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("qc"),
        pod5_indexer=fail_inventory,
    )

    assert plan.status == "blocked"
    assert plan.selection_status == "ready"
    assert plan.selection.selected_count == 1
    assert plan.sources.source_count == 1
    assert plan.sources.signal_read_count is None
    assert "signal_inventory_unavailable" in {reason.code for reason in plan.blockers}


def test_checksum_validated_relocation_is_indexed_and_plan_id_is_path_invariant(
    tmp_path, monkeypatch
):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    original = Path(cfg.source_manifest.rows[0].path)
    content = original.read_bytes()
    original.unlink()
    first_relocation = tmp_path / "archive-a" / "renamed.pod5"
    second_relocation = tmp_path / "archive-b" / "other-name.pod5"
    first_relocation.parent.mkdir()
    second_relocation.parent.mkdir()
    first_relocation.write_bytes(content)
    second_relocation.write_bytes(content)
    source = cfg.source_manifest.rows[0]
    indexed_paths = []

    def index_sources(sources):
        indexed_paths.append(sources)
        return _pod5_index("r1", "r2", "r3")

    def relocated_request(path):
        return _request(
            "all-parent-molecules",
            signal={
                "materialize": False,
                "relocations": [
                    {
                        "source_id": source.source_id,
                        "sha256": source.sha256,
                        "path": str(path),
                    }
                ],
            },
        )

    first = rebasecall_plan.build_rebasecall_plan(
        cfg,
        relocated_request(first_relocation),
        pod5_indexer=index_sources,
    )
    second = rebasecall_plan.build_rebasecall_plan(
        cfg,
        relocated_request(second_relocation),
        pod5_indexer=index_sources,
    )

    assert first.status == "ready"
    assert first.sources.resolution_status == "resolved"
    assert first.sources.recorded_paths_available == 0
    assert first.sources.relocation_candidates == 1
    assert first.sources.resolution_evidence_counts == {"explicit_relocation": 1}
    assert indexed_paths == [
        ((source.source_id, first_relocation.resolve()),),
        ((source.source_id, second_relocation.resolve()),),
    ]
    assert first.request.request_id == second.request.request_id
    assert first.plan_id == second.plan_id
    assert (
        "source_checksum_relocation_and_replayability:srb-03"
        not in first.to_dict()["deferred_capabilities"]
    )
    assert (
        "filtered_signal_materialization_and_replayability:srb-03b"
        not in first.to_dict()["deferred_capabilities"]
    )
    assert first.to_dict()["signal_materialization"] == {
        "status": "not_requested",
        "accepted_plan_id": first.plan_id,
    }


def test_checksum_mismatch_blocks_before_pod5_inventory(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    original = Path(cfg.source_manifest.rows[0].path)
    original.unlink()
    wrong = tmp_path / "archive" / "reads.pod5"
    wrong.parent.mkdir()
    wrong.write_bytes(b"different bytes")
    source = cfg.source_manifest.rows[0]

    def unexpected_index(_sources):
        raise AssertionError("checksum-mismatched sources must not be indexed")

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request(
            "all-parent-molecules",
            signal={
                "relocations": [
                    {
                        "source_id": source.source_id,
                        "sha256": source.sha256,
                        "path": str(wrong),
                    }
                ]
            },
        ),
        pod5_indexer=unexpected_index,
    )

    assert plan.status == "blocked"
    assert plan.sources.checksum_mismatch_count == 1
    assert plan.sources.signal_read_count is None
    assert plan.sources.failures[0]["source_id"] == source.source_id
    assert plan.sources.failures[0]["status"] == "checksum_mismatch"
    assert "source_checksum_mismatch" in {reason.code for reason in plan.blockers}


def test_unknown_explicit_relocation_blocks_without_overriding_valid_original(
    tmp_path, monkeypatch
):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request(
            "all-parent-molecules",
            signal={
                "relocations": [
                    {
                        "source_id": "unknown-source",
                        "path": str(tmp_path / "unknown.pod5"),
                    }
                ]
            },
        ),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3"),
    )

    assert plan.sources.resolution_status == "resolved"
    assert plan.sources.unmatched_relocation_count == 1
    assert "source_relocation_unmatched" in {reason.code for reason in plan.blockers}


def test_missing_explicit_id_blocks_without_silently_dropping_it(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("ids"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    assert plan.status == "blocked"
    assert plan.selection.requested_id_count == 2
    assert plan.selection.matched_id_count == 1
    assert plan.selection.missing_ids == ("missing",)
    assert "selection_ids_missing" in {reason.code for reason in plan.blockers}


def test_qc_plan_blocks_when_preprocess_does_not_descend_from_raw(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch, preprocess_source="different-raw")

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("qc"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    assert plan.status == "blocked"
    assert "parent_generation_mismatch" in {reason.code for reason in plan.blockers}


def test_qc_plan_infers_preprocess_parent_from_generation_scoped_source_path(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch, preprocess_source="")
    cfg.preprocess_parent.manifest["source"] = {
        "artifact": {
            "path": "raw_outputs/generations/raw-a/spine.h5ad",
            "path_kind": "relative",
            "anchor": "run_root",
        },
        "generation_id": None,
        "stage": None,
    }

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("qc"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )

    assert plan.status == "ready"
    assert plan.selection.selected_count == 1


def test_nested_cli_emits_human_and_stable_json(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("r1", "r2", "r3", "signal-4", "signal-5"),
    )
    config_path = tmp_path / "experiment.csv"
    request_path = tmp_path / "request.yaml"
    config_path.touch()
    request_path.touch()
    monkeypatch.setattr(rebasecall_plan, "plan_rebasecall", lambda *_args: plan)

    human = CliRunner().invoke(
        cli_entry.cli,
        ["experiment", "rebasecall", "plan", str(config_path), str(request_path)],
    )
    machine = CliRunner().invoke(
        cli_entry.cli,
        [
            "experiment",
            "rebasecall",
            "plan",
            str(config_path),
            str(request_path),
            "--json",
        ],
    )

    assert human.exit_code == 0, human.output
    assert f"Plan ID: {plan.plan_id}" in human.output
    assert "Dorado model: chem_hac@v1.0.0 (1.3.1+test" in human.output
    assert "Execution: unavailable" in human.output
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    assert payload["schema_version"] == 1
    assert payload["selection"]["mode"] == "all-parent-molecules"
    assert payload["identity"]["status"] == "resolved"
    assert payload["requested_model"]["resolution_status"] == "resolved"
    assert payload["execution_status"] == "not_implemented"


def test_historical_split_child_resolves_from_retained_bam_pi(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    retained_bam = tmp_path / "retained.bam"
    retained_bam.write_bytes(b"bam")
    pd.DataFrame(
        {
            "read_id": ["ns6:lane-a:split-child"],
            "source_read_id": ["split-child"],
            "molecule_uid": ["m1"],
            "bam_path": ["retained.bam"],
        }
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("parent"),
        bam_tag_reader=lambda path: {"split-child": {"pi": "parent"}},
    )

    assert plan.status == "ready"
    assert plan.identity.resolved_molecule_count == 1
    assert plan.identity.evidence_counts == {"bam_pi": 1}


def test_unresolved_selected_molecules_have_stable_blocker_and_bounded_evidence(
    tmp_path, monkeypatch
):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("r1"),
    )

    assert plan.status == "blocked"
    assert plan.identity.unresolved_count == 2
    assert len(plan.identity.failures) == 2
    assert "pod5_identity_unresolved" in {reason.code for reason in plan.blockers}


def test_parent_id_selection_retains_all_split_child_rows(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    pd.DataFrame(
        {
            "read_id": ["child-1", "child-2"],
            "molecule_uid": ["m1", "m2"],
            "pod5_read_id": ["parent", "parent"],
        }
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("ids", id_kind="pod5_read_id", ids=["parent"]),
        pod5_indexer=lambda sources: _pod5_index("parent"),
    )

    assert plan.status == "ready"
    assert plan.selection.matched_id_count == 1
    assert plan.selection.selected_count == 2
    assert plan.identity.duplicate_parent_reference_count == 1


def test_unreadable_retained_bam_becomes_stable_identity_blocker(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    retained_bam = tmp_path / "retained.bam"
    retained_bam.write_bytes(b"bad bam")
    pd.DataFrame(
        {"read_id": ["split-child"], "molecule_uid": ["m1"], "bam_path": ["retained.bam"]}
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)

    def fail_bam(_path):
        raise OSError("unreadable")

    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        _request("all-parent-molecules"),
        pod5_indexer=lambda sources: _pod5_index("parent"),
        bam_tag_reader=fail_bam,
    )

    blocker_codes = {reason.code for reason in plan.blockers}
    assert "retained_bam_identity_unavailable" in blocker_codes
    assert "pod5_identity_unresolved" in blocker_codes
