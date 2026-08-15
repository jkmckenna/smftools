import hashlib

import anndata as ad
import pytest

from smftools import metadata
from smftools._version import __version__
from smftools.constants import SEMANTIC_GRAPH_DEFINITION_VERSION
from smftools.informatics import experiment_manifest
from smftools.informatics.experiment_manifest import (
    MANIFEST_SCHEMA_VERSION,
    StageLifecycle,
    artifact_record,
    config_hash,
    experiment_manifest_path,
    read_experiment_manifest,
    record_stage_completion,
    record_stage_state,
    restore_previous_complete_state,
    stage_is_complete,
    update_experiment_manifest,
)


def test_read_experiment_manifest_empty_when_missing(tmp_path):
    assert read_experiment_manifest(tmp_path) == {}


def test_update_experiment_manifest_merges_fields(tmp_path):
    update_experiment_manifest(tmp_path, experiment="expA", modality="direct")
    update_experiment_manifest(tmp_path, input_data_path="../../data/expA")

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["experiment"] == "expA"
    assert manifest["modality"] == "direct"
    assert manifest["input_data_path"] == "../../data/expA"
    assert experiment_manifest_path(tmp_path).exists()


def test_update_experiment_manifest_skips_none_values(tmp_path):
    update_experiment_manifest(tmp_path, modality="direct")
    update_experiment_manifest(tmp_path, modality=None, fasta_path="./ref.fasta")

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["modality"] == "direct"  # not clobbered by the later None
    assert manifest["fasta_path"] == "./ref.fasta"


def test_record_stage_completion_appends_without_removing_earlier_stages(tmp_path):
    record_stage_completion(tmp_path, "raw", config_hash="abc123", n_molecules=100)
    record_stage_completion(tmp_path, "preprocess", config_hash="def456", n_molecules=80)

    manifest = read_experiment_manifest(tmp_path)
    assert set(manifest["stages"]) == {"raw", "preprocess"}
    assert manifest["stages"]["raw"]["config_hash"] == "abc123"
    assert manifest["stages"]["raw"]["n_molecules"] == 100
    assert "completed_at" in manifest["stages"]["raw"]
    assert manifest["stages"]["preprocess"]["n_molecules"] == 80


def test_smftools_code_identity_uses_installed_version_graph_and_git_commit(monkeypatch):
    monkeypatch.setattr(metadata, "get_git_commit", lambda: "commit-abc")

    assert metadata.smftools_code_identity() == {
        "smftools_version": __version__,
        "graph_definition_version": SEMANTIC_GRAPH_DEFINITION_VERSION,
        "git_commit": "commit-abc",
    }


def test_successful_stage_completion_stamps_top_level_code_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment_manifest,
        "smftools_code_identity",
        lambda: {
            "smftools_version": "2.21.0.dev0",
            "graph_definition_version": 7,
            "git_commit": "commit-one",
        },
    )

    record_stage_completion(tmp_path, "raw")

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["smftools_version"] == "2.21.0.dev0"
    assert manifest["graph_definition_version"] == 7
    assert manifest["git_commit"] == "commit-one"


def test_noncomplete_attempt_preserves_last_successful_code_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment_manifest,
        "smftools_code_identity",
        lambda: {
            "smftools_version": "first-version",
            "graph_definition_version": 1,
            "git_commit": "first-commit",
        },
    )
    record_stage_completion(tmp_path, "spatial")

    def unexpected_identity_read():
        raise AssertionError("noncomplete transitions must not refresh code identity")

    monkeypatch.setattr(experiment_manifest, "smftools_code_identity", unexpected_identity_read)
    record_stage_state(tmp_path, "spatial", "planned")
    record_stage_state(tmp_path, "spatial", "running")
    record_stage_state(tmp_path, "spatial", "failed", outcome="simulated failure")

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["smftools_version"] == "first-version"
    assert manifest["graph_definition_version"] == 1
    assert manifest["git_commit"] == "first-commit"


def test_completion_without_git_metadata_removes_prior_commit(tmp_path, monkeypatch):
    identities = iter(
        (
            {
                "smftools_version": "source-version",
                "graph_definition_version": 1,
                "git_commit": "source-commit",
            },
            {
                "smftools_version": "wheel-version",
                "graph_definition_version": 2,
            },
        )
    )
    monkeypatch.setattr(experiment_manifest, "smftools_code_identity", lambda: next(identities))

    record_stage_completion(tmp_path, "raw")
    record_stage_completion(tmp_path, "preprocess")

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["smftools_version"] == "wheel-version"
    assert manifest["graph_definition_version"] == 2
    assert "git_commit" not in manifest


def test_record_stage_completion_overwrites_same_stage_on_rerun(tmp_path):
    record_stage_completion(tmp_path, "raw", n_molecules=100)
    record_stage_completion(tmp_path, "raw", n_molecules=105)

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["stages"]["raw"]["n_molecules"] == 105


def test_record_stage_completion_and_update_manifest_coexist(tmp_path):
    update_experiment_manifest(tmp_path, experiment="expA")
    record_stage_completion(tmp_path, "raw", n_molecules=100)

    manifest = read_experiment_manifest(tmp_path)
    assert manifest["experiment"] == "expA"
    assert manifest["stages"]["raw"]["n_molecules"] == 100


def test_stage_lifecycle_records_complete_state_and_validates_artifacts(tmp_path):
    spine = tmp_path / "preprocess_adata_outputs" / "spine.h5ad"
    spine.parent.mkdir()
    ad.AnnData().write_h5ad(spine)

    with StageLifecycle(tmp_path, "preprocess", config_hash="abc123") as lifecycle:
        running = read_experiment_manifest(tmp_path)["stages"]["preprocess"]
        assert running["state"] == "running"
        assert "planned_at" in running
        assert "started_at" in running
        lifecycle.complete(
            artifacts={"spine": artifact_record(spine, tmp_path)},
            expected_tasks=3,
            successful_tasks=3,
            schema_versions={"spine": 1},
            timings={"elapsed_seconds": 2.5},
            outcome="success",
        )

    manifest = read_experiment_manifest(tmp_path)
    entry = manifest["stages"]["preprocess"]
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert entry["state"] == "complete"
    assert entry["config_hash"] == "abc123"
    assert entry["expected_tasks"] == entry["successful_tasks"] == 3
    assert stage_is_complete(
        tmp_path,
        "preprocess",
        config_hash="abc123",
        required_artifacts=("spine",),
    )
    assert not stage_is_complete(tmp_path, "preprocess", config_hash="different")

    spine.unlink()
    assert not stage_is_complete(
        tmp_path,
        "preprocess",
        config_hash="abc123",
        required_artifacts=("spine",),
    )


def test_stage_completion_rejects_changed_artifact_and_task_shortfall(tmp_path):
    artifact = tmp_path / "metrics.bin"
    artifact.write_bytes(b"original")
    record = artifact_record(artifact, tmp_path, sha256=hashlib.sha256(b"original").hexdigest())
    with StageLifecycle(tmp_path, "spatial") as lifecycle:
        lifecycle.complete(
            artifacts={"metrics": record},
            expected_tasks=2,
            successful_tasks=1,
        )

    assert not stage_is_complete(tmp_path, "spatial", required_artifacts=("metrics",))

    record_stage_state(tmp_path, "spatial", "planned")
    record_stage_state(tmp_path, "spatial", "running")
    record_stage_state(
        tmp_path,
        "spatial",
        "complete",
        artifacts={"metrics": record},
        expected_tasks=2,
        successful_tasks=2,
    )
    assert stage_is_complete(tmp_path, "spatial", required_artifacts=("metrics",))

    artifact.write_bytes(b"changed!")
    assert not stage_is_complete(tmp_path, "spatial", required_artifacts=("metrics",))


def test_stage_completion_rejects_unreadable_structured_artifact(tmp_path):
    catalog = tmp_path / "task_catalog.parquet"
    catalog.write_bytes(b"not a parquet file")
    with StageLifecycle(tmp_path, "preprocess") as lifecycle:
        lifecycle.complete(artifacts={"task_catalog": artifact_record(catalog, tmp_path)})

    assert not stage_is_complete(
        tmp_path,
        "preprocess",
        required_artifacts=("task_catalog",),
    )


def test_stage_completion_rejects_required_empty_directory(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    record = artifact_record(store, tmp_path, require_nonempty=True)
    with StageLifecycle(tmp_path, "preprocess") as lifecycle:
        lifecycle.complete(artifacts={"store": record})

    assert not stage_is_complete(tmp_path, "preprocess", required_artifacts=("store",))

    (store / "partition.parquet").write_bytes(b"partition")
    assert stage_is_complete(tmp_path, "preprocess", required_artifacts=("store",))


def test_stage_lifecycle_records_failure_without_masking_exception(tmp_path):
    try:
        with StageLifecycle(tmp_path, "spatial", config_hash="abc123"):
            raise RuntimeError("simulated task failure")
    except RuntimeError as exc:
        assert str(exc) == "simulated task failure"
    else:
        raise AssertionError("StageLifecycle suppressed the stage exception")

    entry = read_experiment_manifest(tmp_path)["stages"]["spatial"]
    assert entry["state"] == "failed"
    assert entry["outcome"] == "RuntimeError: simulated task failure"
    assert not stage_is_complete(tmp_path, "spatial")


def test_failed_replacement_preserves_previous_complete_record(tmp_path):
    artifact = tmp_path / "spine.h5ad"
    ad.AnnData().write_h5ad(artifact)
    with StageLifecycle(tmp_path, "latent", config_hash="first") as lifecycle:
        lifecycle.complete(
            artifacts={"spine": artifact_record(artifact, tmp_path)},
            generation_id="generation-one",
        )

    with pytest.raises(RuntimeError, match="replacement failed"):
        with StageLifecycle(tmp_path, "latent", config_hash="second"):
            raise RuntimeError("replacement failed")

    entry = read_experiment_manifest(tmp_path)["stages"]["latent"]
    assert entry["state"] == "failed"
    assert entry["previous_complete"]["state"] == "complete"
    assert entry["previous_complete"]["generation_id"] == "generation-one"
    assert stage_is_complete(
        tmp_path,
        "latent",
        config_hash="first",
        required_artifacts=("spine",),
        extra_matches={"generation_id": "generation-one"},
        allow_previous_complete=True,
    )
    assert not stage_is_complete(
        tmp_path,
        "latent",
        extra_matches={"generation_id": "generation-two"},
        allow_previous_complete=True,
    )


def test_restart_supersedes_an_abandoned_attempt_and_keeps_the_prior_result(tmp_path):
    """A killed run must not leave the experiment permanently unrunnable.

    A process killed mid-stage (OOM, evicted container, hard interrupt) leaves
    the record in `running` with no chance to write a terminal state. Planning
    the next attempt over it is the restart path, and the last complete record
    has to survive the restart -- otherwise a crash silently discards a valid
    published generation and forces a full recompute of work still on disk.
    """
    artifact = tmp_path / "spine.h5ad"
    ad.AnnData().write_h5ad(artifact)
    with StageLifecycle(tmp_path, "raw", config_hash="first") as lifecycle:
        lifecycle.complete(
            artifacts={"spine": artifact_record(artifact, tmp_path)},
            generation_id="generation-one",
        )
    record_stage_state(tmp_path, "raw", "planned", config_hash="second")
    record_stage_state(tmp_path, "raw", "running")

    record_stage_state(tmp_path, "raw", "planned", config_hash="third")

    entry = read_experiment_manifest(tmp_path)["stages"]["raw"]
    assert entry["state"] == "planned"
    assert entry["superseded_attempt"]["state"] == "running"
    assert entry["superseded_attempt"]["started_at"]
    assert entry["previous_complete"]["generation_id"] == "generation-one"
    # The abandoned attempt authorizes nothing; only the prior complete record does.
    assert not stage_is_complete(tmp_path, "raw", config_hash="third")
    assert stage_is_complete(
        tmp_path,
        "raw",
        config_hash="first",
        required_artifacts=("spine",),
        extra_matches={"generation_id": "generation-one"},
        allow_previous_complete=True,
    )


def test_restoring_a_retained_complete_record_makes_the_stage_report_complete(tmp_path):
    """Reusing a retained result must also repair how the stage reads.

    Callers that reuse the retained complete record leave the manifest saying the
    stage is still running, so workflow validation and project discovery report an
    incomplete stage whose artifacts are in fact present and valid.
    """
    artifact = tmp_path / "spine.h5ad"
    ad.AnnData().write_h5ad(artifact)
    with StageLifecycle(tmp_path, "raw", config_hash="first") as lifecycle:
        lifecycle.complete(
            artifacts={"spine": artifact_record(artifact, tmp_path)},
            generation_id="generation-one",
        )
    record_stage_state(tmp_path, "raw", "planned", config_hash="second")
    record_stage_state(tmp_path, "raw", "running")
    assert not stage_is_complete(tmp_path, "raw", required_artifacts=("spine",))

    assert restore_previous_complete_state(tmp_path, "raw")

    entry = read_experiment_manifest(tmp_path)["stages"]["raw"]
    assert entry["state"] == "complete"
    assert entry["generation_id"] == "generation-one"
    assert entry["restored_from_previous_complete"] is True
    assert entry["superseded_attempt"]["state"] == "running"
    assert stage_is_complete(tmp_path, "raw", config_hash="first", required_artifacts=("spine",))
    # Nothing to restore once the live record is the complete one.
    assert not restore_previous_complete_state(tmp_path, "raw")


def test_restore_declines_when_no_complete_record_was_retained(tmp_path):
    record_stage_state(tmp_path, "raw", "planned", config_hash="first")
    record_stage_state(tmp_path, "raw", "running")

    assert not restore_previous_complete_state(tmp_path, "raw")
    assert not restore_previous_complete_state(tmp_path, "preprocess")
    assert read_experiment_manifest(tmp_path)["stages"]["raw"]["state"] == "running"


def test_failure_bookkeeping_never_replaces_the_reported_exception(tmp_path, monkeypatch):
    """The stage's own failure must survive a manifest that cannot be written.

    Recording the failed state is bookkeeping. If it raises -- an unwritable
    manifest, or one another process changed underneath the run -- reporting that
    instead of the real error sends whoever is debugging to the wrong place.
    """
    import smftools.informatics.experiment_manifest as manifest_module

    original = manifest_module.record_stage_state
    calls = []

    def failing_record(run_root, stage, state, **fields):
        if state == "failed":
            calls.append(state)
            raise OSError("manifest is not writable")
        return original(run_root, stage, state, **fields)

    monkeypatch.setattr(manifest_module, "record_stage_state", failing_record)
    with pytest.raises(RuntimeError, match="simulated task failure"):
        with StageLifecycle(tmp_path, "hmm", config_hash="abc123"):
            raise RuntimeError("simulated task failure")

    assert calls == ["failed"]


def test_stage_completion_validates_directory_checksum_and_extra_fields(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "data.bin").write_bytes(b"stable")
    with StageLifecycle(
        tmp_path,
        "latent",
        config_hash="compute",
        input_artifact_ids=["source:one"],
    ) as lifecycle:
        lifecycle.complete(
            artifacts={"store": artifact_record(store, tmp_path, checksum=True)},
            plot_config_hash="plots",
        )

    assert stage_is_complete(
        tmp_path,
        "latent",
        config_hash="compute",
        input_artifact_ids=["source:one"],
        required_artifacts=("store",),
        extra_matches={"plot_config_hash": "plots"},
    )
    assert not stage_is_complete(
        tmp_path,
        "latent",
        input_artifact_ids=["source:two"],
        required_artifacts=("store",),
    )
    assert not stage_is_complete(
        tmp_path,
        "latent",
        required_artifacts=("store",),
        extra_matches={"plot_config_hash": "changed"},
    )

    (store / "data.bin").write_bytes(b"changed")
    assert not stage_is_complete(
        tmp_path,
        "latent",
        required_artifacts=("store",),
    )


def test_stage_lifecycle_requires_explicit_completion(tmp_path):
    with StageLifecycle(tmp_path, "hmm"):
        pass

    entry = read_experiment_manifest(tmp_path)["stages"]["hmm"]
    assert entry["state"] == "failed"
    assert entry["outcome"] == "stage exited without publishing completion"


def test_stage_state_rejects_unknown_state(tmp_path):
    try:
        record_stage_state(tmp_path, "raw", "done")
    except ValueError as exc:
        assert "stage state must be one of" in str(exc)
    else:
        raise AssertionError("unknown lifecycle state was accepted")


def test_stage_state_rejects_transition_that_bypasses_planning(tmp_path):
    record_stage_state(tmp_path, "raw", "planned")
    record_stage_state(tmp_path, "raw", "running")
    record_stage_state(tmp_path, "raw", "failed")

    try:
        record_stage_state(tmp_path, "raw", "running")
    except ValueError as exc:
        assert "'failed' -> 'running'" in str(exc)
    else:
        raise AssertionError("invalid lifecycle transition was accepted")


def test_legacy_completion_record_remains_readable(tmp_path):
    path = experiment_manifest_path(tmp_path)
    path.write_text(
        '{"stages": {"raw": {"completed_at": "2026-01-01T00:00:00+00:00"}}}',
        encoding="utf-8",
    )

    assert stage_is_complete(tmp_path, "raw")
    assert not stage_is_complete(tmp_path, "raw", required_artifacts=("spine",))


def test_config_hash_is_stable_and_key_order_independent():
    a = config_hash({"foo": 1, "bar": 2})
    b = config_hash({"bar": 2, "foo": 1})
    c = config_hash({"foo": 1, "bar": 3})
    assert a == b
    assert a != c


def test_config_hash_handles_non_json_native_values():
    from pathlib import Path

    # Path objects (e.g. from ExperimentConfig.to_dict()) aren't JSON-native --
    # config_hash must not raise.
    h = config_hash({"input_data_path": Path("/some/path"), "n": 3})
    assert isinstance(h, str) and len(h) == 16


def test_config_hash_and_manifest_accept_real_experiment_config(tmp_path):
    """The actual object cli/load_adata.py passes -- ExperimentConfig.to_dict() (an
    asdict() dump, which can carry Path/enum/etc. values, not just plain JSON types)
    -- must round-trip through config_hash() and update_experiment_manifest() without
    special-casing, exactly as the raw-stage wiring in load_adata.py relies on."""
    from smftools.config.experiment_config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.experiment_name = "expA"
    cfg.input_data_path = "/some/data/expA"
    resolved = cfg.to_dict()

    h = config_hash(resolved)
    assert isinstance(h, str) and len(h) == 16

    update_experiment_manifest(tmp_path, experiment="expA", config=resolved)
    manifest = read_experiment_manifest(tmp_path)
    assert manifest["experiment"] == "expA"
    assert manifest["config"]["experiment_name"] == "expA"
