import hashlib

import anndata as ad

from smftools.informatics.experiment_manifest import (
    MANIFEST_SCHEMA_VERSION,
    StageLifecycle,
    artifact_record,
    config_hash,
    experiment_manifest_path,
    read_experiment_manifest,
    record_stage_completion,
    record_stage_state,
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
