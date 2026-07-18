from smftools.informatics.experiment_manifest import (
    config_hash,
    experiment_manifest_path,
    read_experiment_manifest,
    record_stage_completion,
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
