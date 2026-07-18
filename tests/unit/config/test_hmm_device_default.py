from smftools.config.experiment_config import ExperimentConfig


def test_hmm_device_defaults_to_cpu_not_auto(tmp_path):
    # HMM fitting runs a small-state sequential position loop where GPU
    # dispatch overhead dominates (measured ~1.5x slower per-iteration on MPS
    # than CPU on real data, plus GPU fits are forced fully sequential across
    # tasks) -- so hmm_device should default to "cpu" even though the general
    # `device` setting still defaults to "auto" for GPU-friendly stages.
    input_bam = tmp_path / "input.bam"
    input_bam.write_text("stub")
    output_dir = tmp_path / "outputs"

    cfg, _ = ExperimentConfig.from_var_dict(
        {
            "input_data_path": str(input_bam),
            "output_directory": str(output_dir),
            "experiment_name": "test_experiment",
        },
        merge_with_defaults=False,
    )

    assert cfg.hmm_device == "cpu"
    assert cfg.device == "auto"


def test_hmm_device_can_be_overridden_explicitly(tmp_path):
    input_bam = tmp_path / "input.bam"
    input_bam.write_text("stub")
    output_dir = tmp_path / "outputs"

    cfg, _ = ExperimentConfig.from_var_dict(
        {
            "input_data_path": str(input_bam),
            "output_directory": str(output_dir),
            "experiment_name": "test_experiment",
            "hmm_device": "mps",
        },
        merge_with_defaults=False,
    )

    assert cfg.hmm_device == "mps"
