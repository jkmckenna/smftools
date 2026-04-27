import pandas as pd

from smftools.config.experiment_config import ExperimentConfig


def test_skip_bam_qc_defaults_false(tmp_path):
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
    assert cfg.skip_bam_qc is False


def test_skip_bam_qc_parses_true_from_csv(tmp_path):
    input_bam = tmp_path / "input.bam"
    input_bam.write_text("stub")
    output_dir = tmp_path / "outputs"

    cfg_df = pd.DataFrame(
        [
            {"variable": "input_data_path", "value": str(input_bam)},
            {"variable": "output_directory", "value": str(output_dir)},
            {"variable": "experiment_name", "value": "test_experiment"},
            {"variable": "skip_bam_qc", "value": "True"},
        ]
    )
    cfg, _ = ExperimentConfig.from_csv(cfg_df, merge_with_defaults=False)
    assert cfg.skip_bam_qc is True
