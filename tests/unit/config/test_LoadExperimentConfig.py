## test_LoadExperimentConfig

from importlib.resources import as_file, files

import pytest

from smftools.config.experiment_config import LoadExperimentConfig

csv_resource = files("tests._test_inputs").joinpath("test_experiment_config_direct_I.csv")


@pytest.fixture
def config():
    with as_file(csv_resource) as csv_path:
        return LoadExperimentConfig(csv_path)


def test_var_dict(config):
    assert isinstance(config.var_dict, dict)
    assert all(isinstance(k, str) for k in config.var_dict.keys())


def test_latent_partitioned_config_defaults_and_bool_parsing(tmp_path):
    from smftools.cli.helpers import load_experiment_config

    with as_file(csv_resource) as csv_path:
        rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    rows.extend(
        [
            "latent_execution_mode,partitioned,,,str",
            "latent_max_fit_reads,123,,,int",
            "latent_run_cp,False,,,bool",
        ]
    )
    config_path = tmp_path / "latent_config.csv"
    config_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    cfg = load_experiment_config(str(config_path))

    assert cfg.latent_execution_mode == "partitioned"
    assert cfg.latent_max_fit_reads == 123
    assert cfg.latent_run_cp is False
    assert cfg.latent_n_pcs == 10
