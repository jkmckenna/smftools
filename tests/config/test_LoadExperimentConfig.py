## test_LoadExperimentConfig

from importlib.resources import as_file, files

import pytest

from smftools.config.experiment_config import LoadExperimentConfig

csv_resource = files("smftools.tests._test_inputs").joinpath("test_experiment_config_direct_I.csv")


@pytest.fixture
def config():
    with as_file(csv_resource) as csv_path:
        return LoadExperimentConfig(csv_path)


def test_var_dict(config):
    assert isinstance(config.sef.var_dict, dict)
    assert all(isinstance(k, str) for k in config.sef.var_dict.keys())
