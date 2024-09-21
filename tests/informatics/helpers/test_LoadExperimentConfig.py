## test_LoadExperimentConfig

import pytest
from smftools.inform.helpers import LoadExperimentConfig

csv_path = "/tests/_test_inputs/test_experiment_config_direct_I.csv"

@pytest.fixture
def config():
    # Fixture to initialize the LoadExperimentConfig instance
    return LoadExperimentConfig(csv_path)

def test_var_dict(config):
    assert type(config.sef.var_dict) == dict
    
    key_types = [type(key) == str for key in config.sef.var_dict.keys()]
    assert False not in key_types