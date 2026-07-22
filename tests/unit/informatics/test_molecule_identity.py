import pytest

from smftools.informatics.molecule_identity import (
    molecule_uid,
    new_experiment_uid,
    pooled_obs_name,
    split_pooled_obs_name,
)


@pytest.mark.unit
def test_molecule_identity_is_deterministic_and_reversible():
    experiment_uid = new_experiment_uid()
    read_id = "read.with|delimiters/and unicode μ"

    assert molecule_uid(experiment_uid, read_id) == molecule_uid(experiment_uid, read_id)
    assert molecule_uid(experiment_uid, read_id) != molecule_uid(experiment_uid, read_id + "x")
    encoded = pooled_obs_name(experiment_uid, read_id)
    assert split_pooled_obs_name(encoded) == (experiment_uid, read_id)


@pytest.mark.unit
def test_pooled_obs_name_rejects_invalid_values():
    with pytest.raises(ValueError, match="invalid pooled"):
        split_pooled_obs_name("bare-read-id")
