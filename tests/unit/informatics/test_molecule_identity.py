import pytest

from smftools.informatics.molecule_identity import (
    molecule_uid,
    new_experiment_uid,
    pooled_obs_name,
    segment_uid,
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


@pytest.mark.unit
def test_segment_identity_is_unique_within_shared_template_and_across_experiments():
    first_experiment = new_experiment_uid()
    second_experiment = new_experiment_uid()

    assert molecule_uid(first_experiment, "template") == molecule_uid(first_experiment, "template")
    assert segment_uid(first_experiment, "template", "R1") != segment_uid(
        first_experiment, "template", "R2"
    )
    assert segment_uid(first_experiment, "template", "R1") != segment_uid(
        second_experiment, "template", "R1"
    )
