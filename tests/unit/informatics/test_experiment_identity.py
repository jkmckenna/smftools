import pytest

from smftools.informatics.experiment_identity import resolve_experiment_id


def test_resolve_experiment_id_accepts_matching_nonempty_candidates():
    assert (
        resolve_experiment_id(
            {
                "experiment_id": "experiment-a",
                "experiment_name": "experiment-a",
                "empty": "",
            }
        )
        == "experiment-a"
    )


def test_resolve_experiment_id_reports_conflicting_sources():
    with pytest.raises(
        ValueError,
        match="experiment_id='experiment-a'.*experiment_name='experiment-b'",
    ):
        resolve_experiment_id(
            {
                "experiment_id": "experiment-a",
                "experiment_name": "experiment-b",
            }
        )


def test_resolve_experiment_id_can_require_a_candidate():
    with pytest.raises(ValueError, match="experiment identity is required"):
        resolve_experiment_id({"experiment_id": None}, required=True)
