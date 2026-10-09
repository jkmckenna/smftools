"""Training capped at a molecule count per class (for comparing datasets of
different sizes on equal terms)."""

from __future__ import annotations

import pytest
from tests.fixtures.ml_project import make_ml_project, ml_plan

from smftools.machine_learning.orchestration import bind_ml_job, train_and_publish
from smftools.machine_learning.orchestration.runs import PREDICTIONS
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.integration

MODELS = {
    "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
    "rf": {"backend": "sklearn", "family": "random_forest", "parameters": {"n_estimators": 10}},
}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return make_ml_project(tmp_path_factory.mktemp("capped"))


def _run(project, train=None):
    document = ml_plan(MODELS).to_dict()
    if train is not None:
        document["balancing"] = {"capped": {"train": train}}
        document["jobs"]["train"]["balancing"] = "capped"
    bound = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    return train_and_publish(bound, project_dir=project)


def test_every_fold_trains_on_the_capped_counts_and_tests_on_everything(project) -> None:
    natural = _run(project)
    capped = _run(project, {"method": "downsample", "max_per_class": 5, "seed": 0})
    models = capped.read("models.json")
    assert {tuple(item["train_class_counts"]) for item in models} == {(5, 5)}
    assert {item["n_train"] for item in models} == {10}
    assert all(item["n_train"] > 10 for item in natural.read("models.json"))

    # Test sets are untouched: the same molecules are scored either way.
    def scored(run):
        table = run.read(PREDICTIONS)
        return sorted(zip(table["model"], table["molecule_uid"]))

    assert scored(capped) == scored(natural)


def test_cap_seeds_draw_different_cohorts(project) -> None:
    first = _run(project, {"method": "downsample", "max_per_class": 5, "seed": 1})
    second = _run(project, {"method": "downsample", "max_per_class": 5, "seed": 2})
    again = _run(project, {"method": "downsample", "max_per_class": 5, "seed": 1})

    def scores(run):
        table = run.read(PREDICTIONS)
        return table.sort_values(["model", "molecule_uid"])["p_active"].to_numpy()

    assert (scores(first) == scores(again)).all()
    assert not (scores(first) == scores(second)).all()


def test_the_job_policy_reaches_backends_given_explicit_options(project) -> None:
    pytest.importorskip("torch")
    from smftools.machine_learning.orchestration import (
        MLJobServiceError,
        SklearnTrainOptions,
        TorchTrainOptions,
    )
    from smftools.machine_learning.plan import BalanceRoleSpec, BalancingSpec
    from smftools.machine_learning.training.torch_backend import TorchTrainingConfig

    models = {
        **MODELS,
        "scan": {
            "backend": "torch",
            "recipe": "motif_scanner_k21_v1",
            "overrides": {"filters": [2], "kernel_sizes": [5]},
        },
    }
    document = ml_plan(models).to_dict()
    document["splits"]["by_experiment"]["validation_fraction"] = 0.2
    document["balancing"] = {"capped": {"train": {"method": "downsample", "max_per_class": 4}}}
    document["jobs"]["train"]["balancing"] = "capped"
    bound = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    torch_options = TorchTrainOptions(
        training_config=TorchTrainingConfig(max_epochs=2, batch_size=4, device="cpu")
    )
    run = train_and_publish(
        bound,
        project_dir=project,
        sklearn_options=SklearnTrainOptions(seed=0),
        torch_options=torch_options,
    )
    assert {tuple(item["train_class_counts"]) for item in run.read("models.json")} == {(4, 4)}
    conflicting = SklearnTrainOptions(balancing=BalancingSpec(train=BalanceRoleSpec("natural")))
    with pytest.raises(Exception) as failed:  # noqa: PT011 -- the job wraps the cause
        train_and_publish(bound, project_dir=project, sklearn_options=conflicting)
    causes = []
    cause = failed.value
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    assert any(
        isinstance(item, MLJobServiceError) and "but the job declares" in str(item)
        for item in causes
    )
