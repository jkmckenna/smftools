"""MLR-07: a validation fraction inside each leave-one-group-out fold."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.ml_project import (
    EXPERIMENTS,
    READS_PER_BARCODE,
    make_ml_project,
    ml_plan,
)

from smftools.machine_learning.manifests import MLManifestError, SplitManifest
from smftools.machine_learning.orchestration import bind_ml_job, train_and_publish
from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    return make_ml_project(tmp_path_factory.mktemp("validation"))


def _plan(models=None, **split):
    document = ml_plan(models).to_dict()
    document["splits"]["by_experiment"].update(split)
    return parse_ml_plan(document)


def _experiment_of(bound) -> dict[str, str]:
    return {item.molecule_uid: item.experiment_uid for item in bound.snapshot.observations}


def _class_of(bound) -> dict[str, int]:
    return {item.molecule_uid: item.class_id for item in bound.snapshot.observations}


def test_test_stays_one_whole_experiment_and_validation_comes_from_training(project) -> None:
    bound = bind_ml_job(_plan(validation_fraction=0.25), "train", project_dir=project)
    experiment_of, class_of = _experiment_of(bound), _class_of(bound)
    for fold in bound.folds:
        roles = pd.Series(dict(fold.resolution.assignments))
        test_experiments = {experiment_of[uid] for uid in roles[roles == "test"].index}
        assert len(test_experiments) == 1
        assert len(roles[roles == "test"]) == 2 * READS_PER_BARCODE  # the whole experiment
        rest = roles[roles != "test"].index
        assert not test_experiments & {experiment_of[uid] for uid in rest}
        # A quarter of every training experiment x class cell is validation.
        frame = pd.DataFrame(
            {
                "role": roles[rest],
                "experiment": [experiment_of[uid] for uid in rest],
                "class_id": [class_of[uid] for uid in rest],
            }
        )
        counts = frame.groupby(["experiment", "class_id"])["role"].apply(
            lambda r: (r == "validation").sum()
        )
        assert (counts == round(0.25 * READS_PER_BARCODE)).all()
        assert fold.split.shared_roles == ("train", "validation")
        assert fold.resolution.shared_roles == ("train", "validation")


def test_validation_is_seeded(project) -> None:
    def validation(seed):
        bound = bind_ml_job(
            _plan(validation_fraction=0.25, seed=seed), "train", project_dir=project
        )
        return [
            sorted(u for u, r in fold.resolution.assignments.items() if r == "validation")
            for fold in bound.folds
        ]

    assert validation(1) == validation(1)
    assert validation(1) != validation(2)


def test_group_level_validation_holds_out_whole_training_experiments(project) -> None:
    bound = bind_ml_job(
        _plan(validation_fraction=0.25, validation_by="groups"), "train", project_dir=project
    )
    experiment_of = _experiment_of(bound)
    for fold in bound.folds:
        roles = pd.Series(dict(fold.resolution.assignments))
        by_experiment = roles.groupby([experiment_of[uid] for uid in roles.index]).agg(set)
        assert all(len(found) == 1 for found in by_experiment)  # no experiment is split
        assert sorted(next(iter(found)) for found in by_experiment) == [
            "test",
            "train",
            "validation",
        ]
        assert fold.split.shared_roles == ()


def test_without_the_option_folds_are_unchanged(project) -> None:
    plain = bind_ml_job(ml_plan(), "train", project_dir=project)
    assert "validation_fraction" not in json.dumps(ml_plan().to_dict())
    for fold in plain.folds:
        assert set(fold.resolution.assignments.values()) == {"train", "test"}
        assert fold.split.shared_roles == () and "shared_roles" not in fold.split.to_dict()


def test_the_isolation_rule(project) -> None:
    bound = bind_ml_job(_plan(validation_fraction=0.25), "train", project_dir=project)
    fold = bound.folds[0]
    assignments = dict(fold.resolution.assignments)
    # Declared: train and validation share experiments.
    SplitManifest.create(
        dataset=bound.snapshot,
        group_by=fold.split.group_by,
        assignments=assignments,
        shared_roles=("train", "validation"),
    )
    # Undeclared: refused.
    with pytest.raises(MLManifestError, match="occurs in both"):
        SplitManifest.create(
            dataset=bound.snapshot, group_by=fold.split.group_by, assignments=assignments
        )
    # Test leaking into a training experiment: refused even when sharing is declared.
    test_uid = next(uid for uid, role in assignments.items() if role == "test")
    train_uid = next(uid for uid, role in assignments.items() if role == "train")
    leaked = {**assignments, train_uid: "test"}
    with pytest.raises(MLManifestError, match="occurs in both"):
        SplitManifest.create(
            dataset=bound.snapshot,
            group_by=fold.split.group_by,
            assignments=leaked,
            shared_roles=("train", "validation"),
        )
    with pytest.raises(MLManifestError, match="shared_roles"):
        SplitManifest.create(
            dataset=bound.snapshot,
            group_by=fold.split.group_by,
            assignments={**assignments, test_uid: "test"},
            shared_roles=("train", "test"),
        )
    # A declared manifest round-trips.
    restored = SplitManifest.from_dict(fold.split.to_dict(), dataset=bound.snapshot)
    assert restored.split_id == fold.split.split_id


@pytest.mark.parametrize(
    ("split", "message"),
    [
        ({"validation_fraction": 1.5}, "between zero and one"),
        ({"validation_by": "groups"}, "needs validation_fraction"),
        ({"validation_fraction": 0.2, "validation_by": "rows"}, "molecules' or 'groups"),
    ],
)
def test_bad_declarations_are_refused(split, message) -> None:
    with pytest.raises(MLPlanValidationError, match=message):
        _plan(**split)


def test_validation_fraction_is_leave_one_group_out_only() -> None:
    document = ml_plan().to_dict()
    document["splits"]["by_experiment"] = {
        "strategy": "explicit_groups",
        "group_by": ["experiment_uid"],
        "train_groups": [EXPERIMENTS[0]],
        "validation_groups": [EXPERIMENTS[1]],
        "test_groups": [EXPERIMENTS[2]],
        "validation_fraction": 0.2,
    }
    with pytest.raises(MLPlanValidationError, match="only to leave_one_group_out"):
        parse_ml_plan(document)


def test_torch_trains_per_fold_and_as_a_final_model_with_validation(project) -> None:
    pytest.importorskip("torch")
    from smftools.machine_learning.orchestration import TorchTrainOptions
    from smftools.machine_learning.training.torch_backend import TorchTrainingConfig

    models = {
        "cnn": {
            "backend": "torch",
            "recipe": "residual_dilated_cnn_v1",
            "overrides": {"block_channels": [8, 8], "dilations": [1, 2], "stem_channels": 8},
        }
    }
    bound = bind_ml_job(_plan(models, validation_fraction=0.25), "train", project_dir=project)
    run = train_and_publish(
        bound,
        project_dir=project,
        final_model=True,
        torch_options=TorchTrainOptions(
            training_config=TorchTrainingConfig(max_epochs=2, batch_size=16, device="cpu")
        ),
    )
    assert set(run.model_ids["cnn"]) == {fold.fold_name for fold in bound.folds} | {"final"}
    history = run.read("history.parquet")
    # Early stopping reads the validation role: every fold records its loss.
    validation = history[history["metric"] == "validation_loss"]
    assert set(validation["fold"]) == set(run.model_ids["cnn"])
    splits = {split["fold"]: split for split in run.read("data/splits.json")}
    assert set(splits["final"]["n_by_role"]) == {"train", "validation"}
    for fold in bound.folds:
        assert set(splits[fold.fold_name]["n_by_role"]) == {"train", "validation", "test"}
