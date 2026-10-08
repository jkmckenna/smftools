"""MLR-01: a bound train job publishes one self-describing run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.ml_project import EXPERIMENTS, READS_PER_BARCODE, make_ml_project, ml_plan

from smftools.machine_learning.artifacts import validate_published_bundle
from smftools.machine_learning.evaluation import average_precision_at_prevalence
from smftools.machine_learning.models.sklearn_artifacts import load_published_sklearn_model
from smftools.machine_learning.orchestration import (
    MLJobExecutionError,
    apply_partition_model,
    bind_ml_job,
    train_and_publish,
)
from smftools.machine_learning.orchestration import runs as runs_module
from smftools.machine_learning.workspace import resolve_ml_workspace

pytestmark = pytest.mark.integration

MODELS = {
    "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
    "rf": {"backend": "sklearn", "family": "random_forest", "parameters": {"n_estimators": 20}},
}
TAGS = {"task": "fixture/active-vs-inactive", "channel": "accessibility"}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_ml_project(tmp_path)


@pytest.fixture
def published(project: Path):
    bound = bind_ml_job(ml_plan(MODELS), "train", project_dir=project)
    return bound, train_and_publish(bound, project_dir=project, tags=TAGS)


def test_every_record_is_published_and_validates(published) -> None:
    bound, run = published
    workspace = run.workspace

    validate_published_bundle(workspace, run.path, kind="run", expected_id=run.run_id)
    manifest = json.loads((run.path / "run_manifest.json").read_text())
    assert manifest["state"] == "completed"
    assert manifest["model_keys"] == ["nb", "rf"]
    assert manifest["dataset_snapshot_id"] == bound.snapshot.snapshot_id
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert {role for role, _path, _type in runs_module._PAYLOADS} <= roles
    assert run.read(runs_module.TAGS) == dict(sorted(TAGS.items()))

    models = run.read(runs_module.MODELS)
    assert len(models) == len(MODELS) * len(EXPERIMENTS)
    for record in models:
        validate_published_bundle(
            workspace,
            workspace.model_dir(record["model_id"]),
            kind="model",
            expected_id=record["model_id"],
        )
        model_manifest = json.loads(
            (workspace.model_dir(record["model_id"]) / "model_manifest.json").read_text()
        )
        assert model_manifest["originating_run_id"] == run.run_id
    assert {(m["model"], m["fold"]) for m in models} == {
        (name, fold.fold_name) for name in MODELS for fold in bound.folds
    }


def test_membership_is_each_folds_split(published) -> None:
    bound, run = published
    membership = run.read(runs_module.MEMBERSHIP)
    for fold in bound.folds:
        rows = membership[membership["fold"] == fold.fold_name]
        assert dict(zip(rows["molecule_uid"], rows["role"])) == dict(fold.resolution.assignments)
    splits = {split["fold"]: split for split in run.read(runs_module.SPLITS)}
    for fold in bound.folds:
        assert splits[fold.fold_name]["split_id"] == fold.split.split_id
        assert splits[fold.fold_name]["n_by_role"]["test"] == 2 * READS_PER_BARCODE


def test_reloaded_fold_model_reproduces_its_predictions(published) -> None:
    bound, run = published
    predictions = run.read(runs_module.PREDICTIONS)
    for fold in bound.folds:
        model_id = run.model_ids["rf"][fold.fold_name]
        model = load_published_sklearn_model(run.workspace, model_id)
        reapplied = [
            apply_partition_model(model, batch, phase="test", model_id=model_id)
            for batch in fold.dataset.iter_batches("test")
        ]
        uids = [uid for part in reapplied for uid in part.molecule_uids]
        active = list(reapplied[0].class_order).index("active")
        scores = np.concatenate([np.asarray(part.probabilities)[:, active] for part in reapplied])
        stored = predictions[
            (predictions["model"] == "rf") & (predictions["fold"] == fold.fold_name)
        ]
        stored = stored.set_index("molecule_uid").loc[uids]
        np.testing.assert_allclose(stored["p_active"].to_numpy(), scores)


def test_fixed_prevalence_metric_matches_direct_computation(published) -> None:
    _bound, run = published
    predictions = run.read(runs_module.PREDICTIONS)
    metrics = run.read(runs_module.METRICS)
    for (model, fold), rows in predictions.groupby(["model", "fold"]):
        expected = average_precision_at_prevalence(
            (rows["truth"] == "active").to_numpy(), rows["p_active"].to_numpy(), prevalence=0.10
        )
        recorded = metrics[(metrics["model"] == model) & (metrics["fold"] == fold)].set_index(
            "name"
        )
        assert recorded.loc["normalized_average_precision_at_prevalence_reweighted", "value"] == (
            pytest.approx(expected.normalized_reweighted)
        )
        assert recorded.loc["average_precision_at_prevalence_subsampled", "value"] == (
            pytest.approx(expected.subsampled)
        )
        assert recorded.loc["average_precision_at_prevalence_reweighted", "prevalence"] == 0.10


def test_summary_is_the_mean_over_folds(published) -> None:
    _bound, run = published
    metrics = run.read(runs_module.METRICS)
    pooled = metrics[
        (metrics["scope"] == "pooled")
        & (metrics["class_name"] == "active")
        & (metrics["name"] == "roc_auc")
    ]
    for model in MODELS:
        values = pooled[pooled["model"] == model]["value"]
        entry = run.summary[model]["roc_auc"]
        assert entry["mean"] == pytest.approx(values.mean())
        assert entry["n_folds"] == len(EXPERIMENTS)
    # The fixture signal separates the classes almost perfectly.
    assert run.summary["nb"]["average_precision"]["mean"] > 0.95


def test_each_call_publishes_a_new_run_and_the_index_lists_both(project: Path, published) -> None:
    bound, first = published
    second = train_and_publish(bound, project_dir=project, tags={"task": "again"})

    assert second.run_id != first.run_id
    assert set(second.model_ids["nb"].values()).isdisjoint(first.model_ids["nb"].values())
    workspace = resolve_ml_workspace(project_dir=project)
    index = json.loads((workspace.index_root / "runs.json").read_text())
    records = {record["run_id"]: record for record in index["records"]}
    assert records[first.run_id]["tags"] == dict(sorted(TAGS.items()))
    assert records[second.run_id]["tags"] == {"task": "again"}
    assert records[first.run_id]["summary"]["rf"]["roc_auc"]["n_folds"] == len(EXPERIMENTS)


def test_a_failed_training_publishes_a_failed_run(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = bind_ml_job(ml_plan(), "train", project_dir=project)

    def failing(*args, **kwargs):
        raise RuntimeError("fold exploded")
        yield  # a generator, as the real one

    monkeypatch.setattr(runs_module, "iter_bound_train_job", failing)
    with pytest.raises(MLJobExecutionError) as caught:
        train_and_publish(bound, project_dir=project, rebuild_index=False)

    manifest = caught.value.outcome.manifest
    assert manifest.state == "failed"
    assert "fold exploded" in manifest.failure.message
    workspace = resolve_ml_workspace(project_dir=project)
    assert (workspace.runs_root / manifest.run_id / "run_manifest.json").is_file()


def test_requires_exactly_one_destination(project: Path) -> None:
    bound = bind_ml_job(ml_plan(), "train", project_dir=project)
    with pytest.raises(Exception, match="exactly one"):
        train_and_publish(bound)


def test_metric_tables_are_long_form(published) -> None:
    _bound, run = published
    curves = run.read(runs_module.CURVES)
    history = run.read(runs_module.HISTORY)
    assert {"roc", "precision_recall"} & set(curves["kind"])
    kinds = history.groupby("model")["event_type"].agg(set)
    assert kinds["nb"] == {"partial_fit_completed"}  # naive Bayes streams
    assert kinds["rf"] == {"fit_completed"}
    assert isinstance(run.read(runs_module.METRICS), pd.DataFrame)


def test_a_torch_model_publishes_with_its_epoch_history(project: Path) -> None:
    pytest.importorskip("torch")
    from smftools.machine_learning.models.torch_artifacts import load_published_torch_model
    from smftools.machine_learning.orchestration import TorchTrainOptions
    from smftools.machine_learning.training.torch_backend import TorchTrainingConfig

    models = {
        "cnn": {
            "backend": "torch",
            "recipe": "residual_dilated_cnn_v1",
            "overrides": {"block_channels": [8, 8], "dilations": [1, 2], "stem_channels": 8},
        }
    }
    # Torch training stops early on a validation role, which leave-one-group-out
    # lacks: one experiment each for train, validation and test.
    from smftools.machine_learning.plan import parse_ml_plan

    document = ml_plan(models).to_dict()
    document["splits"]["by_experiment"] = {
        "strategy": "explicit_groups",
        "group_by": ["experiment_uid"],
        "train_groups": [EXPERIMENTS[0]],
        "validation_groups": [EXPERIMENTS[1]],
        "test_groups": [EXPERIMENTS[2]],
    }
    bound = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    run = train_and_publish(
        bound,
        project_dir=project,
        torch_options=TorchTrainOptions(
            training_config=TorchTrainingConfig(max_epochs=2, batch_size=16, device="cpu")
        ),
    )

    history = run.read(runs_module.HISTORY)
    assert set(history["metric"]) >= {"train_loss"}
    assert set(history["metric"]) >= {"train_loss", "validation_loss"}
    assert history["epoch"].max() <= 2
    assert run.read(runs_module.SPLITS)[0]["n_by_role"].keys() == {"train", "validation", "test"}
    for model_id in run.model_ids["cnn"].values():
        load_published_torch_model(run.workspace, model_id)
    assert run.read(runs_module.MODELS)[0]["backend"] == "torch"
