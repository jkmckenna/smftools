"""MLR-05: comparing published runs on shared held-out folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from tests.fixtures.ml_project import EXPERIMENTS, N_POSITIONS, make_ml_project, ml_plan

from smftools.machine_learning.orchestration import (
    MLJobServiceError,
    bind_ml_job,
    compare_runs,
    select_runs,
    train_and_publish,
)
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.integration

MODELS = {
    "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
    "rf": {"backend": "sklearn", "family": "random_forest", "parameters": {"n_estimators": 20}},
}


def _masked(models, window):
    document = ml_plan(models).to_dict()
    document["datasets"]["reads"]["positions"] = {"include": [list(window)]}
    return parse_ml_plan(document)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    project = make_ml_project(tmp_path_factory.mktemp("comparison"))
    full = train_and_publish(
        bind_ml_job(ml_plan(MODELS), "train", project_dir=project),
        project_dir=project,
        tags={"task": "t1", "region": "full"},
    )
    half = train_and_publish(
        bind_ml_job(
            _masked({"nb": MODELS["nb"]}, (0, N_POSITIONS // 2)), "train", project_dir=project
        ),
        project_dir=project,
        tags={"task": "t1", "region": "first_half"},
    )
    return project, full, half


def test_select_runs_by_tags(runs) -> None:
    project, full, half = runs
    selected = select_runs(project_dir=project, tags={"task": "t1"})
    assert set(selected["run_id"]) == {full.run_id, half.run_id}
    only = select_runs(project_dir=project, tags={"region": "first_half"})
    assert list(only["run_id"]) == [half.run_id]
    assert select_runs(project_dir=project, tags={"task": "other"}).empty


def test_fold_metrics_and_paired_differences_equal_direct_computation(runs) -> None:
    project, full, half = runs
    comparison = compare_runs(
        [full.run_id, half.run_id], project_dir=project, reference="full/nb", n_bootstrap=50
    )
    assert list(comparison.entries["entry"]) == ["full/nb", "full/rf", "first_half/nb"]
    assert list(comparison.entries["model_class"]) == ["additive", "tabular_nonlinear", "additive"]
    predictions = pd.read_parquet(full.path / "predictions/test.parquet")
    values = comparison.fold_metrics.set_index(["entry", "fold", "metric"])["value"]
    for (model, fold), rows in predictions.groupby(["model", "held_out"]):
        expected = roc_auc_score(rows["truth"] == "active", rows["p_active"])
        assert values[(f"full/{model}", fold, "roc_auc")] == pytest.approx(expected)
    differences = comparison.fold_differences.set_index(["entry", "fold", "metric"])["difference"]
    for fold in comparison.settings["folds"]:
        assert differences[("full/rf", fold, "roc_auc")] == pytest.approx(
            values[("full/rf", fold, "roc_auc")] - values[("full/nb", fold, "roc_auc")]
        )
    summary = comparison.summary.set_index(["entry", "metric"])
    mean = summary.loc[("full/nb", "roc_auc")]
    assert mean["ci_low"] <= mean["mean"] + 1e-9 and mean["n_folds"] == len(EXPERIMENTS)
    assert (comparison.folds.filter(like="dropped:") == 0).all().all()
    # Between-experiment uncertainty beside the molecule bootstrap.
    difference = comparison.differences.set_index(["entry", "metric"]).loc[("full/rf", "roc_auc")]
    assert difference["fold_ci_low"] <= difference["mean"] <= difference["fold_ci_high"]
    assert 2 / 2 ** len(EXPERIMENTS) <= difference["sign_flip_p"] <= 1
    assert {"fold_ci_low", "fold_ci_high"} <= set(comparison.summary.columns)


def test_the_bootstrap_is_seeded(runs) -> None:
    project, full, half = runs
    first = compare_runs([full.run_id, half.run_id], project_dir=project, n_bootstrap=30, seed=4)
    second = compare_runs([full.run_id, half.run_id], project_dir=project, n_bootstrap=30, seed=4)
    pd.testing.assert_frame_equal(first.differences, second.differences)
    assert first.settings["seed"] == 4 and first.settings["n_bootstrap"] == 30


def test_runs_on_different_folds_compare_on_shared_ones(runs) -> None:
    project, full, _half = runs
    document = ml_plan({"nb": MODELS["nb"]}).to_dict()
    document["splits"]["by_experiment"]["train_groups"] = [EXPERIMENTS[-1]]
    fewer = train_and_publish(
        bind_ml_job(parse_ml_plan(document), "train", project_dir=project),
        project_dir=project,
        tags={"task": "t2"},
    )
    with pytest.warns(UserWarning, match="shared folds only"):
        comparison = compare_runs([full.run_id, fewer.run_id], project_dir=project, n_bootstrap=10)
    assert comparison.settings["dropped_folds"] == [EXPERIMENTS[-1]]
    assert len(comparison.settings["folds"]) == len(EXPERIMENTS) - 1


def test_bad_requests_are_refused(runs) -> None:
    project, full, half = runs
    with pytest.raises(MLJobServiceError, match="unknown comparison metrics"):
        compare_runs([full.run_id], project_dir=project, metrics=["accuracy"])
    with pytest.raises(MLJobServiceError, match="reference"):
        compare_runs([full.run_id], project_dir=project, reference="nope")


def test_write_records_tables_settings_and_figure(runs, tmp_path: Path) -> None:
    project, full, half = runs
    comparison = compare_runs([full.run_id, half.run_id], project_dir=project, n_bootstrap=20)
    out = comparison.write(tmp_path / "cmp", figure_metric="roc_auc")
    settings = json.loads((out / "comparison.json").read_text())
    assert settings["run_ids"] == [full.run_id, half.run_id]
    for name in ("entries", "folds", "fold_metrics", "summary", "differences"):
        assert (out / f"{name}.csv").is_file()
    assert (out / "comparison_roc_auc.png").is_file()
    assert np.isfinite(pd.read_csv(out / "summary.csv")["mean"]).all()
