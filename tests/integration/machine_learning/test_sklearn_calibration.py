"""Platt calibration of sklearn models on validation molecules."""

from __future__ import annotations

import numpy as np
import pytest
from tests.fixtures.ml_project import make_ml_project, ml_plan

from smftools.machine_learning.models.sklearn_artifacts import load_published_sklearn_model
from smftools.machine_learning.orchestration import bind_ml_job, explain_run, train_and_publish
from smftools.machine_learning.orchestration.runs import PREDICTIONS
from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

pytestmark = pytest.mark.integration

MODELS = {
    "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
    "rf": {"backend": "sklearn", "family": "random_forest", "parameters": {"n_estimators": 20}},
}


def _document(calibrated: bool) -> dict:
    document = ml_plan(MODELS).to_dict()
    document["splits"]["by_experiment"]["validation_fraction"] = 0.3
    if calibrated:
        for spec in document["models"].values():
            spec["calibration"] = "sigmoid"
    return document


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    project = make_ml_project(tmp_path_factory.mktemp("calibration"))
    published = {}
    for calibrated in (False, True):
        bound = bind_ml_job(parse_ml_plan(_document(calibrated)), "train", project_dir=project)
        published[calibrated] = train_and_publish(bound, project_dir=project)
    return project, published


def _logit(table):
    p = np.clip(table["p_active"].to_numpy(), 1e-300, 1)
    q = np.clip(table["p_inactive"].to_numpy(), 1e-300, 1)
    return np.log(p) - np.log(q)


def test_calibration_keeps_the_ranking_and_tames_the_scale(runs) -> None:
    _project, published = runs
    plain, calibrated = (published[k].read(PREDICTIONS) for k in (False, True))
    for model in MODELS:
        a = plain[plain["model"] == model].sort_values("molecule_uid")
        b = calibrated[calibrated["model"] == model].sort_values("molecule_uid")
        assert list(a["molecule_uid"]) == list(b["molecule_uid"])
        for fold in a["fold"].unique():
            x = a[a["fold"] == fold]["p_active"].to_numpy()
            y = b[b["fold"] == fold]["p_active"].to_numpy()
            # Monotone per fold: a molecule scored higher before is not scored
            # lower after. (Plain probabilities saturate at 0 / 1 in float, so
            # compare groups of equal plain score; calibrated scores, from exact
            # log-odds, may still differ inside such a tie.)
            values = np.unique(x)
            highest = [y[x == value].max() for value in values]
            lowest = [y[x == value].min() for value in values]
            assert all(top <= bottom + 1e-12 for top, bottom in zip(highest[:-1], lowest[1:]))
        if model == "nb":
            assert np.abs(_logit(b)).max() < np.abs(_logit(a)).max()

    def auroc(run):
        table = run.read("metrics.parquet")
        table = table[(table["name"] == "roc_auc") & (table["scope"] == "pooled")]
        return table.sort_values(["model", "fold"])["value"].to_numpy()

    # The ranking, hence AUROC, is unchanged.
    assert np.allclose(auroc(published[False]), auroc(published[True]))


def test_calibration_is_published_and_reloaded(runs) -> None:
    _project, published = runs
    run = published[True]
    for model, folds in run.model_ids.items():
        fitted = load_published_sklearn_model(run.workspace, next(iter(folds.values())))
        assert fitted.calibration["method"] == "sigmoid"
        assert 0 < fitted.calibration["slope"] and fitted.calibration["n_validation"] > 0
    plain = published[False]
    fitted = load_published_sklearn_model(
        plain.workspace, next(iter(plain.model_ids["nb"].values()))
    )
    assert fitted.calibration is None


def test_explanations_use_the_raw_estimator(runs) -> None:
    project, published = runs
    explained = explain_run(
        published[True].run_id, model="nb", project_dir=project, max_per_fold=20, figure=False
    )
    entry = explained.read("attributions/index.json")["folds"][0]
    _molecules, matrix = explained.attributions(entry["fold"])
    assert np.isfinite(matrix).all()


def test_calibration_needs_validation_molecules_and_sklearn() -> None:
    document = _document(True)
    document["splits"]["by_experiment"].pop("validation_fraction")
    with pytest.raises(MLPlanValidationError, match="validation"):
        parse_ml_plan(document)
    document = ml_plan({"cnn": {"backend": "torch", "recipe": "motif_scanner_k21_v1"}}).to_dict()
    document["models"]["cnn"]["calibration"] = "sigmoid"
    with pytest.raises(MLPlanValidationError, match="sklearn"):
        parse_ml_plan(document)
    plain = parse_ml_plan(_document(False)).to_dict()
    assert all("calibration" not in model for model in plain["models"].values())
