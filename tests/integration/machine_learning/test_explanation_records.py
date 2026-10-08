"""MLR-03: out-of-fold explanation records of a published train run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.ml_project import (
    EXPERIMENTS,
    N_POSITIONS,
    READS_PER_BARCODE,
    make_ml_project,
    ml_plan,
)

from smftools.machine_learning.models.sklearn_artifacts import load_published_sklearn_model
from smftools.machine_learning.orchestration import (
    MLJobServiceError,
    bind_ml_job,
    explain_run,
    train_and_publish,
)
from smftools.machine_learning.orchestration import explanations as records
from smftools.machine_learning.orchestration import runs as runs_module
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.integration

MODELS = {
    "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
    "rf": {"backend": "sklearn", "family": "random_forest", "parameters": {"n_estimators": 20}},
    "lr": {"backend": "sklearn", "family": "logistic_regression"},
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_ml_project(tmp_path)


@pytest.fixture
def trained(project: Path):
    bound = bind_ml_job(ml_plan(MODELS), "train", project_dir=project)
    return train_and_publish(bound, project_dir=project)


def _log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def test_naive_bayes_log_odds_rebuild_each_out_of_fold_score(project: Path, trained) -> None:
    explained = explain_run(
        trained.run_id, model="nb", method="NaiveBayesLogOdds", project_dir=project
    )

    index = explained.read(records.ATTRIBUTION_INDEX)
    assert index["per_molecule"] and index["channels"] == ["accessibility"]
    assert len(index["coordinates"]) == N_POSITIONS
    bound = bind_ml_job(ml_plan(MODELS), "train", project_dir=project)
    folds = {fold.fold_name: fold for fold in bound.folds}
    for entry in index["folds"]:
        molecules, matrix = explained.attributions(entry["fold"])
        assert matrix.shape == (len(molecules), 1, N_POSITIONS) and matrix.dtype == np.float32
        model = load_published_sklearn_model(trained.workspace, entry["model_id"])
        data = records._rows(folds[entry["fold"]].dataset, "test", list(molecules["molecule_uid"]))
        log_proba = model.estimator.predict_log_proba(model.transform.transform(data))
        active = list(model.label_schema.class_order).index("active")
        prior = model.estimator.class_log_prior_
        # Positions' contributions plus the prior log odds give the fold
        # model's posterior log odds for each molecule it held out ...
        np.testing.assert_allclose(
            matrix.sum(axis=(1, 2)) + prior[active] - prior[1 - active],
            log_proba[:, active] - log_proba[:, 1 - active],
            rtol=1e-4,
            atol=1e-3,
        )
        # ... which is the run's stored out-of-fold score.
        assert np.array_equal(
            matrix.sum(axis=(1, 2)) > 0, _log_odds(molecules["score"].to_numpy()) > 0
        )


def test_each_molecule_is_explained_by_the_model_that_held_it_out(project: Path, trained) -> None:
    explained = explain_run(
        trained.run_id, model="nb", method="NaiveBayesLogOdds", project_dir=project
    )
    molecules = explained.read(records.MOLECULES)
    membership = trained.read(runs_module.MEMBERSHIP)
    roles = membership.set_index(["fold", "molecule_uid"])["role"]
    for fold, uid in zip(molecules["fold"], molecules["molecule_uid"], strict=True):
        assert roles[(fold, uid)] == "test"
    assert dict(zip(molecules["fold"], molecules["model_id"])) == trained.model_ids["nb"]
    manifest = json.loads((explained.path / "run_manifest.json").read_text())
    assert manifest["action"] == "explain"
    assert sorted(manifest["source_model_ids"]) == sorted(trained.model_ids["nb"].values())
    # Every held-out molecule, once.
    assert len(molecules) == len(EXPERIMENTS) * 2 * READS_PER_BARCODE
    assert molecules["molecule_uid"].is_unique


def test_tree_shap_rows_sum_to_the_model_output(project: Path, trained) -> None:
    shap = pytest.importorskip("shap")
    explained = explain_run(trained.run_id, model="rf", method="TreeSHAP", project_dir=project)
    for entry in explained.read(records.ATTRIBUTION_INDEX)["folds"]:
        molecules, matrix = explained.attributions(entry["fold"])
        model = load_published_sklearn_model(trained.workspace, entry["model_id"])
        active = list(model.label_schema.class_order).index("active")
        expected = np.ravel(shap.TreeExplainer(model.estimator).expected_value)[active]
        np.testing.assert_allclose(
            matrix.sum(axis=(1, 2)) + expected, molecules["score"].to_numpy(), atol=1e-4
        )


def test_importance_is_consistent_across_folds_on_a_planted_signal(project: Path, trained) -> None:
    explained = explain_run(
        trained.run_id, model="nb", method="NaiveBayesLogOdds", project_dir=project
    )
    importance = explained.read(records.IMPORTANCE)
    assert {"mean_abs", "mean", "mean_in_active", "mean_in_inactive"} <= set(
        importance["statistic"]
    )
    # Active molecules are accessible over the first half: there, accessibility
    # pushes active molecules towards "active".
    active = importance[importance["statistic"] == "mean_in_active"]
    first_half = active[active["coordinate"] < N_POSITIONS // 2]["value"].mean()
    second_half = active[active["coordinate"] >= N_POSITIONS // 2]["value"].mean()
    assert first_half > 0 > second_half or first_half > second_half
    consistency = explained.read(records.CONSISTENCY)
    assert len(consistency) == 3  # three fold pairs
    assert explained.summary["nb"]["fold_consistency_spearman"] > 0.3


def test_sampling_is_class_stratified_and_seeded(project: Path, trained) -> None:
    first = explain_run(
        trained.run_id, model="nb", method="NaiveBayesLogOdds", project_dir=project, max_per_fold=10
    )
    second = explain_run(
        trained.run_id, model="nb", method="NaiveBayesLogOdds", project_dir=project, max_per_fold=10
    )
    molecules = first.read(records.MOLECULES)
    counts = molecules.groupby(["fold", "truth"]).size()
    assert (counts == 5).all() and len(counts) == len(EXPERIMENTS) * 2
    assert (
        molecules["molecule_uid"].tolist()
        == second.read(records.MOLECULES)["molecule_uid"].tolist()
    )


def test_global_methods_record_importance_without_matrices(project: Path, trained) -> None:
    explained = explain_run(
        trained.run_id, model="lr", method="LinearCoefficients", project_dir=project
    )
    index = explained.read(records.ATTRIBUTION_INDEX)
    assert not index["per_molecule"]
    assert all(entry["file"] is None for entry in index["folds"])
    assert set(explained.read(records.IMPORTANCE)["statistic"]) == {"value", "abs_value"}
    with pytest.raises(KeyError):
        explained.attributions(index["folds"][0]["fold"])


def test_the_index_lists_the_explain_run(project: Path, trained) -> None:
    explained = explain_run(
        trained.run_id,
        model="nb",
        method="NaiveBayesLogOdds",
        project_dir=project,
        tags={"figure": "promoter"},
    )
    index = json.loads((explained.workspace.index_root / "runs.json").read_text())
    record = {r["run_id"]: r for r in index["records"]}[explained.run_id]
    assert record["action"] == "explain" and record["tags"] == {"figure": "promoter"}
    assert record["summary"]["nb"]["method"] == "NaiveBayesLogOdds"


def test_unknown_model_method_and_changed_data_are_refused(project: Path, trained) -> None:
    with pytest.raises(MLJobServiceError, match="no fold models"):
        explain_run(trained.run_id, model="svm", method="TreeSHAP", project_dir=project)
    with pytest.raises(MLJobServiceError, match="unknown explanation method"):
        explain_run(trained.run_id, model="nb", method="Magic", project_dir=project)
    # Drop one molecule from an experiment's index: the snapshot changes.
    index = next((project.parent / "runs" / EXPERIMENTS[0] / "molecule_index").glob("*.parquet"))
    frame = pd.read_parquet(index)
    frame.iloc[1:].to_parquet(index, index=False)
    with pytest.raises(MLJobServiceError, match="changed since training"):
        explain_run(trained.run_id, model="nb", method="NaiveBayesLogOdds", project_dir=project)


def test_integrated_gradients_for_a_torch_run(project: Path) -> None:
    pytest.importorskip("captum")
    from smftools.machine_learning.orchestration import TorchTrainOptions
    from smftools.machine_learning.training.torch_backend import TorchTrainingConfig

    document = ml_plan(
        {
            "cnn": {
                "backend": "torch",
                "recipe": "residual_dilated_cnn_v1",
                "overrides": {"block_channels": [8, 8], "dilations": [1, 2], "stem_channels": 8},
            }
        }
    ).to_dict()
    document["splits"]["by_experiment"] = {
        "strategy": "explicit_groups",
        "group_by": ["experiment_uid"],
        "train_groups": [EXPERIMENTS[0]],
        "validation_groups": [EXPERIMENTS[1]],
        "test_groups": [EXPERIMENTS[2]],
    }
    bound = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    trained = train_and_publish(
        bound,
        project_dir=project,
        torch_options=TorchTrainOptions(
            training_config=TorchTrainingConfig(max_epochs=2, batch_size=16, device="cpu")
        ),
    )
    explained = explain_run(
        trained.run_id,
        model="cnn",
        method="IntegratedGradients",
        project_dir=project,
        background_size=8,
    )
    (entry,) = explained.read(records.ATTRIBUTION_INDEX)["folds"]
    molecules, matrix = explained.attributions(entry["fold"])
    assert matrix.shape == (2 * READS_PER_BARCODE, 1, N_POSITIONS)
    assert np.isfinite(matrix).all() and np.abs(matrix).sum() > 0
