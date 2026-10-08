"""MLR-02: final models fit on every group, and saved models applied to new data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    apply_and_publish,
    bind_ml_job,
    train_and_publish,
)
from smftools.machine_learning.orchestration import runs as runs_module
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.integration

N_MOLECULES = len(EXPERIMENTS) * 2 * READS_PER_BARCODE


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_ml_project(tmp_path)


@pytest.fixture
def trained(project: Path):
    bound = bind_ml_job(ml_plan(), "train", project_dir=project)
    return train_and_publish(bound, project_dir=project, final_model=True)


def _first_fold(trained) -> str:
    return sorted(fold for fold in trained.model_ids["nb"] if fold != "final")[0]


def _apply_plan(model: str = "nb", *, labels: bool = True, exclude=None):
    document = ml_plan().to_dict()
    dataset = document["datasets"]["reads"]
    if not labels:
        dataset.pop("labels")
    if exclude is not None:
        dataset["positions"] = {"include": [[0, N_POSITIONS]], "exclude": [list(exclude)]}
    document["jobs"] = {"apply": {"action": "apply", "dataset": "reads", "model": model}}
    return parse_ml_plan(document)


def test_final_model_is_fit_on_every_row(trained) -> None:
    final_id = trained.model_ids["nb"]["final"]
    models = {m["fold"]: m for m in trained.read(runs_module.MODELS)}
    assert models["final"]["model_id"] == final_id
    assert models["final"]["held_out"] is None
    assert models["final"]["n_train"] == N_MOLECULES
    membership = trained.read(runs_module.MEMBERSHIP)
    final_rows = membership[membership["fold"] == "final"]
    assert len(final_rows) == N_MOLECULES and set(final_rows["role"]) == {"train"}
    splits = {split["fold"]: split for split in trained.read(runs_module.SPLITS)}
    assert splits["final"]["split_id"] == models["final"]["split_id"]
    # Held-out evaluation comes from the folds only.
    assert "final" not in set(trained.read(runs_module.PREDICTIONS)["fold"])
    load_published_sklearn_model(trained.workspace, final_id)


def test_applying_the_final_model_to_labeled_data(project: Path, trained) -> None:
    final_id = trained.model_ids["nb"]["final"]
    applied = apply_and_publish(
        _apply_plan(), "apply", model_id=final_id, project_dir=project, tags={"cohort": "all"}
    )

    manifest = json.loads((applied.path / "run_manifest.json").read_text())
    assert manifest["action"] == "apply" and manifest["state"] == "completed"
    assert manifest["source_model_ids"] == [final_id]
    predictions = applied.read(runs_module.APPLIED_PREDICTIONS)
    assert len(predictions) == N_MOLECULES and set(predictions["model_id"]) == {final_id}
    assert set(predictions["truth"]) == {"active", "inactive"}
    model = applied.read(runs_module.APPLIED_MODEL)
    assert model["originating_run_id"] == trained.run_id
    assert applied.summary["nb"]["average_precision"]["mean"] > 0.95
    assert len(applied.read(runs_module.APPLIED_MOLECULES)) == N_MOLECULES
    index = json.loads((applied.workspace.index_root / "runs.json").read_text())
    record = {r["run_id"]: r for r in index["records"]}[applied.run_id]
    assert record["tags"] == {"cohort": "all"} and record["source_model_ids"] == [final_id]


def test_applying_to_unlabeled_data_publishes_predictions_only(project: Path, trained) -> None:
    applied = apply_and_publish(
        _apply_plan(labels=False),
        "apply",
        model_id=trained.model_ids["nb"]["final"],
        project_dir=project,
    )
    assert not applied.labeled and applied.summary == {}
    predictions = applied.read(runs_module.APPLIED_PREDICTIONS)
    assert predictions["truth"].isna().all()
    assert np.allclose(predictions[["p_active", "p_inactive"]].sum(axis=1), 1.0)
    assert not (applied.path / runs_module.METRICS).exists()


def test_reapplied_fold_model_matches_its_held_out_predictions(project: Path, trained) -> None:
    fold = _first_fold(trained)
    applied = apply_and_publish(
        _apply_plan(), "apply", model_id=trained.model_ids["nb"][fold], project_dir=project
    )
    held_out = trained.read(runs_module.PREDICTIONS)
    held_out = held_out[held_out["fold"] == fold].set_index("molecule_uid")
    again = applied.read(runs_module.APPLIED_PREDICTIONS).set_index("molecule_uid")
    np.testing.assert_allclose(
        again.loc[held_out.index, "p_active"].to_numpy(), held_out["p_active"].to_numpy()
    )


def test_exact_model_in_the_job(project: Path, trained) -> None:
    final_id = trained.model_ids["nb"]["final"]
    applied = apply_and_publish(_apply_plan(f"model:{final_id}"), "apply", project_dir=project)
    assert applied.model_id == final_id
    with pytest.raises(MLJobServiceError, match="names model"):
        apply_and_publish(
            _apply_plan(f"model:{final_id}"),
            "apply",
            model_id=trained.model_ids["nb"][_first_fold(trained)],
            project_dir=project,
        )


def test_a_model_key_needs_a_model_id(project: Path, trained) -> None:
    with pytest.raises(MLJobServiceError, match="pass the published model_id"):
        apply_and_publish(_apply_plan(), "apply", project_dir=project)


def test_different_positions_are_refused(project: Path, trained) -> None:
    with pytest.raises(MLJobServiceError, match="positions differ.*20 missing"):
        apply_and_publish(
            _apply_plan(exclude=(10, 30)),
            "apply",
            model_id=trained.model_ids["nb"]["final"],
            project_dir=project,
        )


def test_final_torch_models_are_refused_before_training(project: Path) -> None:
    models = {"cnn": {"backend": "torch", "recipe": "residual_dilated_cnn_v1"}}
    bound = bind_ml_job(ml_plan(models), "train", project_dir=project)
    with pytest.raises(MLJobServiceError, match="validation role"):
        train_and_publish(bound, project_dir=project, final_model=True)
