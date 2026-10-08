"""MLR-08: the detector-scale recipes train, publish, reload and explain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.ml_project import EXPERIMENTS, make_ml_project, ml_plan  # noqa: E402

from smftools.machine_learning.models.torch_artifacts import (  # noqa: E402
    load_published_torch_model,
)
from smftools.machine_learning.orchestration import (  # noqa: E402
    TorchTrainOptions,
    bind_ml_job,
    explain_run,
    train_and_publish,
)
from smftools.machine_learning.plan import parse_ml_plan  # noqa: E402
from smftools.machine_learning.training.torch_backend import TorchTrainingConfig  # noqa: E402

pytestmark = pytest.mark.integration

RECIPES = {
    "sub": ("rcnn_subnucleosome_v1", 113),
    "n2_3": ("rcnn_2_3_nucleosomes_v1", 513),
    "n4_6": ("rcnn_4_6_nucleosomes_v1", 1025),
    "full": ("rcnn_full_locus_v1", 5121),
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    project = make_ml_project(tmp_path_factory.mktemp("ladder"))
    document = ml_plan(
        {key: {"backend": "torch", "recipe": recipe} for key, (recipe, _rf) in RECIPES.items()}
    ).to_dict()
    document["splits"]["by_experiment"]["validation_fraction"] = 0.25
    bound = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    run = train_and_publish(
        bound,
        project_dir=project,
        torch_options=TorchTrainOptions(
            training_config=TorchTrainingConfig(max_epochs=2, batch_size=16, device="cpu")
        ),
    )
    return project, run


def test_each_fold_model_records_its_detector_scale(trained) -> None:
    _project, run = trained
    models = run.read("models.json")
    assert {item["model"] for item in models} == set(RECIPES)
    assert len(models) == len(RECIPES) * len(EXPERIMENTS)
    for item in models:
        scale = item["detector_scale"]
        assert scale["receptive_field"] == RECIPES[item["model"]][1]
        assert scale["measured_on"] == "test" and scale["n_molecules"] > 0
        assert 1 <= scale["effective_span_50"] <= scale["effective_span_90"]
        # The fixture's molecules are 40 positions long: no span can exceed that window.
        assert scale["effective_span_90"] <= 2 * 40 - 1


def test_recipes_reload_and_explain(trained) -> None:
    project, run = trained
    predictions = run.read("predictions/test.parquet")
    for key in ("sub", "full"):
        fold, model_id = sorted(run.model_ids[key].items())[0]
        model = load_published_torch_model(run.workspace, model_id)
        assert model.model.config.receptive_field == RECIPES[key][1]
        assert model.model.config.mask_channels and model.model.config.span_masking
        assert np.isfinite(predictions[predictions["model"] == key]["p_active"]).all()
    explained = explain_run(
        run.run_id,
        model="sub",
        method="IntegratedGradients",
        project_dir=project,
        background_size=8,
        figure=False,
    )
    entry = explained.read("attributions/index.json")["folds"][0]
    _molecules, matrix = explained.attributions(entry["fold"])
    assert np.isfinite(matrix).all() and matrix.shape[1:] == (1, 40)
