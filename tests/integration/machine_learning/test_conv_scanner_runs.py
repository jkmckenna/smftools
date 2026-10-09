"""MLR-09: convolutional scanners train, publish, explain and expose their patterns."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tests.fixtures.ml_project import make_ml_project, ml_plan  # noqa: E402
from tests.integration.machine_learning.test_detector_catalogue import (  # noqa: E402
    MOTIF,
    N_POSITIONS,
    READS,
    _best_shift_correlation,
    _planted,
)

from smftools.machine_learning.models.torch_artifacts import (  # noqa: E402
    load_published_torch_model,
)
from smftools.machine_learning.orchestration import (  # noqa: E402
    TorchTrainOptions,
    bind_ml_job,
    detector_catalogue_run,
    explain_run,
    train_and_publish,
)
from smftools.machine_learning.orchestration import detectors as catalogue  # noqa: E402
from smftools.machine_learning.plan import parse_ml_plan  # noqa: E402
from smftools.machine_learning.training.torch_backend import TorchTrainingConfig  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    project = make_ml_project(
        tmp_path_factory.mktemp("scanner"),
        signal=_planted,
        n_positions=N_POSITIONS,
        reads_per_barcode=READS,
    )
    document = ml_plan(
        {
            "scanner": {
                "backend": "torch",
                "recipe": "motif_scanner_k21_v1",
                "overrides": {"filters": [4], "kernel_sizes": [9]},
            },
            "stacked": {
                "backend": "torch",
                "recipe": "two_layer_scanner_v1",
                "overrides": {"filters": [4, 4]},
            },
        }
    ).to_dict()
    document["splits"]["by_experiment"]["validation_fraction"] = 0.2
    bound = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    run = train_and_publish(
        bound,
        project_dir=project,
        torch_options=TorchTrainOptions(
            training_config=TorchTrainingConfig(
                max_epochs=40, batch_size=32, learning_rate=3e-3, patience=8, device="cpu"
            )
        ),
    )
    return project, run


def test_scanners_publish_with_their_detector_scale(trained) -> None:
    _project, run = trained
    models = run.read("models.json")
    assert {item["model_class"] for item in models} == {"spatial"}
    for item in models:
        scale = item["detector_scale"]
        expected = 9 if item["model"] == "scanner" else 50
        assert scale["receptive_field"] == expected
        assert 1 <= scale["effective_span_90"] <= 2 * N_POSITIONS


def test_a_filter_s_weights_are_the_planted_motif(trained) -> None:
    _project, run = trained
    matches = []
    for fold_model in run.model_ids["scanner"].values():
        model = load_published_torch_model(run.workspace, fold_model)
        weights = model.model.layers[0]["conv"].weight.detach().numpy()[:, 0, :]  # value channel
        head = model.model.head[-1].weight.detach().numpy()[0]
        # The filter the head leans on most for "active", read as a pattern.
        strongest = int(np.argmax(head))
        matches.append(_best_shift_correlation(weights[strongest], MOTIF))
    assert np.mean(matches) > 0.6, matches


def test_scanner_catalogue_and_explanation(trained) -> None:
    project, run = trained
    record = detector_catalogue_run(run.run_id, model="scanner", project_dir=project, figure=False)
    detectors = record.read(catalogue.DETECTORS)
    assert detectors.groupby("fold")["detector"].nunique().eq(4).all()
    assert detectors["auroc"].max() > 0.75
    stacked = detector_catalogue_run(run.run_id, model="stacked", project_dir=project, figure=False)
    windows = stacked.read(catalogue.WINDOWS)
    # Downsampled positions map back inside the molecule.
    assert windows["centre"].between(0, N_POSITIONS - 1).all()
    explained = explain_run(run.run_id, model="scanner", project_dir=project, figure=False)
    entry = explained.read("attributions/index.json")["folds"][0]
    _molecules, matrix = explained.attributions(entry["fold"])
    assert np.isfinite(matrix).all() and matrix.shape[-1] == N_POSITIONS
