"""MLR-03b: a CNN's detector catalogue recovers a planted pattern."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from tests.fixtures.ml_project import make_ml_project, ml_plan  # noqa: E402

from smftools.machine_learning.orchestration import (  # noqa: E402
    MLJobServiceError,
    TorchTrainOptions,
    bind_ml_job,
    detector_catalogue_run,
    train_and_publish,
)
from smftools.machine_learning.orchestration import detectors as catalogue  # noqa: E402
from smftools.machine_learning.plan import parse_ml_plan  # noqa: E402
from smftools.machine_learning.training.torch_backend import TorchTrainingConfig  # noqa: E402

pytestmark = pytest.mark.integration

MOTIF = np.array([1, 1, 0, 0, 1, 1], dtype=np.float32)
N_POSITIONS = 80
READS = 80


def _planted(rng: np.random.Generator, active: bool) -> np.ndarray:
    """Background calls everywhere; active reads carry MOTIF at a random place."""
    calls = (rng.random((READS, N_POSITIONS)) < 0.3).astype(np.float32)
    if active:
        for row in range(READS):
            start = rng.integers(5, N_POSITIONS - MOTIF.size - 5)
            calls[row, start : start + MOTIF.size] = MOTIF
    calls[rng.random(calls.shape) < 0.03] = np.nan
    return calls


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    project = make_ml_project(
        tmp_path_factory.mktemp("planted"),
        signal=_planted,
        n_positions=N_POSITIONS,
        reads_per_barcode=READS,
    )
    document = ml_plan(
        {
            "cnn": {
                "backend": "torch",
                "recipe": "residual_dilated_cnn_v1",
                "overrides": {
                    "stem_channels": 8,
                    "block_channels": [8, 8],
                    "dilations": [1, 1],
                    "stem_kernel_size": 3,
                    "kernel_size": 3,
                    "use_se": False,
                    "mask_channels": True,
                    "span_masking": True,
                    "dropout": 0.0,
                },
            },
            "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
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


def _best_shift_correlation(pattern: np.ndarray, motif: np.ndarray) -> float:
    best = -1.0
    for start in range(pattern.size - motif.size + 1):
        segment = pattern[start : start + motif.size]
        if np.isfinite(segment).all() and segment.std() > 0:
            best = max(best, float(np.corrcoef(segment, motif)[0, 1]))
    return best


def test_a_detector_recovers_the_planted_pattern(trained) -> None:
    project, run = trained
    record = detector_catalogue_run(run.run_id, model="cnn", project_dir=project, top_windows=40)

    detectors = record.read(catalogue.DETECTORS)
    index = record.read(catalogue.PATTERN_INDEX)
    assert detectors.groupby("fold")["detector"].nunique().eq(8).all()
    found = []
    for entry in index["folds"]:
        patterns = np.load(record.path / entry["patterns"])
        rows = detectors[detectors["fold"] == entry["fold"]].set_index("detector")
        best = int(rows["auroc"].idxmax())
        found.append(
            {
                "auroc": rows.at[best, "auroc"],
                "match": _best_shift_correlation(patterns[best, 0], MOTIF),
                "spread": rows.at[best, "centre_sd"],
                "enrichment": rows.at[best, "log2_enrichment"],
            }
        )
    found = pd.DataFrame(found)
    # In every fold the most predictive detector reads the motif, wherever it sits,
    # and its top windows come from active molecules.
    assert (found["auroc"] > 0.75).all(), found
    assert (found["match"] > 0.7).all(), found
    assert (found["spread"] > 5).all(), found
    assert (found["enrichment"] > 0.5).all(), found
    assert (record.path / catalogue.FIGURE).is_file()


def test_windows_are_one_per_molecule_and_ranked(trained) -> None:
    project, run = trained
    record = detector_catalogue_run(
        run.run_id, model="cnn", project_dir=project, top_windows=10, figure=False
    )
    windows = record.read(catalogue.WINDOWS)
    for (_fold, _detector), rows in windows.groupby(["fold", "detector"]):
        assert rows["molecule_uid"].is_unique
        assert rows.sort_values("rank")["activation"].is_monotonic_decreasing
        assert len(rows) <= 10
    assert not (record.path / catalogue.FIGURE).exists()


def test_classical_models_have_no_detectors(trained) -> None:
    project, run = trained
    with pytest.raises(MLJobServiceError, match="not a torch model"):
        detector_catalogue_run(run.run_id, model="nb", project_dir=project)


def test_windows_helper_pads_outside_the_molecule() -> None:
    values = np.arange(10, dtype=np.float32).reshape(1, 10, 1)
    window = catalogue._windows(values, np.array([1]), half=3)
    assert window.shape == (1, 1, 7)
    assert np.isnan(window[0, 0, :2]).all()
    assert window[0, 0, 2:].tolist() == [0, 1, 2, 3, 4]
