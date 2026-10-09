"""MLR-06: one store read per molecule for every fold and model of a job."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.fixtures.ml_project import EXPERIMENTS, READS_PER_BARCODE, make_ml_project, ml_plan

import smftools.machine_learning.data.partition_dataset as reader
from smftools.machine_learning.data.partition_dataset import (
    MLPartitionDataError,
    PartitionDataset,
    PartitionReadPolicy,
)
from smftools.machine_learning.orchestration import bind_ml_job, run_bound_train_job
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.integration

MODELS = {
    "nb": {"backend": "sklearn", "family": "bernoulli_nb"},
    "rf": {"backend": "sklearn", "family": "random_forest", "parameters": {"n_estimators": 10}},
}
N_MOLECULES = len(EXPERIMENTS) * 2 * READS_PER_BARCODE


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    return make_ml_project(tmp_path_factory.mktemp("cache"))


@pytest.fixture
def decoded(monkeypatch):
    """Rows decoded from the stores, counted."""
    counts = {"rows": 0, "calls": 0}
    original = PartitionDataset._decode_rows

    def counting(self, entries):
        counts["rows"] += len(entries)
        counts["calls"] += 1
        return original(self, entries)

    monkeypatch.setattr(PartitionDataset, "_decode_rows", counting)
    return counts


def _train(project, policy):
    bound = bind_ml_job(ml_plan(MODELS), "train", project_dir=project, policy=policy)
    return bound, run_bound_train_job(bound)


def test_every_molecule_is_read_once_for_all_folds_and_models(project, decoded) -> None:
    bound, _runs = _train(project, PartitionReadPolicy(batch_size=8))
    # 3 folds x 2 models x (train + test) read every molecule several times;
    # the shared cache decodes each one once.
    assert decoded["rows"] == N_MOLECULES
    cache = bound.folds[0].dataset.row_cache
    assert all(fold.dataset.row_cache is cache for fold in bound.folds)
    assert len(cache) == N_MOLECULES and cache.hits > 0


def test_without_the_cache_rows_are_read_again(project, decoded) -> None:
    bound, _runs = _train(project, PartitionReadPolicy(batch_size=8, row_cache_bytes=0))
    assert bound.folds[0].dataset.row_cache is None
    assert decoded["rows"] > 3 * N_MOLECULES


def test_results_equal_an_uncached_run(project) -> None:
    _bound, cached = _train(project, PartitionReadPolicy(batch_size=8))
    _bound, plain = _train(project, PartitionReadPolicy(batch_size=8, row_cache_bytes=0))
    for left, right in zip(cached, plain, strict=True):
        assert (left.model_name, left.fold_name) == (right.model_name, right.fold_name)
        assert left.predictions.molecule_uids == right.predictions.molecule_uids
        np.testing.assert_array_equal(
            np.asarray(left.predictions.probabilities), np.asarray(right.predictions.probabilities)
        )


def test_a_full_cache_falls_back_to_reading(project, decoded) -> None:
    probe = bind_ml_job(ml_plan(MODELS), "train", project_dir=project).folds[0].dataset.plan
    room = 10  # molecules' worth of bytes
    bound, capped = _train(
        project,
        PartitionReadPolicy(batch_size=8, row_cache_bytes=probe.bytes_per_row * room),
    )
    cache = bound.folds[0].dataset.row_cache
    # Partly filled, never over its budget (rows are smaller than the reader's
    # conservative per-row estimate, so more than ``room`` fit).
    assert 0 < len(cache) < N_MOLECULES and cache.bytes <= cache.max_bytes
    assert decoded["rows"] > N_MOLECULES
    _bound, plain = _train(project, PartitionReadPolicy(batch_size=8, row_cache_bytes=0))
    for left, right in zip(capped, plain, strict=True):
        np.testing.assert_array_equal(
            np.asarray(left.predictions.probabilities), np.asarray(right.predictions.probabilities)
        )


def test_a_cache_serves_one_snapshot_and_positions_only(project) -> None:
    bound = bind_ml_job(ml_plan(MODELS), "train", project_dir=project)
    cache = bound.folds[0].dataset.row_cache
    document = ml_plan(MODELS).to_dict()
    document["datasets"]["reads"]["positions"] = {"include": [[0, 10]]}
    other = bind_ml_job(parse_ml_plan(document), "train", project_dir=project)
    with pytest.raises(MLPartitionDataError, match="shared only by datasets of one snapshot"):
        PartitionDataset(other.folds[0].dataset.plan, row_cache=cache)


def test_the_policy_validates_its_budget() -> None:
    with pytest.raises(MLPartitionDataError, match="row_cache_bytes"):
        PartitionReadPolicy(row_cache_bytes=-1)
    default = PartitionReadPolicy()
    assert default.effective_row_cache_bytes == default.max_materialization_bytes
    assert reader.PartitionRowCache(max_bytes=0).max_bytes == 0
