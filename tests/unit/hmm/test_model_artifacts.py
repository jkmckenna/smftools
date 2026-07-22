from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smftools.cli.hmm_adata import HMMTrainer
from smftools.hmm.model_artifacts import (
    HMMArtifactConflictError,
    HMMArtifactError,
    HMMModelKey,
    checkpoint_path,
    hmm_fit_config_hash,
    load_artifact_record,
    model_fit_lock,
    publish_checkpoint,
)


def _key(**overrides) -> HMMModelKey:
    values = {
        "fit_kind": "PER",
        "reference": "ref/top",
        "barcode": "barcode:01",
        "label": "GpC accessibility",
        "architecture": "single",
        "fit_config_hash": "0123456789abcdef",
        "core_start": 0,
        "core_end": 100,
    }
    values.update(overrides)
    return HMMModelKey(**values)


def _cfg(**overrides):
    values = {
        "hmm_fit_scope": "per_sample",
        "hmm_n_states": 2,
        "hmm_dtype": "float64",
        "hmm_max_iter": 1,
        "hmm_tol": 0.0,
        "force_redo_hmm_fit": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_model_ids_do_not_alias_legacy_filename_collisions(tmp_path):
    slash = _key(reference="ref/a")
    underscore = _key(reference="ref_a")

    assert slash.model_id != underscore.model_id
    for key in (slash, underscore):
        path = checkpoint_path(tmp_path, key)
        assert path.parent.parent == tmp_path
        assert path.name == f"{key.model_id}.pt"
        assert set(path.parent.name) <= set("0123456789abcdef")


def test_fit_config_hash_includes_effective_defaults_and_fit_changes():
    implicit = hmm_fit_config_hash(SimpleNamespace())
    explicit_default = hmm_fit_config_hash(SimpleNamespace(hmm_max_iter=50, hmm_tol=1e-5))
    changed = hmm_fit_config_hash(SimpleNamespace(hmm_max_iter=51, hmm_tol=1e-5))

    assert implicit == explicit_default
    assert changed != implicit


def test_checkpoint_publication_rejects_different_content(tmp_path):
    torch = pytest.importorskip("torch")
    key = _key()
    path = checkpoint_path(tmp_path, key)
    first = {"state_dict": {"emission": torch.tensor([0.2, 0.8])}}
    second = {"state_dict": {"emission": torch.tensor([0.3, 0.7])}}

    with model_fit_lock(path):
        record = publish_checkpoint(first, path, key)
    with model_fit_lock(path), pytest.raises(HMMArtifactConflictError):
        publish_checkpoint(second, path, key)

    assert load_artifact_record(path)["checkpoint_sha256"] == record["checkpoint_sha256"]
    payload = torch.load(path, map_location="cpu")
    assert payload["artifact_schema_version"] == 1
    assert payload["model_id"] == key.model_id
    assert payload["fit_config_hash"] == key.fit_config_hash


def test_checkpoint_checksum_detects_tampering(tmp_path):
    torch = pytest.importorskip("torch")
    key = _key()
    path = checkpoint_path(tmp_path, key)
    with model_fit_lock(path):
        publish_checkpoint({"state_dict": {"x": torch.tensor([1.0])}}, path, key)

    with path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(HMMArtifactError, match="checksum mismatch"):
        load_artifact_record(path)


def test_concurrent_trainers_reuse_one_immutable_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    values = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    def fit_once(_index: int) -> dict:
        trainer = HMMTrainer(cfg=_cfg(), models_dir=models_dir)
        trainer.fit_or_load(
            sample="bc1",
            ref="ref/top",
            label="GpC",
            arch="single",
            X=values,
            coords=None,
            device="cpu",
            reference="ref/top",
            core_start=0,
            core_end=3,
            training_selection={"selection_sha256": "selection", "n_reads": 2},
        )
        return dict(trainer.last_artifact)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fit_once, range(2)))

    assert records[0]["model_id"] == records[1]["model_id"]
    assert records[0]["checkpoint_sha256"] == records[1]["checkpoint_sha256"]
    assert len(list(models_dir.rglob("*.pt"))) == 1
    assert len(list(models_dir.rglob("*.json"))) == 1


def test_global_then_adapt_publishes_distinct_base_and_adapted_artifacts(tmp_path):
    trainer = HMMTrainer(
        cfg=_cfg(hmm_fit_scope="global_then_adapt", hmm_adapt_iters=1),
        models_dir=tmp_path / "models",
    )
    values = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    trainer.fit_or_load(
        sample="bc1",
        ref="ref",
        label="GpC",
        arch="single",
        X=values,
        coords=None,
        device="cpu",
        core_start=0,
        core_end=3,
    )

    assert trainer.last_artifact["model_key"]["fit_kind"] == "ADAPT"
    assert len(list(trainer.models_dir.rglob("*.pt"))) == 2


def test_forced_fit_revision_changes_model_identity(tmp_path):
    first = HMMTrainer(
        cfg=_cfg(force_redo_hmm_fit=True, _hmm_fit_revision="run-one"),
        models_dir=Path(tmp_path),
    )
    second = HMMTrainer(
        cfg=_cfg(force_redo_hmm_fit=True, _hmm_fit_revision="run-two"),
        models_dir=Path(tmp_path),
    )

    first_key = first._key(
        kind="PER",
        sample="bc1",
        reference="ref",
        label="GpC",
        arch="single",
        core_start=0,
        core_end=10,
    )
    second_key = second._key(
        kind="PER",
        sample="bc1",
        reference="ref",
        label="GpC",
        arch="single",
        core_start=0,
        core_end=10,
    )

    assert first_key.model_id != second_key.model_id
