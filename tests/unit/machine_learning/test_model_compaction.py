"""`SLC-02`: lossless compaction of stored model payloads."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from smftools.machine_learning.artifacts import validate_published_bundle
from smftools.machine_learning.artifacts.compaction import (
    ModelCompactionError,
    apply_compaction,
    compact_payload,
    compacted_models,
    plan_compaction,
)
from smftools.machine_learning.inference import apply_sklearn_partition_model
from smftools.machine_learning.models import (
    load_published_sklearn_model,
    publish_sklearn_model,
)
from smftools.machine_learning.training import fit_sklearn_partition_model

from .test_ml_sklearn_vertical import (
    DONE,
    _Dataset,
    _environment,
    _publish_completed_run,
    _resolved,
    _workspace,
)

pytestmark = pytest.mark.unit


def _published(tmp_path: Path, compression: str):
    dataset = _Dataset()
    trained = fit_sklearn_partition_model(dataset, _resolved(dataset, "random_forest"))
    workspace = _workspace(tmp_path)
    run = _publish_completed_run(tmp_path, workspace)
    published = publish_sklearn_model(
        trained.model,
        workspace,
        model_key="rf",
        originating_run_id=run.run_id,
        environment=_environment(),
        created_at=DONE,
        compression=compression,
    )
    before = apply_sklearn_partition_model(trained.model, dataset.data)
    return workspace, published, dataset, before


def _payload(workspace, model_id):
    return workspace.model_dir(model_id) / "payload" / "model.skops"


def test_plan_lists_only_uncompressed_payloads(tmp_path):
    workspace, published, *_ = _published(tmp_path, "none")
    plan = plan_compaction(workspace)
    assert [c.model_id for c in plan.candidates] == [published.manifest.model_id]
    assert plan.candidate_bytes == published.manifest.artifact.size_bytes

    (tmp_path / "other").mkdir()
    deflated, *_ = _published(tmp_path / "other", "deflate")
    assert plan_compaction(deflated).candidates == []


def test_compaction_is_lossless_and_validates(tmp_path):
    workspace, published, dataset, before = _published(tmp_path, "none")
    model_id = published.manifest.model_id
    manifest_before = (workspace.model_dir(model_id) / "model_manifest.json").read_bytes()
    original_size = _payload(workspace, model_id).stat().st_size

    (record,) = apply_compaction(workspace, plan_compaction(workspace))

    payload = _payload(workspace, model_id)
    assert payload.stat().st_size == record["stored_size"] < original_size
    with zipfile.ZipFile(payload) as archive:
        assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_DEFLATED}
    # identity and manifest untouched; the workspace records the new encoding
    assert (workspace.model_dir(model_id) / "model_manifest.json").read_bytes() == manifest_before
    assert compacted_models(workspace)[model_id]["original_size"] == original_size
    validate_published_bundle(
        workspace, workspace.model_dir(model_id), kind="model", expected_id=model_id
    )
    after = apply_sklearn_partition_model(
        load_published_sklearn_model(workspace, model_id), dataset.data
    )
    np.testing.assert_allclose(after.probabilities, before.probabilities)
    assert plan_compaction(workspace).candidates == []  # never compacted twice


def test_a_corrupted_compacted_payload_is_caught(tmp_path):
    workspace, published, *_ = _published(tmp_path, "none")
    model_id = published.manifest.model_id
    apply_compaction(workspace, plan_compaction(workspace))
    payload = _payload(workspace, model_id)
    data = bytearray(payload.read_bytes())
    data[len(data) // 2] ^= 0xFF
    payload.write_bytes(bytes(data))
    with pytest.raises(Exception, match="checksum|size"):
        validate_published_bundle(
            workspace, workspace.model_dir(model_id), kind="model", expected_id=model_id
        )


def test_crash_between_record_and_swap_leaves_a_valid_bundle(tmp_path, monkeypatch):
    """The record is written before the payload is replaced: if the process
    dies in between, the original payload still validates."""
    import smftools.machine_learning.artifacts.compaction as compaction

    workspace, published, *_ = _published(tmp_path, "none")
    model_id = published.manifest.model_id

    def crash(*args, **kwargs):
        raise RuntimeError("killed before the swap")

    monkeypatch.setattr(compaction, "_replace_payload", crash)
    (candidate,) = plan_compaction(workspace).candidates
    with pytest.raises(RuntimeError, match="killed"):
        compact_payload(workspace, candidate)
    assert model_id in compacted_models(workspace)
    validate_published_bundle(
        workspace, workspace.model_dir(model_id), kind="model", expected_id=model_id
    )
    # the interrupted model is offered again and completes
    monkeypatch.undo()
    (again,) = plan_compaction(workspace).candidates
    compact_payload(workspace, again)
    validate_published_bundle(
        workspace, workspace.model_dir(model_id), kind="model", expected_id=model_id
    )
    assert plan_compaction(workspace).candidates == []


def test_a_payload_not_matching_its_manifest_is_refused(tmp_path):
    workspace, published, *_ = _published(tmp_path, "none")
    (candidate,) = plan_compaction(workspace).candidates
    with zipfile.ZipFile(candidate.payload, "a") as archive:
        archive.writestr("extra.json", "{}")
    with pytest.raises(ModelCompactionError, match="does not match its manifest"):
        compact_payload(workspace, candidate)
    assert compacted_models(workspace) == {}


def test_parallel_compaction_commits_every_model(tmp_path):
    workspace, published, *_ = _published(tmp_path, "none")
    trained_ids = [published.manifest.model_id]
    dataset = _Dataset()
    for key in ("rf2", "rf3"):
        trained = fit_sklearn_partition_model(dataset, _resolved(dataset, "random_forest"))
        extra = publish_sklearn_model(
            trained.model,
            workspace,
            model_key=key,
            originating_run_id=published.manifest.originating_run_id,
            environment=_environment(),
            created_at=DONE,
            compression="none",
        )
        trained_ids.append(extra.manifest.model_id)
    plan = plan_compaction(workspace)
    assert len(plan.candidates) == len(set(trained_ids))
    records = apply_compaction(workspace, plan, workers=2)
    assert {r["model_id"] for r in records} == set(trained_ids)
    for model_id in set(trained_ids):
        validate_published_bundle(
            workspace, workspace.model_dir(model_id), kind="model", expected_id=model_id
        )


def test_cli_command_is_registered():
    """The command itself needs a project workspace; its plan / apply logic is
    covered above on an experiment workspace."""
    from click.testing import CliRunner

    from smftools.cli_entry import cli

    result = CliRunner().invoke(cli, ["project", "ml-storage", "--help"])
    assert result.exit_code == 0 and "compact" in result.output
