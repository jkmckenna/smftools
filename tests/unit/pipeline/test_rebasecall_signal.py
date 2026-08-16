from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from tests.unit.pipeline.test_rebasecall_plan import (
    _install_parent_fixtures,
    _request,
)
from tests.unit.pipeline.test_rebasecall_selection import _skip_parent_validation

from smftools.informatics.input_manifest import (
    InputManifestRow,
    ResolvedInputManifest,
    checksum_input_source,
    input_manifest_digest,
)
from smftools.informatics.pod5_identity import Pod5DatasetIndex, build_pod5_dataset_index
from smftools.pipeline import rebasecall_plan
from smftools.pipeline import rebasecall_signal as signal_module
from smftools.pipeline.rebasecall_selection import freeze_rebasecall_selection
from smftools.pipeline.rebasecall_signal import (
    RebasecallSignalError,
    materialize_rebasecall_signal,
    prepare_rebasecall_signal,
    read_materialized_rebasecall_signal,
)

pytestmark = pytest.mark.unit


class _FakePod5IO:
    def __init__(self, inventories):
        self.inventories = {
            Path(path).resolve(): set(map(str, read_ids)) for path, read_ids in inventories.items()
        }

    def _read_ids(self, path):
        resolved = Path(path).resolve()
        if resolved in self.inventories:
            return self.inventories[resolved]
        return set(json.loads(resolved.read_text(encoding="utf-8")))

    def indexer(self, sources):
        occurrences: dict[str, list[str]] = {}
        for source_id, path in sources:
            for read_id in self._read_ids(path):
                occurrences.setdefault(read_id, []).append(str(source_id))
        return Pod5DatasetIndex(
            {
                read_id: tuple(sorted(source_ids))
                for read_id, source_ids in sorted(occurrences.items())
            }
        )

    def writer(self, source_path, output_path, read_ids):
        found = sorted(set(read_ids).intersection(self._read_ids(source_path)))
        output_path.write_text(json.dumps(found), encoding="utf-8")


def _build_fake_case(tmp_path, monkeypatch, *, mode="all-parent-molecules", materialize=True):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    source_path = Path(cfg.source_manifest.rows[0].path)
    fake_io = _FakePod5IO({source_path: ("r1", "r2", "r3", "signal-4", "signal-5")})
    request = _request(
        mode,
        signal={"materialize": materialize},
    )
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        request,
        pod5_indexer=fake_io.indexer,
    )
    assert plan.status == "ready"
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    return cfg, request, plan, frozen, fake_io


def _materialize(plan, frozen, fake_io, root):
    return materialize_rebasecall_signal(
        plan,
        frozen,
        root,
        accepted_plan_id=plan.plan_id,
        pod5_writer=fake_io.writer,
        pod5_indexer=fake_io.indexer,
    )


def test_materializes_exact_selection_and_reuses_content_addressed_result(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    root = tmp_path / "signal-results"

    result = _materialize(plan, frozen, fake_io, root)
    reused = _materialize(plan, frozen, fake_io, root)

    assert reused.directory == result.directory
    assert result.directory.name == result.signal_id
    assert result.manifest["counts"] == {
        "selection_record_count": 3,
        "requested_unique_read_count": 3,
        "found_unique_read_count": 3,
        "missing_read_count": 0,
        "duplicate_output_read_id_count": 0,
        "duplicate_selection_reference_count": 0,
    }
    assert fake_io._read_ids(result.source_paths[0]) == {"r1", "r2", "r3"}
    assert not any(path.name.endswith(".tmp") for path in root.iterdir())


def test_all_signal_materializes_full_inventory(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(
        tmp_path,
        monkeypatch,
        mode="all-signal",
    )

    result = _materialize(plan, frozen, fake_io, tmp_path / "signal-results")

    assert result.manifest["counts"]["requested_unique_read_count"] == 5
    assert fake_io._read_ids(result.source_paths[0]) == {
        "r1",
        "r2",
        "r3",
        "signal-4",
        "signal-5",
    }


def test_materializes_each_source_to_a_deterministic_exact_output(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    source_paths = (tmp_path / "a.pod5", tmp_path / "b.pod5")
    source_paths[0].write_bytes(b"source-a")
    source_paths[1].write_bytes(b"source-b")
    source_rows = tuple(
        InputManifestRow(
            source_id=source_id,
            path=str(path),
            sha256=checksum_input_source(path)[0],
            size_bytes=path.stat().st_size,
            source_kind="pod5",
            source_role="raw_signal",
        )
        for source_id, path in zip(("source-a", "source-b"), source_paths, strict=True)
    )
    manifest = ResolvedInputManifest(
        rows=source_rows,
        digest=input_manifest_digest(source_rows),
        resolution_method="published",
        base_directory=str(tmp_path),
    )
    monkeypatch.setattr(rebasecall_plan, "_read_input_manifest", lambda _parent: manifest)
    pd.DataFrame(
        {
            "read_id": ["r1", "r2", "r3"],
            "molecule_uid": ["m1", "m2", "m3"],
        }
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)
    fake_io = _FakePod5IO(
        {
            source_paths[0]: ("r1", "r2"),
            source_paths[1]: ("r3", "signal-4"),
        }
    )
    request = _request("all-parent-molecules", signal={"materialize": True})
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        request,
        pod5_indexer=fake_io.indexer,
    )
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )

    result = _materialize(plan, frozen, fake_io, tmp_path / "signal-results")

    assert [record["source_id"] for record in result.manifest["outputs"]] == [
        "source-a",
        "source-b",
    ]
    assert [path.name for path in result.source_paths] == ["000000.pod5", "000001.pod5"]
    assert fake_io._read_ids(result.source_paths[0]) == {"r1", "r2"}
    assert fake_io._read_ids(result.source_paths[1]) == {"r3"}


def test_split_children_materialize_one_signal_and_record_duplicate_reference(
    tmp_path, monkeypatch
):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    source_path = Path(cfg.source_manifest.rows[0].path)
    fake_io = _FakePod5IO({source_path: ("parent",)})
    pd.DataFrame(
        {
            "read_id": ["child-1", "child-2"],
            "molecule_uid": ["m1", "m2"],
            "pod5_read_id": ["parent", "parent"],
        }
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)
    request = _request(
        "ids",
        id_kind="pod5_read_id",
        ids=["parent"],
        signal={"materialize": True},
    )
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        request,
        pod5_indexer=fake_io.indexer,
    )
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )

    result = _materialize(plan, frozen, fake_io, tmp_path / "signal-results")

    assert result.manifest["counts"]["selection_record_count"] == 2
    assert result.manifest["counts"]["requested_unique_read_count"] == 1
    assert result.manifest["counts"]["duplicate_selection_reference_count"] == 1


def test_rejects_stale_acceptance_unrequested_materialization_and_wrong_selection(
    tmp_path, monkeypatch
):
    cfg, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)

    with pytest.raises(RebasecallSignalError, match="does not match") as stale:
        materialize_rebasecall_signal(
            plan,
            frozen,
            tmp_path / "stale",
            accepted_plan_id="stale-plan",
            pod5_writer=fake_io.writer,
            pod5_indexer=fake_io.indexer,
        )
    assert stale.value.code == "accepted_plan_mismatch"

    no_materialize = _request("all-parent-molecules", signal={"materialize": False})
    no_materialize_plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        no_materialize,
        pod5_indexer=fake_io.indexer,
    )
    with pytest.raises(RebasecallSignalError, match="did not enable") as not_requested:
        materialize_rebasecall_signal(
            no_materialize_plan,
            frozen,
            tmp_path / "not-requested",
            accepted_plan_id=no_materialize_plan.plan_id,
            pod5_writer=fake_io.writer,
            pod5_indexer=fake_io.indexer,
        )
    assert not_requested.value.code == "signal_materialization_not_requested"

    other_request = _request(
        "ids",
        id_kind="read_id",
        ids=["r1"],
        signal={"materialize": True},
    )
    other_plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        other_request,
        pod5_indexer=fake_io.indexer,
    )
    with pytest.raises(RebasecallSignalError, match="does not belong") as mismatch:
        materialize_rebasecall_signal(
            other_plan,
            frozen,
            tmp_path / "wrong-selection",
            accepted_plan_id=other_plan.plan_id,
            pod5_writer=fake_io.writer,
            pod5_indexer=fake_io.indexer,
        )
    assert mismatch.value.code == "signal_selection_mismatch"


def test_source_drift_blocks_before_materialization(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    source_path = Path(plan._source_resolution.rows[0].resolved_path)
    source_path.write_bytes(b"changed")
    root = tmp_path / "signal-results"

    with pytest.raises(RebasecallSignalError, match="changed") as error:
        _materialize(plan, frozen, fake_io, root)

    assert error.value.code == "signal_source_changed"
    assert not root.exists()


def test_missing_writer_output_leaves_no_partial_artifact(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    root = tmp_path / "signal-results"

    def omit_one(source_path, output_path, read_ids):
        fake_io.writer(source_path, output_path, read_ids[:-1])

    with pytest.raises(RebasecallSignalError, match="exact requested UUID") as error:
        materialize_rebasecall_signal(
            plan,
            frozen,
            root,
            accepted_plan_id=plan.plan_id,
            pod5_writer=omit_one,
            pod5_indexer=fake_io.indexer,
        )

    assert error.value.code == "signal_uuid_count_mismatch"
    assert root.is_dir()
    assert not list(root.iterdir())


def test_duplicate_writer_output_is_rejected_before_publication(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    root = tmp_path / "signal-results"

    def duplicate_output_indexer(sources):
        if Path(sources[0][1]).resolve() in fake_io.inventories:
            return fake_io.indexer(sources)
        return Pod5DatasetIndex(
            {
                "r1": ("source-a", "source-a"),
                "r2": ("source-a",),
                "r3": ("source-a",),
            }
        )

    with pytest.raises(RebasecallSignalError, match="duplicate POD5 UUID") as error:
        materialize_rebasecall_signal(
            plan,
            frozen,
            root,
            accepted_plan_id=plan.plan_id,
            pod5_writer=fake_io.writer,
            pod5_indexer=duplicate_output_indexer,
        )

    assert error.value.code == "signal_uuid_duplicate"
    assert root.is_dir()
    assert not list(root.iterdir())


def test_manifest_failure_is_atomic(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    root = tmp_path / "signal-results"

    def fail_manifest(*_args, **_kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(signal_module, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        _materialize(plan, frozen, fake_io, root)

    assert root.is_dir()
    assert not list(root.iterdir())


def test_tampered_output_fails_validation(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    result = _materialize(plan, frozen, fake_io, tmp_path / "signal-results")
    result.source_paths[0].write_text(json.dumps(["changed"]), encoding="utf-8")

    with pytest.raises(RebasecallSignalError, match="checksum") as error:
        read_materialized_rebasecall_signal(
            result.directory,
            pod5_indexer=fake_io.indexer,
        )

    assert error.value.code == "signal_artifact_invalid"


def test_malformed_manifest_count_fails_with_stable_artifact_error(tmp_path, monkeypatch):
    _, _, plan, frozen, fake_io = _build_fake_case(tmp_path, monkeypatch)
    result = _materialize(plan, frozen, fake_io, tmp_path / "signal-results")
    manifest = dict(result.manifest)
    manifest["identity"] = dict(manifest["identity"])
    manifest["identity"]["sources"] = [dict(manifest["identity"]["sources"][0])]
    manifest["identity"]["sources"][0]["requested_read_count"] = "3"
    identity_bytes = json.dumps(manifest["identity"], sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest["signal_id"] = hashlib.sha256(identity_bytes).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RebasecallSignalError, match="requested read count") as error:
        read_materialized_rebasecall_signal(
            result.directory,
            pod5_indexer=fake_io.indexer,
        )

    assert error.value.code == "signal_artifact_invalid"


def test_prepare_rebuilds_plan_freezes_selection_and_materializes(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    source_path = Path(cfg.source_manifest.rows[0].path)
    fake_io = _FakePod5IO({source_path: ("r1", "r2", "r3")})
    request = _request("all-parent-molecules", signal={"materialize": True})
    plan = rebasecall_plan.build_rebasecall_plan(
        cfg,
        request,
        pod5_indexer=fake_io.indexer,
    )

    result = prepare_rebasecall_signal(
        cfg,
        request,
        tmp_path / "selection-results",
        tmp_path / "signal-results",
        accepted_plan_id=plan.plan_id,
        pod5_writer=fake_io.writer,
        pod5_indexer=fake_io.indexer,
        parent_validator=_skip_parent_validation,
    )

    assert result.manifest["accepted_plan_id"] == plan.plan_id
    assert result.manifest["counts"]["found_unique_read_count"] == 3


def test_real_filtered_pod5_remains_valid_after_original_is_removed(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    fixture = Path(__file__).parents[2] / "_test_inputs" / "_test_pod5_I.pod5"
    original = tmp_path / "source" / "reads.pod5"
    original.parent.mkdir()
    shutil.copyfile(fixture, original)
    sha256, size_bytes = checksum_input_source(original)
    source_row = InputManifestRow(
        source_id="source-a",
        path=str(original),
        sha256=sha256,
        size_bytes=size_bytes,
        source_kind="pod5",
        source_role="raw_signal",
    )
    manifest = ResolvedInputManifest(
        rows=(source_row,),
        digest=input_manifest_digest((source_row,)),
        resolution_method="published",
        base_directory=str(original.parent),
    )
    monkeypatch.setattr(rebasecall_plan, "_read_input_manifest", lambda _parent: manifest)
    index = build_pod5_dataset_index(((source_row.source_id, original),))
    selected_ids = tuple(sorted(index.sources_by_read_id)[:2])
    pd.DataFrame(
        {
            "read_id": list(index.sources_by_read_id),
            "molecule_uid": [f"m{index}" for index in range(index.unique_read_count)],
        }
    ).to_parquet(cfg.raw_parent.generation_dir / "obs.parquet", index=False)
    request = _request(
        "ids",
        id_kind="read_id",
        ids=list(selected_ids),
        signal={"materialize": True},
    )
    plan = rebasecall_plan.build_rebasecall_plan(cfg, request)
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )

    result = materialize_rebasecall_signal(
        plan,
        frozen,
        tmp_path / "signal-results",
        accepted_plan_id=plan.plan_id,
    )
    original.unlink()
    replayed = read_materialized_rebasecall_signal(result.directory)
    output_index = build_pod5_dataset_index(((source_row.source_id, replayed.source_paths[0]),))

    assert replayed.signal_id == result.signal_id
    assert set(output_index.sources_by_read_id) == set(selected_ids)
    assert result.manifest["counts"]["missing_read_count"] == 0
