from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from tests.unit.pipeline.test_rebasecall_plan import _install_parent_fixtures, _request
from tests.unit.pipeline.test_rebasecall_selection import _skip_parent_validation
from tests.unit.pipeline.test_rebasecall_signal import _FakePod5IO

from smftools.pipeline import rebasecall_plan
from smftools.pipeline.rebasecall_basecall import (
    BASECALL_BAM_FILENAME,
    BASECALL_MANIFEST_FILENAME,
    BASECALL_ORIGIN_FILENAME,
    BasecallOutputInspection,
    RebasecallBasecallError,
    execute_rebasecall_basecall,
    prepare_rebasecall_basecall,
    read_published_rebasecall_basecall,
)
from smftools.pipeline.rebasecall_selection import freeze_rebasecall_selection

pytestmark = pytest.mark.unit

_MODEL = "chem_hac@v1.0.0"


class _FakeDorado:
    """A stand-in Dorado that emits the reads its ``--read-ids`` file requests."""

    def __init__(
        self,
        *,
        model=_MODEL,
        version="1.3.1+test",
        splits=(),
        drop=(),
        extra_reads=(),
        returncode=0,
    ):
        self.model = model
        self.version = version
        self.splits = dict(splits)
        self.drop = set(drop)
        self.extra_reads = tuple(extra_reads)
        self.returncode = returncode
        self.calls: list[tuple[str, ...]] = []

    def _argument(self, argv, flag):
        return argv[argv.index(flag) + 1]

    def runner(self, argv):
        argv = tuple(map(str, argv))
        self.calls.append(argv)
        if self.returncode != 0:
            return subprocess.CompletedProcess(argv, self.returncode, "", "fake dorado failed")
        requested = [
            line
            for line in Path(self._argument(argv, "--read-ids")).read_text().splitlines()
            if line
        ]
        records = []
        for read_id in requested:
            if read_id in self.drop:
                continue
            children = self.splits.get(read_id, 1)
            if children == 1:
                records.append({"read_id": read_id, "parent": None})
                continue
            for ordinal in range(children):
                records.append({"read_id": f"{read_id}-{ordinal}", "parent": read_id})
        records.extend({"read_id": read_id, "parent": None} for read_id in self.extra_reads)
        output_directory = Path(self._argument(argv, "--output-dir"))
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "calls_fake.bam").write_text(
            json.dumps(
                {
                    "records": records,
                    "version": self.version,
                    "model": self.model,
                }
            ),
            encoding="utf-8",
        )
        (output_directory / "sequencing_summary.txt").write_text(
            "read_id\n" + "\n".join(record["read_id"] for record in records) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "done", "")

    @staticmethod
    def inspector(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload["records"]
        return BasecallOutputInspection(
            record_count=len(records),
            read_ids=tuple(record["read_id"] for record in records),
            parent_ids=tuple(record["parent"] for record in records),
            programs=({"ID": "basecaller", "PN": "dorado", "VN": payload["version"]},),
            read_groups=({"ID": "rg-1", "DS": f"basecall_model={payload['model']} runid=abc"},),
        )


def _case(tmp_path, monkeypatch, *, mode="all-parent-molecules", signal=None):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    source_path = Path(cfg.source_manifest.rows[0].path)
    fake_io = _FakePod5IO({source_path: ("r1", "r2", "r3", "signal-4", "signal-5")})
    request = _request(mode, signal=signal)
    plan = rebasecall_plan.build_rebasecall_plan(cfg, request, pod5_indexer=fake_io.indexer)
    assert plan.status == "ready"
    frozen = freeze_rebasecall_selection(
        plan,
        tmp_path / "selection-results",
        accepted_plan_id=plan.plan_id,
        parent_validator=_skip_parent_validation,
    )
    return cfg, request, plan, frozen


def _execute(plan, frozen, root, dorado, **kwargs):
    return execute_rebasecall_basecall(
        plan,
        frozen,
        root,
        accepted_plan_id=plan.plan_id,
        runner=dorado.runner,
        bam_inspector=dorado.inspector,
        **kwargs,
    )


def test_publishes_validated_basecall_and_reuses_content_addressed_result(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    dorado = _FakeDorado()
    root = tmp_path / "basecalls"

    published = _execute(plan, frozen, root, dorado)
    reused = _execute(plan, frozen, root, dorado)

    assert published.directory == reused.directory
    assert published.directory.name == published.basecall_id
    assert published.manifest["counts"] == {
        "selection_record_count": 3,
        "requested_unique_read_count": 3,
        "source_parent_observed_count": 3,
        "output_record_count": 3,
        "split_child_record_count": 0,
        "missing_read_count": 0,
        "duplicate_output_read_id_count": 0,
    }
    # The second call reused the published tree instead of running Dorado again.
    assert len(dorado.calls) == 1
    assert not any(path.name.endswith(".tmp") for path in root.iterdir())
    assert (published.directory / BASECALL_BAM_FILENAME).is_file()
    assert (published.directory / BASECALL_ORIGIN_FILENAME).is_file()


def test_selection_mode_determines_the_stamped_generation_kind(tmp_path, monkeypatch):
    _, _, universe_plan, universe_frozen = _case(tmp_path, monkeypatch)
    universe = _execute(
        universe_plan,
        universe_frozen,
        tmp_path / "basecalls",
        _FakeDorado(),
    )

    assert universe.generation_kind == "parent_universe"
    assert universe.manifest["identity"]["generation_kind"] == "parent_universe"

    _, _, signal_plan, signal_frozen = _case(
        tmp_path / "all-signal",
        monkeypatch,
        mode="all-signal",
    )
    full_source = _execute(
        signal_plan,
        signal_frozen,
        tmp_path / "all-signal" / "basecalls",
        _FakeDorado(),
    )

    assert full_source.generation_kind == "full_source"


def test_same_alias_with_a_changed_model_bundle_cannot_reuse(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    root = tmp_path / "basecalls"
    assert plan._model_resolution is not None
    upgraded_model = replace(plan._model_resolution.simplex_model, name="chem_hac@v2.0.0")
    upgraded = replace(
        plan._model_resolution,
        simplex_model=upgraded_model,
        model_bundle_digest="d" * 64,
        simplex_path=plan._model_resolution.model_directory / upgraded_model.name,
    )
    upgraded_plan = replace(plan, _model_resolution=upgraded)

    first = _execute(plan, frozen, root, _FakeDorado())
    second = _execute(
        upgraded_plan,
        frozen,
        root,
        _FakeDorado(model="chem_hac@v2.0.0"),
    )

    # The floating request string is identical; only the resolved bundle moved.
    assert plan.request.basecall.model == upgraded_plan.request.basecall.model == "hac@latest"
    assert first.basecall_id != second.basecall_id
    assert (
        first.manifest["dorado"]["model_bundle_digest"]
        != (second.manifest["dorado"]["model_bundle_digest"])
    )


def test_split_children_are_counted_against_their_source_parent(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    dorado = _FakeDorado(splits={"r2": 3})

    published = _execute(plan, frozen, tmp_path / "basecalls", dorado)

    assert published.manifest["counts"]["output_record_count"] == 5
    assert published.manifest["counts"]["split_child_record_count"] == 2
    assert published.manifest["counts"]["source_parent_observed_count"] == 3
    origin = (published.directory / BASECALL_ORIGIN_FILENAME).read_text(encoding="utf-8")
    assert "r2-0,r2," in origin.replace(", ", ",")
    assert origin.count("source-a") == 5


def test_missing_reads_are_reported_without_blocking_publication(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)

    published = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado(drop=("r3",)))

    assert published.manifest["counts"]["missing_read_count"] == 1
    assert published.manifest["counts"]["source_parent_observed_count"] == 2


@pytest.mark.parametrize(
    ("dorado", "code"),
    [
        (_FakeDorado(returncode=2), "basecall_execution_failed"),
        (_FakeDorado(extra_reads=("foreign-read",)), "basecall_foreign_parent"),
        (_FakeDorado(version="9.9.9"), "basecall_header_mismatch"),
        (_FakeDorado(model="chem_sup@v1.0.0"), "basecall_model_mismatch"),
    ],
)
def test_failures_leave_no_reusable_commit(tmp_path, monkeypatch, dorado, code):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    root = tmp_path / "basecalls"

    with pytest.raises(RebasecallBasecallError) as error:
        _execute(plan, frozen, root, dorado)

    assert error.value.code == code
    assert not any(root.iterdir())


def test_duplicate_output_read_ids_are_rejected(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    dorado = _FakeDorado()
    dorado.extra_reads = ("r1",)

    with pytest.raises(RebasecallBasecallError) as error:
        _execute(plan, frozen, tmp_path / "basecalls", dorado)

    assert error.value.code == "basecall_duplicate_read_id"


def test_a_request_that_materializes_signal_requires_that_artifact(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch, signal={"materialize": True})
    dorado = _FakeDorado()

    with pytest.raises(RebasecallBasecallError) as error:
        _execute(plan, frozen, tmp_path / "basecalls", dorado)

    assert error.value.code == "basecall_signal_missing"
    assert dorado.calls == []


def test_blocked_or_mismatched_plans_never_execute(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    dorado = _FakeDorado()

    with pytest.raises(RebasecallBasecallError) as mismatch:
        execute_rebasecall_basecall(
            plan,
            frozen,
            tmp_path / "basecalls",
            accepted_plan_id="not-the-plan",
            runner=dorado.runner,
            bam_inspector=dorado.inspector,
        )
    blocked = replace(
        plan,
        blockers=(rebasecall_plan.RebasecallPlanReason("blocked", "blocked for test"),),
    )
    with pytest.raises(RebasecallBasecallError) as blocked_error:
        _execute(blocked, frozen, tmp_path / "basecalls", dorado)

    assert mismatch.value.code == "accepted_plan_mismatch"
    assert blocked_error.value.code == "accepted_plan_blocked"
    assert dorado.calls == []


def test_published_artifacts_are_revalidated_on_read(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    published = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())

    read_published_rebasecall_basecall(
        published.directory,
        expected_basecall_id=published.basecall_id,
    )
    (published.directory / BASECALL_BAM_FILENAME).write_text("tampered", encoding="utf-8")

    with pytest.raises(RebasecallBasecallError) as error:
        read_published_rebasecall_basecall(published.directory)

    assert error.value.code == "basecall_artifact_invalid"


def test_manifest_identity_must_match_its_directory(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    published = _execute(plan, frozen, tmp_path / "basecalls", _FakeDorado())
    manifest_path = published.directory / BASECALL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_kind"] = "full_source"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RebasecallBasecallError) as error:
        read_published_rebasecall_basecall(published.directory)

    assert error.value.code == "basecall_artifact_invalid"


def _write_dorado_shaped_bam(path, records, *, version="1.3.1+test", model=_MODEL):
    """Write a real BAM whose header carries the metadata Dorado emits."""
    pysam = pytest.importorskip("pysam")
    header = {
        "HD": {"VN": "1.6", "SO": "unknown"},
        "PG": [
            {
                "ID": "basecaller",
                "PN": "dorado",
                "VN": version,
                "CL": "dorado basecaller --emit-moves",
            }
        ],
        "RG": [
            {
                "ID": "rg-1",
                "DS": f"basecall_model={model} runid=abc",
                "PL": "ONT",
                "SM": "sample",
            }
        ],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for read_id, parent in records:
            segment = pysam.AlignedSegment(bam.header)
            segment.query_name = read_id
            segment.query_sequence = "ACGT"
            segment.query_qualities = pysam.qualitystring_to_array("IIII")
            segment.flag = 4
            tags = [("RG", "rg-1")]
            if parent is not None:
                tags.append(("pi", parent))
            segment.set_tags(tags)
            bam.write(segment)


def test_real_dorado_shaped_bam_validates_through_the_pysam_inspector(tmp_path, monkeypatch):
    """Exercise the default BAM inspector against real bytes, not a fake."""
    _, _, plan, frozen = _case(tmp_path, monkeypatch)
    records = (("r1", None), ("r2-0", "r2"), ("r2-1", "r2"), ("r3", None))

    def runner(argv):
        argv = tuple(map(str, argv))
        output_directory = Path(argv[argv.index("--output-dir") + 1])
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_dorado_shaped_bam(output_directory / "calls_fake.bam", records)
        return subprocess.CompletedProcess(argv, 0, "done", "")

    published = execute_rebasecall_basecall(
        plan,
        frozen,
        tmp_path / "basecalls",
        accepted_plan_id=plan.plan_id,
        runner=runner,
    )

    assert published.manifest["counts"]["output_record_count"] == 4
    assert published.manifest["counts"]["split_child_record_count"] == 1
    assert published.manifest["counts"]["source_parent_observed_count"] == 3
    assert published.manifest["dorado"]["header"]["observed_models"] == [_MODEL]
    read_published_rebasecall_basecall(
        published.directory,
        expected_basecall_id=published.basecall_id,
    )


def test_real_bam_from_a_foreign_model_is_rejected(tmp_path, monkeypatch):
    _, _, plan, frozen = _case(tmp_path, monkeypatch)

    def runner(argv):
        argv = tuple(map(str, argv))
        output_directory = Path(argv[argv.index("--output-dir") + 1])
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_dorado_shaped_bam(
            output_directory / "calls_fake.bam",
            (("r1", None),),
            model="chem_sup@v9.9.9",
        )
        return subprocess.CompletedProcess(argv, 0, "done", "")

    root = tmp_path / "basecalls"
    with pytest.raises(RebasecallBasecallError) as error:
        execute_rebasecall_basecall(
            plan,
            frozen,
            root,
            accepted_plan_id=plan.plan_id,
            runner=runner,
        )

    assert error.value.code == "basecall_model_mismatch"
    assert not any(root.iterdir())


def test_prepare_runs_the_whole_chain_from_a_request(tmp_path, monkeypatch):
    cfg = _install_parent_fixtures(tmp_path, monkeypatch)
    source_path = Path(cfg.source_manifest.rows[0].path)
    fake_io = _FakePod5IO({source_path: ("r1", "r2", "r3", "signal-4", "signal-5")})
    dorado = _FakeDorado()

    published = prepare_rebasecall_basecall(
        cfg,
        _request("all-parent-molecules"),
        tmp_path / "selection-results",
        tmp_path / "basecalls",
        accepted_plan_id=rebasecall_plan.build_rebasecall_plan(
            cfg,
            _request("all-parent-molecules"),
            pod5_indexer=fake_io.indexer,
        ).plan_id,
        pod5_indexer=fake_io.indexer,
        parent_validator=_skip_parent_validation,
        runner=dorado.runner,
        bam_inspector=dorado.inspector,
    )

    assert published.manifest["counts"]["output_record_count"] == 3
    assert published.generation_kind == "parent_universe"


def test_plan_reports_execution_as_available_rather_than_deferred(tmp_path, monkeypatch):
    _, _, plan, _ = _case(tmp_path, monkeypatch)

    payload = plan.to_dict()

    assert payload["execution_status"] == "basecall_only"
    assert payload["basecall_execution"] == {
        "status": "available_during_run_preparation",
        "accepted_plan_id": plan.plan_id,
        "generation_kind": "parent_universe",
    }
    assert (
        "dorado_basecall_execution_and_validation:srb-04b" not in (payload["deferred_capabilities"])
    )
