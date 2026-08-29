"""Bundle/unbundle a run's published generations for transfer (`TAB-01`, `TAB-02`)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from smftools.data.analysis_bundle import (
    AnalysisBundleError,
    bundle_analysis_generations,
    unbundle_analysis_generations,
)
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    GenerationRecord,
)

pytestmark = pytest.mark.unit


def _publish(
    run_root: Path,
    stage_dir: str,
    generation_id: str,
    *,
    status: str = "complete",
    manifest_text: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Write a minimal generation tree matching the shared on-disk vocabulary."""
    container = run_root / stage_dir
    generation_dir = container / GENERATIONS_SUBDIR / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": status,
        "generation_id": generation_id,
        "config_hash": "abc123",
    }
    manifest_path = generation_dir / GENERATION_MANIFEST
    if manifest_text is not None:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    for name, content in (extra_files or {"spine.h5ad": "fake-spine-bytes"}).items():
        (generation_dir / name).write_text(content, encoding="utf-8")
    (container / CURRENT_FILENAME).write_text(
        json.dumps({"schema_version": 1, "generation_id": generation_id}), encoding="utf-8"
    )
    return generation_dir


def test_bundle_analysis_generations_bundles_a_complete_generation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-1")
    bundle_root = tmp_path / "bundles"

    results = bundle_analysis_generations(run_root, bundle_root=bundle_root)

    assert len(results) == 1
    assert results[0]["status"] == "bundled"
    assert results[0]["kind"] == "preprocess"
    bundle_path = bundle_root / "preprocess" / "gen-1.tar"
    assert bundle_path.is_file()
    assert (bundle_root / "preprocess" / "gen-1.tar.json").is_file()
    assert not list(bundle_path.parent.glob("*.partial"))


def test_bundle_is_self_contained_and_extracts_the_generation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "hmm_adata_outputs", "gen-hmm", extra_files={"store.txt": "payload"})
    bundle_root = tmp_path / "bundles"

    bundle_analysis_generations(run_root, bundle_root=bundle_root)

    bundle_path = bundle_root / "hmm" / "gen-hmm.tar"
    extract_dir = tmp_path / "extracted"
    with tarfile.open(bundle_path) as tar:
        tar.extractall(extract_dir, filter="data")

    manifest = json.loads((extract_dir / "gen-hmm" / GENERATION_MANIFEST).read_text())
    assert manifest["generation_id"] == "gen-hmm"
    assert (extract_dir / "gen-hmm" / "store.txt").read_text() == "payload"


def test_bundle_analysis_generations_is_idempotent_on_a_rerun(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-1")
    bundle_root = tmp_path / "bundles"
    first = bundle_analysis_generations(run_root, bundle_root=bundle_root)
    bundle_path = bundle_root / "preprocess" / "gen-1.tar"
    first_mtime = bundle_path.stat().st_mtime_ns

    second = bundle_analysis_generations(run_root, bundle_root=bundle_root)

    assert first[0]["status"] == "bundled"
    assert second[0]["status"] == "already_bundled"
    assert bundle_path.stat().st_mtime_ns == first_mtime


def test_bundle_analysis_generations_rebundles_a_corrupted_bundle(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-1")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(run_root, bundle_root=bundle_root)
    bundle_path = bundle_root / "preprocess" / "gen-1.tar"
    bundle_path.write_bytes(b"corrupted")

    results = bundle_analysis_generations(run_root, bundle_root=bundle_root)

    assert results[0]["status"] == "bundled"
    with tarfile.open(bundle_path) as tar:
        assert tar.getnames()  # a real tar again, not the corrupted bytes


def test_bundle_analysis_generations_skips_incomplete_generation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-1", status="in_progress")
    bundle_root = tmp_path / "bundles"

    results = bundle_analysis_generations(run_root, bundle_root=bundle_root)

    assert results[0]["status"] == "skipped"
    assert "'in_progress'" in results[0]["reason"]
    assert not (bundle_root / "preprocess").exists()


def test_bundle_analysis_generations_skips_unreadable_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-1", manifest_text="not json")
    bundle_root = tmp_path / "bundles"

    results = bundle_analysis_generations(run_root, bundle_root=bundle_root)

    assert results[0]["status"] == "skipped"
    assert "not readable" in results[0]["reason"]


def test_bundle_analysis_generations_filters_by_stage(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-pp")
    _publish(run_root, "hmm_adata_outputs", "gen-hmm")
    bundle_root = tmp_path / "bundles"

    results = bundle_analysis_generations(run_root, bundle_root=bundle_root, stage="hmm")

    assert len(results) == 1
    assert results[0]["kind"] == "hmm"


def test_bundle_analysis_generations_filters_by_generation_id(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _publish(run_root, "preprocess_adata_outputs", "gen-a")
    _publish(run_root, "preprocess_adata_outputs", "gen-b")
    bundle_root = tmp_path / "bundles"

    results = bundle_analysis_generations(run_root, bundle_root=bundle_root, generation_id="gen-b")

    assert len(results) == 1
    assert results[0]["generation_id"] == "gen-b"


def test_bundle_analysis_generations_returns_empty_list_when_nothing_published(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()

    assert bundle_analysis_generations(run_root, bundle_root=tmp_path / "bundles") == []


def test_bundle_one_refuses_when_directory_missing_despite_complete_manifest(
    tmp_path: Path,
) -> None:
    from smftools.data.analysis_bundle import _bundle_one

    record = GenerationRecord(
        scope="experiment",
        kind="preprocess",
        container="preprocess_adata_outputs",
        generation_id="ghost",
        path="preprocess_adata_outputs/generations/ghost",
        is_current=False,
        state="ok",
        status="complete",
        config_hash=None,
        manifest_schema_version=None,
        created_at=None,
        modified_at="",
        artifact_count=None,
        size_bytes=None,
        pinned=False,
        retention_reasons=(),
        issues=(),
    )

    with pytest.raises(AnalysisBundleError, match="no directory"):
        _bundle_one(tmp_path / "run", record, tmp_path / "bundles")


def test_unbundle_analysis_generations_extracts_into_a_fresh_run_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _publish(source_root, "preprocess_adata_outputs", "gen-1", extra_files={"note.txt": "hello"})
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)
    destination_root = tmp_path / "destination"

    results = unbundle_analysis_generations(bundle_root, run_root=destination_root)

    assert len(results) == 1
    assert results[0]["status"] == "unbundled"
    assert results[0]["kind"] == "preprocess"
    generation_dir = destination_root / "preprocess_adata_outputs" / GENERATIONS_SUBDIR / "gen-1"
    assert generation_dir.is_dir()
    assert (generation_dir / "note.txt").read_text() == "hello"
    manifest = json.loads((generation_dir / GENERATION_MANIFEST).read_text())
    assert manifest["generation_id"] == "gen-1"
    assert not (
        destination_root / "preprocess_adata_outputs" / GENERATIONS_SUBDIR / ".bundle-staging"
    ).exists()


def test_unbundle_analysis_generations_is_idempotent_on_a_rerun(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _publish(source_root, "preprocess_adata_outputs", "gen-1")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)
    destination_root = tmp_path / "destination"
    first = unbundle_analysis_generations(bundle_root, run_root=destination_root)

    second = unbundle_analysis_generations(bundle_root, run_root=destination_root)

    assert first[0]["status"] == "unbundled"
    assert second[0]["status"] == "already_unbundled"


def test_unbundle_analysis_generations_verifies_recorded_artifact_checksums(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    generation_dir = _publish(source_root, "raw_outputs", "gen-1", extra_files={})
    artifact = generation_dir / "spine.h5ad"
    artifact.write_text("fake-spine-bytes", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path = generation_dir / GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = {"spine": {"path": "spine.h5ad", "sha256": digest}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)
    destination_root = tmp_path / "destination"

    results = unbundle_analysis_generations(bundle_root, run_root=destination_root)

    assert results[0]["checksums_verified"] is True


def test_unbundle_analysis_generations_verifies_a_directory_artifact(tmp_path: Path) -> None:
    """Production `raw`/`preprocess` manifests checksum whole directories (the bulk
    partitioned `store/`), not just single files -- via `experiment_manifest.artifact_record`'s
    name+content digest across every file in the tree, not one file's bytes."""
    from smftools.informatics.experiment_manifest import artifact_record

    source_root = tmp_path / "source"
    generation_dir = _publish(source_root, "preprocess_adata_outputs", "gen-1", extra_files={})
    store_dir = generation_dir / "store"
    (store_dir / "reference=a" / "chunk=0").mkdir(parents=True)
    (store_dir / "reference=a" / "chunk=0" / "data.bin").write_bytes(b"payload-a")
    (store_dir / "reference=b").mkdir(parents=True)
    (store_dir / "reference=b" / "data.bin").write_bytes(b"payload-b")
    digest = artifact_record(store_dir, generation_dir, checksum=True)["sha256"]
    manifest_path = generation_dir / GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = {"store": {"path": "store", "kind": "directory", "sha256": digest}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)
    destination_root = tmp_path / "destination"

    results = unbundle_analysis_generations(bundle_root, run_root=destination_root)

    assert results[0]["status"] == "unbundled"
    assert results[0]["checksums_verified"] is True


def test_unbundle_analysis_generations_reports_no_checksums_for_stages_without_them(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _publish(source_root, "hmm_adata_outputs", "gen-1")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)
    destination_root = tmp_path / "destination"

    results = unbundle_analysis_generations(bundle_root, run_root=destination_root)

    assert results[0]["checksums_verified"] is False


def test_unbundle_analysis_generations_refuses_a_corrupted_bundle(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _publish(source_root, "preprocess_adata_outputs", "gen-1")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)
    (bundle_root / "preprocess" / "gen-1.tar").write_bytes(b"corrupted")

    with pytest.raises(AnalysisBundleError, match="does not match its recorded checksum"):
        unbundle_analysis_generations(bundle_root, run_root=tmp_path / "destination")


def test_unbundle_analysis_generations_refuses_a_content_checksum_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    generation_dir = _publish(source_root, "raw_outputs", "gen-1", extra_files={})
    artifact = generation_dir / "spine.h5ad"
    artifact.write_text("fake-spine-bytes", encoding="utf-8")
    manifest_path = generation_dir / GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = {"spine": {"path": "spine.h5ad", "sha256": "0" * 64}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)

    with pytest.raises(AnalysisBundleError, match="failed verification"):
        unbundle_analysis_generations(bundle_root, run_root=tmp_path / "destination")
    assert not (tmp_path / "destination" / "raw_outputs" / GENERATIONS_SUBDIR / "gen-1").exists()


def test_unbundle_analysis_generations_filters_by_stage(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _publish(source_root, "preprocess_adata_outputs", "gen-pp")
    _publish(source_root, "hmm_adata_outputs", "gen-hmm")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)

    results = unbundle_analysis_generations(
        bundle_root, run_root=tmp_path / "destination", stage="hmm"
    )

    assert len(results) == 1
    assert results[0]["kind"] == "hmm"


def test_unbundle_analysis_generations_filters_by_generation_id(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _publish(source_root, "preprocess_adata_outputs", "gen-a")
    _publish(source_root, "preprocess_adata_outputs", "gen-b")
    bundle_root = tmp_path / "bundles"
    bundle_analysis_generations(source_root, bundle_root=bundle_root)

    results = unbundle_analysis_generations(
        bundle_root, run_root=tmp_path / "destination", generation_id="gen-b"
    )

    assert len(results) == 1
    assert results[0]["generation_id"] == "gen-b"


def test_unbundle_analysis_generations_returns_empty_list_when_bundle_root_missing(
    tmp_path: Path,
) -> None:
    results = unbundle_analysis_generations(
        tmp_path / "no-such-bundles", run_root=tmp_path / "destination"
    )

    assert results == []
