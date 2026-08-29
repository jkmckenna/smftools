"""Bundle a run's published generations into few, large files for transfer (`TAB-01`)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from smftools.data.analysis_bundle import (
    AnalysisBundleError,
    bundle_analysis_generations,
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
