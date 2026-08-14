from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.constants import LATENT_DIR, PREPROCESS_DIR, RAW_DIR, SPATIAL_DIR
from smftools.informatics.generation_listing import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    STATE_MISSING,
    STATE_OK,
    STATE_UNREADABLE,
    list_experiment_generations,
    list_project_generations,
)

pytestmark = pytest.mark.unit


def _publish(
    container: Path,
    generation_id: str,
    *,
    current: bool = False,
    manifest: dict | None = None,
    manifest_text: str | None = None,
) -> Path:
    """Write a minimal generation tree matching the shared on-disk vocabulary."""
    generation_dir = container / GENERATIONS_SUBDIR / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    payload = (
        manifest
        if manifest is not None
        else {
            "schema_version": 2,
            "status": "complete",
            "generation_id": generation_id,
            "config_hash": "abc123",
            "artifacts": {"spine": "spine.h5ad", "obs": "obs.parquet"},
        }
    )
    target = generation_dir / GENERATION_MANIFEST
    if manifest_text is not None:
        target.write_text(manifest_text, encoding="utf-8")
    else:
        target.write_text(json.dumps(payload), encoding="utf-8")
    if current:
        (container / CURRENT_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "generation_path": f"{GENERATIONS_SUBDIR}/{generation_id}",
                }
            ),
            encoding="utf-8",
        )
    return generation_dir


def test_lists_generations_across_stages_and_marks_current(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-old")
    _publish(run_root / RAW_DIR, "raw-new", current=True)
    _publish(run_root / PREPROCESS_DIR, "pp-1", current=True)
    _publish(run_root / LATENT_DIR, "lat-1", current=True)

    records = list_experiment_generations(run_root)

    assert [(r.kind, r.generation_id) for r in records] == [
        ("raw", "raw-new"),
        ("raw", "raw-old"),
        ("preprocess", "pp-1"),
        ("latent", "lat-1"),
    ]
    assert all(record.state == STATE_OK for record in records)
    current = {r.generation_id for r in records if r.is_current}
    assert current == {"raw-new", "pp-1", "lat-1"}
    raw_new = next(r for r in records if r.generation_id == "raw-new")
    assert raw_new.config_hash == "abc123"
    assert raw_new.artifact_count == 2
    assert raw_new.manifest_schema_version == 2
    assert raw_new.container == RAW_DIR
    assert raw_new.path == f"{RAW_DIR}/{GENERATIONS_SUBDIR}/raw-new"
    assert raw_new.size_bytes is None


def test_stage_without_generations_contributes_nothing(tmp_path: Path) -> None:
    """A legacy in-place stage is absence of data, not an error."""
    run_root = tmp_path / "outputs"
    (run_root / SPATIAL_DIR).mkdir(parents=True)
    (run_root / SPATIAL_DIR / "spine.h5ad").write_bytes(b"")
    _publish(run_root / RAW_DIR, "raw-1", current=True)

    records = list_experiment_generations(run_root)

    assert [r.kind for r in records] == ["raw"]


def test_experiment_predating_generations_lists_empty(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    (run_root / RAW_DIR).mkdir(parents=True)
    (run_root / PREPROCESS_DIR).mkdir(parents=True)

    assert list_experiment_generations(run_root) == []


def test_unreadable_manifest_is_reported_not_raised(tmp_path: Path) -> None:
    """The truncated-manifest case: listing is the tool you use when it is broken."""
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-good", current=True)
    _publish(run_root / RAW_DIR, "raw-torn", manifest_text='{"schema_version": 2, "artifa')

    records = list_experiment_generations(run_root)
    torn = next(r for r in records if r.generation_id == "raw-torn")

    assert torn.state == STATE_UNREADABLE
    assert torn.config_hash is None
    assert torn.issues and "unreadable" in torn.issues[0]
    # The healthy sibling is still reported.
    assert next(r for r in records if r.generation_id == "raw-good").state == STATE_OK


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    (run_root / RAW_DIR / GENERATIONS_SUBDIR / "raw-empty").mkdir(parents=True)

    records = list_experiment_generations(run_root)

    assert len(records) == 1
    assert records[0].state == STATE_UNREADABLE
    assert "missing" in records[0].issues[0]


def test_dangling_current_pointer_is_surfaced(tmp_path: Path) -> None:
    """Readers fail hard on this; an inventory that hid it would be worse than useless."""
    run_root = tmp_path / "outputs"
    container = run_root / RAW_DIR
    _publish(container, "raw-1", current=True)
    (container / GENERATIONS_SUBDIR / "raw-1" / GENERATION_MANIFEST).unlink()
    (container / GENERATIONS_SUBDIR / "raw-1").rmdir()

    records = list_experiment_generations(run_root)

    assert len(records) == 1
    assert records[0].state == STATE_MISSING
    assert records[0].is_current is True
    assert records[0].generation_id == "raw-1"
    assert "does not exist" in records[0].issues[0]


def test_generation_id_mismatch_is_flagged_but_listed(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(
        run_root / RAW_DIR,
        "dir-name",
        manifest={"schema_version": 2, "status": "complete", "generation_id": "other-name"},
    )

    records = list_experiment_generations(run_root)

    assert records[0].state == STATE_OK
    assert any("does not match directory name" in issue for issue in records[0].issues)


def test_staging_directory_is_ignored(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-1", current=True)
    (run_root / RAW_DIR / GENERATIONS_SUBDIR / ".staging").mkdir()

    records = list_experiment_generations(run_root)

    assert [r.generation_id for r in records] == ["raw-1"]


def test_include_size_totals_generation_bytes(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    generation = _publish(run_root / RAW_DIR, "raw-1", current=True)
    (generation / "payload.bin").write_bytes(b"x" * 128)

    records = list_experiment_generations(run_root, include_size=True)

    assert records[0].size_bytes is not None
    assert records[0].size_bytes >= 128


def test_to_dict_is_json_serializable(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs"
    _publish(run_root / RAW_DIR, "raw-1", current=True)

    payload = list_experiment_generations(run_root)[0].to_dict()

    assert json.loads(json.dumps(payload))["generation_id"] == "raw-1"
    assert isinstance(payload["issues"], list)


def test_project_embedding_generations_are_listed(tmp_path: Path) -> None:
    from smftools.project.set_store import sets_root

    definition_dir = sets_root(tmp_path) / "my_set" / "embeddings" / "def0123"
    _publish(definition_dir, "emb-1", current=True)

    records = list_project_generations(tmp_path)

    assert len(records) == 1
    assert records[0].scope == "project"
    assert records[0].kind == "embedding"
    assert records[0].is_current is True
    assert records[0].container.endswith("embeddings/def0123")


def test_project_without_embeddings_lists_empty(tmp_path: Path) -> None:
    assert list_project_generations(tmp_path) == []
