from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from smftools.informatics.generation import (
    CURRENT_FILENAME,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    STAGING_SUBDIR,
    GenerationError,
    has_published_generations,
    resolve_current_generation,
    resolve_stage_generation,
    staged_generation,
)

pytestmark = pytest.mark.unit


def _publish(output_dir: Path, *, payload=None, generation_id=None, body=b"x") -> str:
    with staged_generation(output_dir, generation_id=generation_id) as staged:
        staged.artifact("store", "part-0.parquet").write_bytes(body)
        staged.record_manifest(payload or {"schema_version": 1, "status": "complete"})
    return staged.generation_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publishes_generation_and_advances_current(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"

    generation_id = _publish(out)

    generation_dir = out / GENERATIONS_SUBDIR / generation_id
    assert (generation_dir / GENERATION_MANIFEST).is_file()
    assert (generation_dir / "store" / "part-0.parquet").is_file()

    pointer = json.loads((out / CURRENT_FILENAME).read_text())
    assert pointer["generation_id"] == generation_id
    assert pointer["generation_path"] == f"{GENERATIONS_SUBDIR}/{generation_id}"
    assert not Path(pointer["generation_path"]).is_absolute()

    resolved = resolve_current_generation(out)
    assert resolved is not None
    assert resolved[0] == generation_dir
    assert resolved[1]["generation_id"] == generation_id


def test_manifest_generation_id_is_stamped_not_trusted(tmp_path: Path) -> None:
    """The manifest and directory name cannot disagree, even if a caller tries."""
    out = tmp_path / "hmm_adata_outputs"

    generation_id = _publish(out, payload={"schema_version": 1, "generation_id": "lies"})

    manifest = json.loads(
        (out / GENERATIONS_SUBDIR / generation_id / GENERATION_MANIFEST).read_text()
    )
    assert manifest["generation_id"] == generation_id


def test_second_generation_supersedes_without_destroying_the_first(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"
    first = _publish(out, body=b"first")
    second = _publish(out, body=b"second")

    assert first != second
    # Both remain addressable; only the pointer moved.
    assert (out / GENERATIONS_SUBDIR / first / "store" / "part-0.parquet").read_bytes() == b"first"
    assert (
        out / GENERATIONS_SUBDIR / second / "store" / "part-0.parquet"
    ).read_bytes() == b"second"
    assert resolve_current_generation(out)[1]["generation_id"] == second


def test_stage_generation_resolves_current_or_lineage_pin(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"
    first = _publish(out, body=b"first")
    second = _publish(out, body=b"second")

    assert resolve_stage_generation(out)[1]["generation_id"] == second
    pinned = resolve_stage_generation(out, lineage=first)
    assert pinned is not None
    assert pinned[0] == out / GENERATIONS_SUBDIR / first
    assert pinned[1]["generation_id"] == first


@pytest.mark.parametrize("lineage", ["", "../escape", "nested/generation", "/absolute"])
def test_stage_generation_rejects_unsafe_lineage_pin(tmp_path: Path, lineage: str) -> None:
    with pytest.raises(GenerationError, match="not portable"):
        resolve_stage_generation(tmp_path / "hmm_adata_outputs", lineage=lineage)


def test_stage_generation_rejects_missing_lineage_pin(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="does not exist"):
        resolve_stage_generation(tmp_path / "hmm_adata_outputs", lineage="missing")


def test_stage_generation_rejects_lineage_manifest_identity_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"
    generation_id = _publish(out)
    manifest_path = out / GENERATIONS_SUBDIR / generation_id / GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["generation_id"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GenerationError, match="disagrees with its manifest"):
        resolve_stage_generation(out, lineage=generation_id)


def test_failure_inside_the_block_publishes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"
    published = _publish(out, body=b"good")

    with pytest.raises(RuntimeError, match="boom"):
        with staged_generation(out) as staged:
            staged.artifact("store", "part-0.parquet").write_bytes(b"bad")
            staged.record_manifest({"schema_version": 1})
            raise RuntimeError("boom")

    # The previously current generation is untouched and still selected.
    assert resolve_current_generation(out)[1]["generation_id"] == published
    assert [d.name for d in (out / GENERATIONS_SUBDIR).iterdir()] == [published]
    assert not any((out / STAGING_SUBDIR).iterdir())


def test_validator_failure_aborts_before_publication(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"

    def _reject(staging_dir, final_dir, run_root):
        raise GenerationError("incomplete")

    with pytest.raises(GenerationError, match="incomplete"):
        with staged_generation(out, validate=_reject) as staged:
            staged.artifact("store", "part-0.parquet").write_bytes(b"x")
            staged.record_manifest({"schema_version": 1})

    assert not (out / CURRENT_FILENAME).exists()
    assert not has_published_generations(out)


def test_manifest_checksum_pointer_is_opt_in_and_verified(tmp_path: Path) -> None:
    out = tmp_path / "raw_outputs"
    with staged_generation(out, generation_id="checked", manifest_checksum=_sha256) as staged:
        staged.artifact("artifact.bin").write_bytes(b"content")
        staged.record_manifest({"schema_version": 1, "status": "complete"})

    pointer = json.loads((out / CURRENT_FILENAME).read_text())
    manifest_path = out / pointer["generation_path"] / GENERATION_MANIFEST
    assert pointer == {
        "schema_version": 1,
        "generation_id": "checked",
        "generation_path": "generations/checked",
        "manifest_sha256": _sha256(manifest_path),
    }

    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(GenerationError, match="checksum does not match"):
        resolve_current_generation(out, manifest_checksum=_sha256)


def test_after_current_failure_restores_previous_selection(tmp_path: Path) -> None:
    out = tmp_path / "preprocess_adata_outputs"
    first = _publish(out, generation_id="first")

    def reject_after_current(_staging: Path, _final: Path, _run_root: Path) -> None:
        raise RuntimeError("canonical publication failed")

    with pytest.raises(RuntimeError, match="canonical publication failed"):
        with staged_generation(
            out,
            generation_id="second",
            after_current=reject_after_current,
        ) as staged:
            staged.artifact("artifact.bin").write_bytes(b"content")
            staged.record_manifest({"schema_version": 1, "status": "complete"})

    assert resolve_current_generation(out)[1]["generation_id"] == first
    assert not (out / GENERATIONS_SUBDIR / "second").exists()


def test_missing_manifest_is_an_error_and_publishes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"

    with pytest.raises(GenerationError, match="recorded no manifest"):
        with staged_generation(out) as staged:
            staged.artifact("store", "part-0.parquet").write_bytes(b"x")

    assert not (out / CURRENT_FILENAME).exists()
    assert not has_published_generations(out)


def test_republishing_an_existing_id_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"
    _publish(out, generation_id="fixed-id")

    with pytest.raises(GenerationError, match="already published"):
        _publish(out, generation_id="fixed-id")


def test_legacy_in_place_directory_resolves_to_none(tmp_path: Path) -> None:
    """A stage that predates generations is absence of a pointer, not an error."""
    out = tmp_path / "hmm_adata_outputs"
    out.mkdir()
    (out / "spine.h5ad").write_bytes(b"")

    assert resolve_current_generation(out) is None
    assert not has_published_generations(out)


@pytest.mark.parametrize(
    ("pointer", "message"),
    [
        ({"schema_version": 99, "generation_path": "generations/x"}, "schema is incompatible"),
        ({"schema_version": 1, "generation_path": "/etc"}, "not portable"),
        ({"schema_version": 1, "generation_path": "../escape"}, "not portable"),
        ({"schema_version": 1, "generation_path": ""}, "not portable"),
    ],
)
def test_unsafe_current_pointers_are_rejected(tmp_path: Path, pointer, message) -> None:
    out = tmp_path / "hmm_adata_outputs"
    out.mkdir()
    (out / CURRENT_FILENAME).write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(GenerationError, match=message):
        resolve_current_generation(out)


def test_pointer_naming_a_different_generation_than_the_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    out = tmp_path / "hmm_adata_outputs"
    generation_id = _publish(out)
    pointer_path = out / CURRENT_FILENAME
    pointer = json.loads(pointer_path.read_text())
    pointer["generation_id"] = "someone-else"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(GenerationError, match="does not"):
        resolve_current_generation(out)

    assert generation_id  # the generation itself is still on disk


def test_unreadable_pointer_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "hmm_adata_outputs"
    out.mkdir()
    (out / CURRENT_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(GenerationError, match="unreadable"):
        resolve_current_generation(out)
