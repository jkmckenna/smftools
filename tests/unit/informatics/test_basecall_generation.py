"""Basecall's generation-publishing lifecycle (`BCS-05`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from smftools.constants import BASECALL_DIR
from smftools.informatics.basecall_generation import (
    BasecallGenerationError,
    publish_basecall_generation,
    resolve_current_basecall_generation,
    validate_basecall_generation,
)

pytestmark = pytest.mark.unit


def _bam(tmp_path: Path, content: bytes = b"fake-bam-bytes") -> Path:
    path = tmp_path / "source" / "calls.bam"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _publish(tmp_path: Path, **overrides) -> dict:
    kwargs = {
        "run_root": tmp_path / "run",
        "bam_path": _bam(tmp_path),
        "model": "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
        "modality": "deaminase",
        "config_hash": "hash0001",
        "input_artifact_ids": ["sha256:abc123"],
        "dorado_version": "1.3.1",
    }
    kwargs.update(overrides)
    return publish_basecall_generation(**kwargs)


def test_publish_writes_the_expected_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    outputs = _publish(tmp_path, run_root=run_root)

    basecall_dir = run_root / BASECALL_DIR
    assert (basecall_dir / "current.json").is_file()
    generation_dir = basecall_dir / "generations" / outputs["generation_id"]
    assert generation_dir.is_dir()
    assert (generation_dir / "generation_manifest.json").is_file()
    assert outputs["bam"] == generation_dir / "basecalls.bam"
    assert outputs["bam"].read_bytes() == b"fake-bam-bytes"


def test_publish_records_model_and_dorado_version(tmp_path: Path) -> None:
    outputs = _publish(tmp_path)

    import json

    manifest = json.loads(outputs["generation_manifest"].read_text())
    assert manifest["model"] == "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
    assert manifest["dorado_version"] == "1.3.1"
    assert manifest["input_artifact_ids"] == ["sha256:abc123"]


def test_publish_requires_an_existing_bam(tmp_path: Path) -> None:
    with pytest.raises(BasecallGenerationError, match="missing"):
        _publish(tmp_path, bam_path=tmp_path / "nope.bam")


def test_publish_requires_a_model_name(tmp_path: Path) -> None:
    with pytest.raises(BasecallGenerationError, match="model"):
        _publish(tmp_path, model="")


def test_publish_requires_input_artifact_ids(tmp_path: Path) -> None:
    with pytest.raises(BasecallGenerationError, match="input_artifact_ids"):
        _publish(tmp_path, input_artifact_ids=[])


def test_resolve_current_returns_the_published_generation(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    outputs = _publish(tmp_path, run_root=run_root)

    resolved = resolve_current_basecall_generation(run_root / BASECALL_DIR)

    assert resolved is not None
    generation_dir, manifest = resolved
    assert generation_dir == outputs["generation"]
    assert manifest["generation_id"] == outputs["generation_id"]


def test_resolve_current_none_when_nothing_published(tmp_path: Path) -> None:
    assert resolve_current_basecall_generation(tmp_path / "nothing" / BASECALL_DIR) is None


def test_validate_rejects_a_corrupted_bam(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    outputs = _publish(tmp_path, run_root=run_root)

    outputs["bam"].write_bytes(b"tampered")

    with pytest.raises(BasecallGenerationError, match="corrupt"):
        validate_basecall_generation(outputs["generation"])


def test_two_generations_can_coexist_and_current_points_at_the_second(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    first = _publish(tmp_path, run_root=run_root)
    second = _publish(
        tmp_path,
        run_root=run_root,
        bam_path=_bam(tmp_path, b"second-bam-bytes"),
        model="dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
    )

    assert first["generation"] != second["generation"]
    resolved = resolve_current_basecall_generation(run_root / BASECALL_DIR)
    assert resolved is not None
    assert resolved[1]["generation_id"] == second["generation_id"]


def test_extra_manifest_fields_are_merged_without_overriding_core_fields(tmp_path: Path) -> None:
    outputs = _publish(
        tmp_path,
        extra_manifest_fields={"barcode_kit": "SQK-NBD114-24", "status": "not-actually-complete"},
    )

    import json

    manifest = json.loads(outputs["generation_manifest"].read_text())
    assert manifest["barcode_kit"] == "SQK-NBD114-24"
    assert manifest["status"] == "complete"  # extra fields never shadow the real ones
