from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.informatics.generation import GENERATION_MANIFEST, staged_generation
from smftools.informatics.generation_retention import (
    RETENTION_FILENAME,
    GenerationRetentionError,
    pin_generation,
    read_generation_retention,
    unpin_generation,
)

pytestmark = pytest.mark.unit


def _publish(container: Path, generation_id: str = "generation-a") -> Path:
    with staged_generation(container, generation_id=generation_id) as staged:
        staged.artifact("artifact.bin").write_bytes(b"immutable")
        staged.record_manifest({"schema_version": 1, "status": "complete"})
    return staged.final_dir


def test_pin_is_external_to_immutable_generation_and_supports_multiple_reasons(
    tmp_path: Path,
) -> None:
    container = tmp_path / "raw_outputs"
    generation = _publish(container)
    manifest_path = generation / GENERATION_MANIFEST
    manifest_before = manifest_path.read_bytes()
    pointer_before = (container / "current.json").read_bytes()

    first = pin_generation(container, "generation-a", reason="paper figure 3")
    second = pin_generation(container, "generation-a", reason="doi:10.1234/example")

    assert first.pinned is True
    assert [reason.reason for reason in second.reasons] == [
        "paper figure 3",
        "doi:10.1234/example",
    ]
    assert manifest_path.read_bytes() == manifest_before
    assert (container / "current.json").read_bytes() == pointer_before
    assert not (generation / RETENTION_FILENAME).exists()
    assert (container / RETENTION_FILENAME).is_file()


def test_repeating_a_reason_is_idempotent(tmp_path: Path) -> None:
    container = tmp_path / "raw_outputs"
    _publish(container)
    pin_generation(container, "generation-a", reason="manual hold")
    registry = container / RETENTION_FILENAME
    before = registry.read_bytes()

    entry = pin_generation(container, "generation-a", reason="manual hold")

    assert [reason.reason for reason in entry.reasons] == ["manual hold"]
    assert registry.read_bytes() == before


def test_unpin_one_reason_then_all_reasons(tmp_path: Path) -> None:
    container = tmp_path / "raw_outputs"
    _publish(container)
    pin_generation(container, "generation-a", reason="paper figure 3")
    pin_generation(container, "generation-a", reason="SRA:ABC123")

    remaining = unpin_generation(container, "generation-a", reason="paper figure 3")
    assert remaining is not None
    assert [reason.reason for reason in remaining.reasons] == ["SRA:ABC123"]

    assert unpin_generation(container, "generation-a") is None
    assert read_generation_retention(container) == {}
    payload = json.loads((container / RETENTION_FILENAME).read_text())
    assert payload == {"generations": {}, "schema_version": 1}


@pytest.mark.parametrize("generation_id", ["", "../escape", "nested/id", "/absolute"])
def test_pin_rejects_unsafe_generation_ids(tmp_path: Path, generation_id: str) -> None:
    with pytest.raises(GenerationRetentionError, match="not portable"):
        pin_generation(tmp_path / "raw_outputs", generation_id, reason="hold")


def test_pin_rejects_missing_generation(tmp_path: Path) -> None:
    with pytest.raises(GenerationRetentionError, match="does not exist"):
        pin_generation(tmp_path / "raw_outputs", "missing", reason="hold")


def test_corrupt_registry_is_rejected_without_replacement(tmp_path: Path) -> None:
    container = tmp_path / "raw_outputs"
    _publish(container)
    registry = container / RETENTION_FILENAME
    registry.write_text("{", encoding="utf-8")

    with pytest.raises(GenerationRetentionError, match="unreadable"):
        pin_generation(container, "generation-a", reason="hold")

    assert registry.read_text(encoding="utf-8") == "{"
