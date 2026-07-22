import json

import pytest

from smftools.readwrite import atomic_write_json


def test_atomic_write_json_publishes_complete_document(tmp_path):
    path = tmp_path / "manifest.json"

    result = atomic_write_json(path, {"state": "complete", "tasks": 3})

    assert result == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "state": "complete",
        "tasks": 3,
    }
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_atomic_write_json_preserves_existing_target_when_publish_fails(tmp_path, monkeypatch):
    import smftools.readwrite as readwrite

    path = tmp_path / "manifest.json"
    path.write_text('{"state": "complete"}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(readwrite.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated publication failure"):
        atomic_write_json(path, {"state": "running"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "complete"}
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))
