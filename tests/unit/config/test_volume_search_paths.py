"""Extra volume search paths for drives outside the platform's mount conventions (`PSR-09`)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from smftools.config.roots import ENV_VOLUME_SEARCH_PATHS, extra_volume_search_paths

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_search_paths(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VOLUME_SEARCH_PATHS, raising=False)
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))
    return tmp_path


def _roots_file(directory: Path, *, extra_search_paths: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entries = ", ".join(f'"{p}"' for p in extra_search_paths)
    path = directory / "roots.toml"
    path.write_text(f"[volumes]\nextra_search_paths = [{entries}]\n", encoding="utf-8")
    return path


def test_no_configuration_returns_empty(tmp_path):
    assert extra_volume_search_paths(config_dir=tmp_path) == []


def test_env_var_wins_outright(tmp_path, monkeypatch):
    _roots_file(tmp_path / "cfg", extra_search_paths=[str(tmp_path / "from-file")])
    monkeypatch.setenv(
        ENV_VOLUME_SEARCH_PATHS, os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")])
    )

    result = extra_volume_search_paths(config_dir=tmp_path / "cfg")

    assert result == [tmp_path / "a", tmp_path / "b"]


def test_user_file_and_walk_up_file_are_unioned(tmp_path, monkeypatch):
    user_dir = tmp_path / "user-config"
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(user_dir))
    _roots_file(user_dir, extra_search_paths=[str(tmp_path / "from-user")])

    lab = tmp_path / "lab"
    nested = lab / "analyses" / "runs" / "demo"
    nested.mkdir(parents=True)
    _roots_file(lab, extra_search_paths=[str(tmp_path / "from-walkup")])

    result = extra_volume_search_paths(config_dir=nested)

    assert result == [tmp_path / "from-user", tmp_path / "from-walkup"]


def test_duplicate_entries_are_deduped(tmp_path, monkeypatch):
    user_dir = tmp_path / "user-config"
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(user_dir))
    shared = str(tmp_path / "shared")
    _roots_file(user_dir, extra_search_paths=[shared])

    lab = tmp_path / "lab"
    lab.mkdir()
    _roots_file(lab, extra_search_paths=[shared])

    result = extra_volume_search_paths(config_dir=lab)

    assert result == [tmp_path / "shared"]
