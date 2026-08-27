"""Named storage roots make a config portable (`PSR-04`-`PSR-07`).

Written as absolute paths, a config's values are correct on exactly one machine
with exactly one set of drives mounted, so moving a tree means editing every
config. `${data}/...` resolves through a machine-local binding instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from smftools.config.roots import (
    ENV_PREFIX,
    RootResolutionError,
    expand_roots,
    known_roots,
    qualify_with_root,
    referenced_roots,
    resolve_config_path,
    resolve_root,
)
from smftools.informatics.artifact_paths import resolve_artifact_path, serialize_artifact_path

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path, monkeypatch):
    """No inherited bindings, and a config dir that owns its own roots file."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))
    return tmp_path


def _roots_file(directory: Path, **bindings) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'{name} = "{value}"' for name, value in bindings.items())
    path = directory / "roots.toml"
    path.write_text(f"[roots]\n{body}\n", encoding="utf-8")
    return path


def test_environment_binding_wins(tmp_path, monkeypatch):
    _roots_file(tmp_path / "cfg", data=str(tmp_path / "from-file"))
    monkeypatch.setenv(f"{ENV_PREFIX}DATA", str(tmp_path / "from-env"))
    binding = resolve_root("data", config_dir=tmp_path / "cfg")
    assert binding.path == tmp_path / "from-env"
    assert binding.source == f"{ENV_PREFIX}DATA"


def test_walk_up_finds_a_roots_file_above_the_config(tmp_path):
    _roots_file(tmp_path / "lab", data=str(tmp_path / "archive"))
    nested = tmp_path / "lab" / "analyses" / "runs" / "demo"
    nested.mkdir(parents=True)
    binding = resolve_root("data", config_dir=nested)
    assert binding.path == tmp_path / "archive"


def test_nearest_roots_file_wins(tmp_path):
    _roots_file(tmp_path / "lab", data=str(tmp_path / "far"))
    _roots_file(tmp_path / "lab" / "analyses", data=str(tmp_path / "near"))
    nested = tmp_path / "lab" / "analyses" / "runs"
    nested.mkdir(parents=True)
    assert resolve_root("data", config_dir=nested).path == tmp_path / "near"


def test_unbound_root_is_an_error_not_a_literal(tmp_path):
    """A typo'd root name must not become a directory name."""
    with pytest.raises(RootResolutionError) as excinfo:
        expand_roots("${nowhere}/x", config_dir=tmp_path, field="input_data_path")
    message = str(excinfo.value)
    assert "input_data_path" in message
    # The remedy must be actionable, naming both ways to bind it.
    assert f"{ENV_PREFIX}NOWHERE" in message
    assert "roots.toml" in message or "[roots]" in message


def test_expansion_is_only_for_braced_references(tmp_path):
    assert referenced_roots("${data}/a/${other}") == ["data", "other"]
    # Bare `$data` collides with ordinary path text and is left alone.
    assert referenced_roots("$data/a") == []
    assert expand_roots("/plain/path", config_dir=tmp_path) == "/plain/path"


def test_relative_paths_anchor_to_the_config_not_the_cwd(tmp_path):
    """The same config must mean the same thing wherever it is run from."""
    config_dir = tmp_path / "runs" / "demo"
    config_dir.mkdir(parents=True)
    resolved = resolve_config_path("ref.fasta", config_dir=config_dir, field="fasta")
    assert Path(resolved) == config_dir / "ref.fasta"


def test_absolute_paths_are_untouched(tmp_path):
    assert resolve_config_path(str(tmp_path), config_dir=tmp_path) == str(tmp_path)


def test_no_config_dir_preserves_legacy_relative_behaviour():
    """Programmatic construction has no config file to anchor against."""
    assert resolve_config_path("ref.fasta", config_dir=None) == "ref.fasta"


def test_known_roots_reports_where_each_binding_came_from(tmp_path, monkeypatch):
    _roots_file(tmp_path / "lab", analyses=str(tmp_path / "an"))
    monkeypatch.setenv(f"{ENV_PREFIX}DATA", str(tmp_path / "dt"))
    bindings = known_roots(config_dir=tmp_path / "lab")
    assert bindings["data"].source == f"{ENV_PREFIX}DATA"
    assert bindings["analyses"].source.endswith("roots.toml")


# --- artifact pointer encodings ---------------------------------------------


def test_path_inside_the_anchor_stays_plain_relative(tmp_path):
    """Nothing is more stable than a pair that moves as a unit."""
    anchor = tmp_path / "project"
    target = anchor / "runs" / "expA"
    target.mkdir(parents=True)
    assert serialize_artifact_path(target, anchor) == "runs/expA"


def test_path_under_a_bound_root_is_qualified(tmp_path, monkeypatch):
    """Otherwise the pointer encodes the mount name and the anchor's depth."""
    runs = tmp_path / "external" / "runs" / "expA"
    runs.mkdir(parents=True)
    anchor = tmp_path / "project"
    anchor.mkdir()
    monkeypatch.setenv(f"{ENV_PREFIX}ANALYSES", str(tmp_path / "external"))
    assert serialize_artifact_path(runs, anchor) == "${analyses}/runs/expA"


def test_the_most_specific_root_wins(tmp_path, monkeypatch):
    target = tmp_path / "lab" / "analyses" / "runs" / "expA"
    target.mkdir(parents=True)
    anchor = tmp_path / "project"
    anchor.mkdir()
    monkeypatch.setenv(f"{ENV_PREFIX}LAB", str(tmp_path / "lab"))
    monkeypatch.setenv(f"{ENV_PREFIX}ANALYSES", str(tmp_path / "lab" / "analyses"))
    assert serialize_artifact_path(target, anchor) == "${analyses}/runs/expA"


def test_all_three_encodings_resolve(tmp_path, monkeypatch):
    anchor = tmp_path / "project"
    anchor.mkdir()
    external = tmp_path / "external" / "runs" / "expA"
    external.mkdir(parents=True)
    monkeypatch.setenv(f"{ENV_PREFIX}ANALYSES", str(tmp_path / "external"))
    assert resolve_artifact_path("runs/expA", anchor) == (anchor / "runs" / "expA").resolve()
    assert resolve_artifact_path("${analyses}/runs/expA", anchor) == external
    # Legacy absolute pointers keep working without being rewritten.
    assert resolve_artifact_path(str(external), anchor) == external


def test_qualification_returns_none_without_a_containing_root(tmp_path):
    assert qualify_with_root(tmp_path / "somewhere", config_dir=tmp_path) is None


def test_round_trip_through_an_unbound_root_is_an_error(tmp_path):
    """A pointer that cannot resolve must say so, not silently become literal."""
    with pytest.raises(RootResolutionError):
        resolve_artifact_path("${unbound}/runs/expA", tmp_path)


def test_working_directory_fallback_keeps_pre_psr05_configs_working(tmp_path, monkeypatch, caplog):
    """The gate `PSR-05` called for.

    Configs written before this resolved relative paths against the working
    directory. Silently repointing them at the config's directory would break
    working setups on upgrade, so the old reading is honoured -- but only when
    the new one names nothing and the old one names something real, and it says
    so, so the config gets fixed rather than depending on the cwd forever.
    """
    import logging

    config_dir = tmp_path / "runs" / "demo"
    config_dir.mkdir(parents=True)
    workdir = tmp_path / "elsewhere"
    (workdir / "inputs").mkdir(parents=True)
    (workdir / "inputs" / "reads.fastq").write_text("", encoding="utf-8")
    monkeypatch.chdir(workdir)

    with caplog.at_level(logging.WARNING, logger="smftools.config.roots"):
        resolved = resolve_config_path(
            "inputs/reads.fastq", config_dir=config_dir, field="input_data_path"
        )
    assert Path(resolved) == (workdir / "inputs" / "reads.fastq").resolve()
    assert any("working directory" in record.message for record in caplog.records)


def test_config_relative_wins_when_both_exist(tmp_path, monkeypatch):
    """The ambiguous case takes the new, intended reading."""
    config_dir = tmp_path / "runs" / "demo"
    (config_dir / "inputs").mkdir(parents=True)
    (config_dir / "inputs" / "reads.fastq").write_text("", encoding="utf-8")
    workdir = tmp_path / "elsewhere"
    (workdir / "inputs").mkdir(parents=True)
    (workdir / "inputs" / "reads.fastq").write_text("", encoding="utf-8")
    monkeypatch.chdir(workdir)

    resolved = resolve_config_path("inputs/reads.fastq", config_dir=config_dir)
    assert Path(resolved) == (config_dir / "inputs" / "reads.fastq").resolve()


def test_nonexistent_relative_path_reports_the_config_relative_reading(tmp_path, monkeypatch):
    """When neither exists, the error should name the path the config meant."""
    config_dir = tmp_path / "runs" / "demo"
    config_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    resolved = resolve_config_path("missing.fastq", config_dir=config_dir)
    assert Path(resolved) == (config_dir / "missing.fastq").resolve()
