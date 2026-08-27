"""A root resolves over an ordered set of locations, not just one (`PSR-16`).

`data/` is simultaneously where new collection lands and where archive drives
mount; `analyses/` may hold some runs locally and others on an external SSD.
A root's binding may therefore be a list of candidate locations rather than a
single string.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from smftools.config.roots import (
    ENV_PREFIX,
    expand_roots,
    known_roots,
    qualify_with_root,
    resolve_root,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))
    return tmp_path


def _roots_file_with_list(directory: Path, name: str, *candidates: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entries = ", ".join(f'"{candidate}"' for candidate in candidates)
    path = directory / "roots.toml"
    path.write_text(f"[roots]\n{name} = [{entries}]\n", encoding="utf-8")
    return path


def test_resolve_picks_the_first_existing_candidate(tmp_path):
    first = tmp_path / "first"  # does not exist
    second = tmp_path / "second"
    second.mkdir()
    _roots_file_with_list(tmp_path / "cfg", "analyses", first, second)

    binding = resolve_root("analyses", config_dir=tmp_path / "cfg")

    assert binding.path == second
    assert binding.all_paths == (first, second)


def test_resolve_falls_back_to_the_first_candidate_when_none_exist(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _roots_file_with_list(tmp_path / "cfg", "analyses", first, second)

    binding = resolve_root("analyses", config_dir=tmp_path / "cfg")

    # Neither exists yet -- a run being created for the first time lands at
    # the primary (first-listed) location.
    assert binding.path == first


def test_a_single_string_binding_still_has_a_one_element_all_paths(tmp_path):
    (tmp_path / "cfg").mkdir(parents=True)
    (tmp_path / "cfg" / "roots.toml").write_text(
        f'[roots]\ndata = "{tmp_path / "only"}"\n', encoding="utf-8"
    )

    binding = resolve_root("data", config_dir=tmp_path / "cfg")

    assert binding.all_paths == (tmp_path / "only",)


def test_expand_roots_substitutes_the_winning_candidate(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    second.mkdir()
    _roots_file_with_list(tmp_path / "cfg", "analyses", first, second)

    expanded = expand_roots("${analyses}/run1", config_dir=tmp_path / "cfg")

    assert expanded == str(second / "run1")


def test_known_roots_preserves_all_paths_for_list_bindings(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    second.mkdir()
    _roots_file_with_list(tmp_path / "cfg", "analyses", first, second)

    bindings = known_roots(config_dir=tmp_path / "cfg")

    assert bindings["analyses"].all_paths == (first, second)


def test_qualify_with_root_finds_a_path_under_a_non_winning_candidate(tmp_path):
    """The gap this item closes.

    The *first* candidate happens to exist (so it wins expansion), but the
    path being qualified actually lives under the *second* candidate -- e.g.
    a run whose analysis tree sits on a second, currently-detached SSD.
    Naively checking only the winning candidate would fail to recognize it.
    """
    first = tmp_path / "first"
    first.mkdir()  # exists, and wins expansion -- but contains nothing relevant
    second = tmp_path / "second"
    run_dir = second / "runs" / "run1"
    run_dir.mkdir(parents=True)  # exists too, and *does* contain the target path
    _roots_file_with_list(tmp_path / "cfg", "analyses", first, second)

    qualified = qualify_with_root(run_dir, config_dir=tmp_path / "cfg")

    assert qualified == "${analyses}/runs/run1"


def test_qualify_with_root_still_prefers_the_most_specific_match(tmp_path):
    outer = tmp_path / "lab"
    inner = outer / "analyses"
    inner.mkdir(parents=True)
    (tmp_path / "cfg").mkdir(parents=True)
    (tmp_path / "cfg" / "roots.toml").write_text(
        f'[roots]\nlab = "{outer}"\nanalyses = "{inner}"\n', encoding="utf-8"
    )

    qualified = qualify_with_root(inner / "run1", config_dir=tmp_path / "cfg")

    assert qualified == "${analyses}/run1"


def test_qualify_with_root_none_when_no_candidate_of_any_root_contains_it(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    _roots_file_with_list(tmp_path / "cfg", "analyses", first, second)
    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()

    assert qualify_with_root(elsewhere, config_dir=tmp_path / "cfg") is None
