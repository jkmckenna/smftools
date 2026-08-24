"""Sweeping staging trees left by a run that died mid-publish."""

from __future__ import annotations

import os
import time

import pytest

from smftools.informatics.generation import (
    GENERATIONS_SUBDIR,
    STAGING_SUBDIR,
    staged_generation,
    sweep_abandoned_staging,
)

pytestmark = pytest.mark.unit


def _staging_tree(output_dir, name, *, age_seconds=0.0):
    tree = output_dir / STAGING_SUBDIR / name
    tree.mkdir(parents=True)
    (tree / "partial.bin").write_bytes(b"x")
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(tree, (stamp, stamp))
    return tree


def test_aged_abandoned_tree_is_swept(tmp_path):
    """A SIGKILL leaves the tree behind and nothing else ever reclaims it."""
    old = _staging_tree(tmp_path, "abandoned", age_seconds=48 * 3600)

    removed = sweep_abandoned_staging(tmp_path)

    assert removed == (old,)
    assert not old.exists()


def test_recent_tree_is_left_for_inspection(tmp_path):
    """A staging tree is the only evidence when a publish dies."""
    fresh = _staging_tree(tmp_path, "just-failed")

    assert sweep_abandoned_staging(tmp_path) == ()
    assert fresh.exists()


def test_a_concurrent_publish_is_not_swept(tmp_path):
    """`keep` protects the tree the caller is about to build into.

    Age alone would already protect it, but not if a build runs longer than the
    cutoff -- and raw extraction alone is over half an hour.
    """
    mine = _staging_tree(tmp_path, "mine", age_seconds=48 * 3600)

    assert sweep_abandoned_staging(tmp_path, keep="mine") == ()
    assert mine.exists()


def test_a_published_id_is_never_removed(tmp_path):
    """If the id exists as a real generation the move already happened."""
    tree = _staging_tree(tmp_path, "published", age_seconds=48 * 3600)
    (tmp_path / GENERATIONS_SUBDIR / "published").mkdir(parents=True)

    assert sweep_abandoned_staging(tmp_path) == ()
    assert tree.exists()


def test_publishing_sweeps_older_abandoned_trees(tmp_path):
    """The sweep runs where the leak happens, not only when asked."""
    stale = _staging_tree(tmp_path, "stale", age_seconds=48 * 3600)

    with staged_generation(tmp_path, generation_id="fresh") as staged:
        staged.record_manifest({"status": "complete"})

    assert not stale.exists()
    assert (tmp_path / GENERATIONS_SUBDIR / "fresh").is_dir()


def test_missing_staging_root_is_not_an_error(tmp_path):
    assert sweep_abandoned_staging(tmp_path) == ()
