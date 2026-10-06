"""MRC-02: materialize reuses a loaded spine within a process."""

import os

import numpy as np
import pytest

from smftools.informatics import partition_read
from smftools.informatics.partition_read import clear_spine_cache, materialize

from .test_preprocess_x_fast_path import _build


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_spine_cache()
    yield
    clear_spine_cache()


@pytest.fixture
def loads(monkeypatch):
    calls = []
    original = partition_read.load_spine

    def counting(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(partition_read, "load_spine", counting)
    return calls


def _built(tmp_path, loads):
    """Stores built with the real writers; their own spine loads not counted."""
    raw, preprocess = _build(tmp_path)
    loads.clear()
    return raw, preprocess


def _read(spine):
    return materialize(spine, read_ids=["read1"], start=0, end=12, layers=["nan_half"])


def test_repeat_calls_load_the_spine_once(tmp_path, loads):
    _, preprocess = _built(tmp_path, loads)
    first = _read(preprocess["spine"])
    second = _read(preprocess["spine"])
    assert len(loads) == 1
    np.testing.assert_array_equal(first.X, second.X)


def test_a_rewritten_spine_is_reloaded(tmp_path, loads):
    _, preprocess = _built(tmp_path, loads)
    _read(preprocess["spine"])
    stat = os.stat(preprocess["spine"])
    os.utime(preprocess["spine"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    _read(preprocess["spine"])
    assert len(loads) == 2


def test_the_cache_is_bounded(tmp_path, loads, monkeypatch):
    monkeypatch.setattr(partition_read, "_SPINE_CACHE_SIZE", 1)
    _, first = _build(tmp_path / "a")
    _, second = _build(tmp_path / "b")
    loads.clear()
    _read(first["spine"])
    _read(second["spine"])  # evicts the first
    _read(first["spine"])
    assert len(loads) == 3
    assert len(partition_read._SPINE_CACHE) == 1


def test_cached_and_uncached_reads_agree(tmp_path, monkeypatch):
    _, preprocess = _build(tmp_path)
    cached = _read(preprocess["spine"])
    monkeypatch.setenv("SMFTOOLS_SPINE_CACHE", "0")
    uncached = _read(preprocess["spine"])
    assert list(cached.obs_names) == list(uncached.obs_names)
    np.testing.assert_array_equal(cached.X, uncached.X)
    np.testing.assert_array_equal(cached.layers["nan_half"], uncached.layers["nan_half"])


def test_editing_a_result_does_not_touch_the_cached_spine(tmp_path, monkeypatch):
    raw, _ = _build(tmp_path)
    # The raw spine takes the per-partition path, which copies spine maps
    # into the result's uns.
    first = materialize(raw["spine"], read_ids=["read1"], start=0, end=12)
    first.uns["References"]["ref_FASTA_sequence"] = "edited"
    second = materialize(raw["spine"], read_ids=["read1"], start=0, end=12)
    assert second.uns["References"]["ref_FASTA_sequence"] == "ACGCGTACGTAC"
