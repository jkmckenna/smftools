import numpy as np

from smftools.informatics.modkit_extract_to_adata import (
    _load_sample_record_batches_cached,
    _load_sequence_batches,
    _normalize_sequence_batch_files,
    _write_integer_batches,
)


def test_normalize_sequence_batch_files_handles_scalars() -> None:
    result = _normalize_sequence_batch_files("foo/bar.h5ad")
    assert [path.as_posix() for path in result] == ["foo/bar.h5ad"]


def test_normalize_sequence_batch_files_filters_empty_values() -> None:
    result = _normalize_sequence_batch_files(["", ".", None, "foo.h5ad"])
    assert [path.as_posix() for path in result] == ["foo.h5ad"]


def test_normalize_sequence_batch_files_handles_numpy_arrays() -> None:
    result = _normalize_sequence_batch_files(np.array(["foo.h5ad", "bar.h5ad"]))
    assert [path.as_posix() for path in result] == ["foo.h5ad", "bar.h5ad"]


def test_write_integer_batches_round_trip(tmp_path) -> None:
    sequences = {
        "read1": np.array([1, 2, 3], dtype=np.int16),
        "read2": np.array([4, 5, 6], dtype=np.int16),
    }

    batch_files = _write_integer_batches(
        sequences,
        tmp_path,
        record="chr1",
        prefix="mismatch_fwd",
        batch_size=1,
    )

    loaded, fwd_reads, rev_reads = _load_sequence_batches(batch_files)
    assert loaded["read1"].tolist() == [1, 2, 3]
    assert loaded["read2"].tolist() == [4, 5, 6]
    assert fwd_reads == {"read1", "read2"}
    assert rev_reads == set()


def test_load_sample_record_batches_cached_reuses_dicts_across_calls(tmp_path) -> None:
    """Regression test for the redundant-reload fix: a second call with the same
    cache_key must return the exact same in-memory dict objects (not just equal
    values) as the first call, and must not re-read the batch files from disk.
    """
    sequences = {
        "read1": np.array([1, 2, 3], dtype=np.int16),
        "read2": np.array([4, 5, 6], dtype=np.int16),
    }
    sequence_files = _write_integer_batches(
        sequences, tmp_path, record="chr1", prefix="fwd", batch_size=1
    )

    cache: dict[str, tuple] = {}
    result_a = _load_sample_record_batches_cached(
        cache, "0_chr1", sequence_files, [], [], []
    )
    result_b = _load_sample_record_batches_cached(
        cache, "0_chr1", sequence_files, [], [], []
    )

    # Same object identity, not just equal content -- proves the second call
    # was served entirely from cache without touching disk again.
    assert result_a[0] is result_b[0]
    assert result_a[1] is result_b[1]
    assert result_a[2] is result_b[2]
    assert result_a[0]["read1"].tolist() == [1, 2, 3]

    # A different cache_key must not reuse the first entry.
    result_c = _load_sample_record_batches_cached(
        cache, "1_chr1", sequence_files, [], [], []
    )
    assert result_c[0] is not result_a[0]
    assert result_c[0]["read1"].tolist() == [1, 2, 3]
