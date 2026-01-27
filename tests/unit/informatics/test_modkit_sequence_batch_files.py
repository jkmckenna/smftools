import numpy as np

from smftools.informatics.modkit_extract_to_adata import (
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
