import numpy as np

from smftools.informatics.modkit_extract_to_adata import _normalize_sequence_batch_files


def test_normalize_sequence_batch_files_handles_scalars() -> None:
    result = _normalize_sequence_batch_files("foo/bar.h5ad")
    assert [path.as_posix() for path in result] == ["foo/bar.h5ad"]


def test_normalize_sequence_batch_files_filters_empty_values() -> None:
    result = _normalize_sequence_batch_files(["", ".", None, "foo.h5ad"])
    assert [path.as_posix() for path in result] == ["foo.h5ad"]


def test_normalize_sequence_batch_files_handles_numpy_arrays() -> None:
    result = _normalize_sequence_batch_files(np.array(["foo.h5ad", "bar.h5ad"]))
    assert [path.as_posix() for path in result] == ["foo.h5ad", "bar.h5ad"]
