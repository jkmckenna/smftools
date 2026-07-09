from __future__ import annotations

import pandas as pd

from smftools.informatics.modkit_extract_to_adata import (
    _filter_reads_by_names,
    _split_extract_tsv_by_read_filters,
)


def test_filter_reads_by_names_keeps_only_matching_keys() -> None:
    mapping = {"r1": 1, "r2": 2, "r3": 3}
    assert _filter_reads_by_names(mapping, {"r1", "r3"}) == {"r1": 1, "r3": 3}


def test_filter_reads_by_names_empty_filter_returns_empty() -> None:
    assert _filter_reads_by_names({"r1": 1}, set()) == {}


def test_split_extract_tsv_by_read_filters_partitions_rows(tmp_path) -> None:
    tsv_path = tmp_path / "extract.tsv.gz"
    df = pd.DataFrame(
        {
            "read_id": ["r0", "r1", "r2", "r3"],
            "value": [10, 11, 12, 13],
        }
    )
    df.to_csv(tsv_path, sep="\t", header=True, index=False, compression="gzip")

    read_name_filters = {0: {"r0", "r1"}, 1: {"r2", "r3"}}
    chunk_paths = _split_extract_tsv_by_read_filters(tsv_path, read_name_filters, tmp_path)

    assert set(chunk_paths) == {0, 1}
    chunk0 = pd.read_csv(chunk_paths[0], sep="\t")
    chunk1 = pd.read_csv(chunk_paths[1], sep="\t")
    assert sorted(chunk0["read_id"]) == ["r0", "r1"]
    assert sorted(chunk1["read_id"]) == ["r2", "r3"]
    # Every source row lands in exactly one chunk -- no duplication, no drops.
    assert len(chunk0) + len(chunk1) == len(df)


def test_split_extract_tsv_by_read_filters_is_resumable(tmp_path, monkeypatch) -> None:
    tsv_path = tmp_path / "extract.tsv.gz"
    df = pd.DataFrame({"read_id": ["r0", "r1"], "value": [1, 2]})
    df.to_csv(tsv_path, sep="\t", header=True, index=False, compression="gzip")

    read_name_filters = {0: {"r0"}, 1: {"r1"}}
    first = _split_extract_tsv_by_read_filters(tsv_path, read_name_filters, tmp_path)
    mtimes_before = {idx: p.stat().st_mtime_ns for idx, p in first.items()}

    # A second call with the same already-written chunk files must not re-read the
    # source TSV at all -- this is what makes it safe on a killed/retried run
    # without re-paying the full-dataset parse cost.
    read_csv_calls: list[object] = []
    real_read_csv = pd.read_csv

    def _spy_read_csv(*args, **kwargs):
        read_csv_calls.append((args, kwargs))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _spy_read_csv)
    second = _split_extract_tsv_by_read_filters(tsv_path, read_name_filters, tmp_path)

    assert read_csv_calls == []
    assert second == first
    mtimes_after = {idx: p.stat().st_mtime_ns for idx, p in second.items()}
    assert mtimes_after == mtimes_before
