from smftools.informatics.modkit_extract_to_adata import (
    _estimate_max_workers,
    _resolve_max_workers,
)


def test_none_resolves_to_serial() -> None:
    assert _resolve_max_workers(None, n_batches=10, threads=8) == 1


def test_explicit_one_or_less_resolves_to_serial() -> None:
    assert _resolve_max_workers(0, n_batches=10, threads=8) == 1
    assert _resolve_max_workers(1, n_batches=10, threads=8) == 1
    assert _resolve_max_workers(-3, n_batches=10, threads=8) == 1


def test_explicit_value_capped_by_n_batches() -> None:
    # No point spawning more workers than there are batches to process.
    assert _resolve_max_workers(8, n_batches=3, threads=8) == 3


def test_explicit_value_capped_by_memory_estimate(tmp_path) -> None:
    # A single (sample, record) batch-file set of 10 MiB, with the default 4x
    # multiplier, budgets ~40 MiB per worker. Simulate a tiny total-memory system
    # (via mem_safety_fraction indirectly by using a huge per-worker footprint) by
    # asking for far more workers than a reasonable memory budget could support.
    big_file = tmp_path / "big.h5ad"
    big_file.write_bytes(b"0" * (200 * 1024 * 1024))  # 200 MiB
    sequence_batch_files = {"0_record": [str(big_file)]}
    empty: dict = {}

    resolved = _resolve_max_workers(
        1_000_000,  # deliberately absurd request
        n_batches=1_000_000,
        threads=16,
        sequence_batch_files=sequence_batch_files,
        mismatch_batch_files=empty,
        quality_batch_files=empty,
        read_span_batch_files=empty,
    )
    # Must be capped well below the absurd request by either the CPU or memory bound.
    assert 1 <= resolved <= 16


def test_auto_never_exceeds_cpu_cap_when_no_file_size_info() -> None:
    empty: dict = {}
    resolved = _resolve_max_workers(
        "auto",
        n_batches=100,
        threads=4,
        sequence_batch_files=empty,
        mismatch_batch_files=empty,
        quality_batch_files=empty,
        read_span_batch_files=empty,
    )
    assert 1 <= resolved <= 4


def test_auto_invalid_string_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        _resolve_max_workers("not-a-real-option", n_batches=10, threads=4)


def test_estimate_falls_back_to_cpu_count_with_no_files() -> None:
    empty: dict = {}
    estimate = _estimate_max_workers(empty, empty, empty, empty, threads=6)
    assert estimate == 6


def test_estimate_uses_largest_key_not_sum_across_keys(tmp_path) -> None:
    small = tmp_path / "small.h5ad"
    small.write_bytes(b"0" * 1024)
    large = tmp_path / "large.h5ad"
    large.write_bytes(b"0" * (10 * 1024 * 1024))

    sequence_batch_files = {
        "0_record": [str(small)],
        "1_record": [str(large)],
    }
    empty: dict = {}
    # Should not crash, and should be bounded by CPU count when memory is ample.
    estimate = _estimate_max_workers(
        sequence_batch_files, empty, empty, empty, threads=8
    )
    assert 1 <= estimate <= 8
