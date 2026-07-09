from smftools.informatics.modkit_extract_to_adata import (
    _count_active_mod_dict_types,
    _estimate_max_workers,
    _estimate_worker_peak_bytes,
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
    # With the default keys_per_batch=1, only the single largest key should be
    # used -- should not crash, and should be bounded by CPU count when memory
    # is ample.
    estimate = _estimate_max_workers(
        sequence_batch_files, empty, empty, empty, threads=8
    )
    assert 1 <= estimate <= 8


def test_estimate_sums_across_dict_types_for_same_key(tmp_path) -> None:
    # A worker loads sequence + mismatch + quality + read-span together for the
    # same (bam_index, record) key within one batch (_load_sample_record_batches_cached),
    # so estimating from all four dict types populated for one key must never
    # permit *more* workers than estimating from just one of them alone.
    files = []
    for name in ("a", "b", "c", "d"):
        f = tmp_path / f"{name}.h5ad"
        f.write_bytes(b"0" * (50 * 1024 * 1024))
        files.append(f)

    one_dict_only = _estimate_max_workers(
        {"0_record": [str(files[0])]}, {}, {}, {}, threads=32
    )
    all_four_dicts = _estimate_max_workers(
        {"0_record": [str(files[0])]},
        {"0_record": [str(files[1])]},
        {"0_record": [str(files[2])]},
        {"0_record": [str(files[3])]},
        threads=32,
    )
    assert all_four_dicts <= one_dict_only


def test_estimate_sums_top_keys_per_batch_keys(tmp_path) -> None:
    # sample_record_batch_cache is reset only at batch boundaries, so up to
    # keys_per_batch distinct keys can be resident in one worker at once --
    # asking for a larger keys_per_batch must never permit more workers.
    small = tmp_path / "small.h5ad"
    small.write_bytes(b"0" * (50 * 1024 * 1024))
    large = tmp_path / "large.h5ad"
    large.write_bytes(b"0" * (50 * 1024 * 1024))
    sequence_batch_files = {
        "0_record": [str(small)],
        "1_record": [str(large)],
    }
    empty: dict = {}
    single_key = _estimate_max_workers(
        sequence_batch_files, empty, empty, empty, threads=32, keys_per_batch=1
    )
    two_keys = _estimate_max_workers(
        sequence_batch_files, empty, empty, empty, threads=32, keys_per_batch=2
    )
    assert two_keys <= single_key


def test_estimate_includes_mod_dict_memory_term() -> None:
    # The modkit-extract TSV / per-read modification-dict arrays are built in
    # the same worker as (and independently of) the on-disk batch files, and
    # weren't reflected in the estimate at all before -- supplying that
    # information must never permit more workers than ignoring it entirely.
    empty: dict = {}
    without_mod_term = _estimate_max_workers(empty, empty, empty, empty, threads=32)
    with_mod_term = _estimate_max_workers(
        empty,
        empty,
        empty,
        empty,
        threads=32,
        reads_per_batch=50_000,
        max_reference_length=6_600,
        n_mod_dict_types=8,
    )
    assert with_mod_term <= without_mod_term


def test_count_active_mod_dict_types() -> None:
    assert _count_active_mod_dict_types([]) == 0
    assert _count_active_mod_dict_types(["6mA"]) == 3
    assert _count_active_mod_dict_types(["5mC"]) == 3
    assert _count_active_mod_dict_types(["6mA", "5mC"]) == 8


def test_overall_safety_multiplier_scales_the_estimate() -> None:
    # Regression coverage for a real production incident: after the per-batch
    # scoping fix made both components of the estimate accurate for the first
    # time, every worker still landed 3-16% over an unscaled estimate and got
    # killed by the memory watchdog -- harmless in absolute terms, but a 100%
    # batch failure rate. overall_safety_multiplier exists to absorb exactly
    # that kind of real-world slop a purely analytical model can't predict.
    kwargs = dict(
        # Large enough that the computed mod-dict term clearly exceeds
        # min_worker_budget_gb regardless of mod_array_dtype_bytes, so these
        # tests aren't sensitive to the exact per-value byte size used.
        reads_per_batch=50_096,
        max_reference_length=6_616,
        n_mod_dict_types=8,
    )
    empty: dict = {}
    baseline = _estimate_worker_peak_bytes(
        empty, empty, empty, empty, overall_safety_multiplier=1.0, **kwargs
    )
    scaled = _estimate_worker_peak_bytes(
        empty, empty, empty, empty, overall_safety_multiplier=1.25, **kwargs
    )
    assert scaled == round(baseline * 1.25)


def test_default_overall_safety_multiplier_is_greater_than_one() -> None:
    # The default must add real margin -- 1.0 (a no-op) would silently regress
    # back to the exact incident above.
    empty: dict = {}
    with_default = _estimate_worker_peak_bytes(
        empty,
        empty,
        empty,
        empty,
        # Large enough that the computed mod-dict term clearly exceeds
        # min_worker_budget_gb regardless of mod_array_dtype_bytes, so these
        # tests aren't sensitive to the exact per-value byte size used.
        reads_per_batch=50_096,
        max_reference_length=6_616,
        n_mod_dict_types=8,
    )
    without_margin = _estimate_worker_peak_bytes(
        empty,
        empty,
        empty,
        empty,
        # Large enough that the computed mod-dict term clearly exceeds
        # min_worker_budget_gb regardless of mod_array_dtype_bytes, so these
        # tests aren't sensitive to the exact per-value byte size used.
        reads_per_batch=50_096,
        max_reference_length=6_616,
        n_mod_dict_types=8,
        overall_safety_multiplier=1.0,
    )
    assert with_default > without_margin
