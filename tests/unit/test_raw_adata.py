import gzip
from types import SimpleNamespace

import numpy as np
import pandas as pd
from click.testing import CliRunner

from smftools.cli.raw_adata import (
    _attach_direct_signals,
    _attach_direct_signals_from_bam,
    _bucket_read_ids,
    _ChromosomeGroupAccumulator,
    _conversion_signal,
    _map_references_parallel,
    _n_buckets_for_reference,
    _resolve_direct_call,
    _split_by_reference_strand,
    _split_modkit_tsv_by_bucket,
    _yield_flush_result,
)
from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT, PREPROCESS_DIR


def _double_record(record, *, multiplier):
    """Module-level (picklable) worker for exercising the process-pool path
    of ``_map_references_parallel`` -- a spawned worker process can only
    unpickle a top-level function, not a test-local closure.
    """
    return [record * multiplier], [str(record)]


def test_conversion_signal_matches_existing_binarization_maps():
    record = {
        "sequence": [
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["C"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["T"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["A"],
        ],
        "dataset": "5mC",
        "strand": "top",
        "Read_mismatch_trend": "C->T",
    }
    signal = _conversion_signal(record, deaminase=False)
    np.testing.assert_array_equal(signal, [1.0, 0.0, np.nan])

    record["sequence"] = [
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["G"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["A"],
    ]
    record["Read_mismatch_trend"] = "G->A"
    signal = _conversion_signal(record, deaminase=True)
    np.testing.assert_array_equal(signal, [0.0, 1.0])


def test_split_by_reference_strand_separates_mixed_deaminase_chromosome():
    # A single chromosome's extracted frame can mix "_top" and "_bottom" rows
    # for deaminase modality, since each read's own mismatch trend (not the
    # chromosome's canonical strand) decides its Reference_strand -- this is
    # the exact shape that broke streaming raw ingestion (only 4 of 8 expected
    # Reference_strand groups were produced, with the "bottom" reads silently
    # mislabeled under "_top") until callers were required to split before
    # yielding to the shard writer.
    frame = pd.DataFrame(
        {
            "read_id": ["r1", "r2", "r3", "r4"],
            "Reference_strand": ["chr1_top", "chr1_bottom", "chr1_top", "chr1_bottom"],
            "Read_mismatch_trend": ["C->T", "G->A", "C->T", "G->A"],
        }
    )

    groups = list(_split_by_reference_strand(frame))

    assert len(groups) == 2
    strand_values = {str(group["Reference_strand"].unique()[0]) for group in groups}
    assert strand_values == {"chr1_top", "chr1_bottom"}
    for group in groups:
        assert group["Reference_strand"].nunique() == 1
    total_rows = sum(len(group) for group in groups)
    assert total_rows == len(frame)
    recombined_read_ids = sorted(
        read_id for group in groups for read_id in group["read_id"].tolist()
    )
    assert recombined_read_ids == sorted(frame["read_id"].tolist())


def test_split_by_reference_strand_single_strand_frame_yields_one_group():
    frame = pd.DataFrame(
        {
            "read_id": ["r1", "r2"],
            "Reference_strand": ["chr2_top", "chr2_top"],
        }
    )

    groups = list(_split_by_reference_strand(frame))

    assert len(groups) == 1
    assert len(groups[0]) == 2


def _rows(*read_ids, reference_strand="chr1_top"):
    return pd.DataFrame({"read_id": list(read_ids), "Reference_strand": reference_strand})


def test_chromosome_group_accumulator_conversion_waits_for_all_conversion_state_records():
    # Regression test for a real data-loss bug: conversion modality aligns
    # each chromosome against multiple conversion-state variants (here,
    # "chr1_unconverted_top" and "chr1_5mC_top", matching a real
    # conversion_types=["5mC"] config), both belonging to chromosome "chr1".
    # Before this fix, completion was tracked per raw record, so each variant
    # independently yielded a "complete" group for the same Reference_strand
    # -- the streaming shard writer always starts a fresh write at
    # shard_index=0, so the second write silently overwrote the first's
    # shard file on disk while obs.parquet kept pointers for both (confirmed
    # on a real dataset: 3 of 4 references affected, up to 23% read loss).
    record_chromosome = {
        "chr1_unconverted_top": "chr1",
        "chr1_5mC_top": "chr1",
    }
    accumulator = _ChromosomeGroupAccumulator(record_chromosome)

    # First record completes -- must NOT yield yet, its chromosome sibling
    # ("chr1_5mC_top") hasn't completed.
    accumulator.add_partial("chr1_unconverted_top", [_rows("r1", "r2")])
    result = accumulator.complete("chr1_unconverted_top")
    assert result is None

    # Second (and last) record for "chr1" completes -- now the combined group
    # for the whole chromosome is returned, with reads from BOTH records.
    accumulator.add_partial("chr1_5mC_top", [_rows("r3", "r4", "r5")])
    result = accumulator.complete("chr1_5mC_top")
    assert result is not None
    frames, is_final = result
    assert is_final is True
    combined = pd.concat(frames, ignore_index=True)
    assert sorted(combined["read_id"]) == ["r1", "r2", "r3", "r4", "r5"]


def test_chromosome_group_accumulator_deaminase_yields_immediately_single_record():
    # Deaminase modality has exactly one reference_map record per chromosome
    # (its top/bottom split happens per-read from mismatch trend, not via
    # separate alignment targets) -- confirm this one-record-per-chromosome
    # shape still yields as soon as that record completes, unchanged from
    # before this fix (not blocked waiting on phantom siblings).
    record_chromosome = {"chr1": "chr1"}
    accumulator = _ChromosomeGroupAccumulator(record_chromosome)

    accumulator.add_partial("chr1", [_rows("r1", "r2", reference_strand="chr1_top")])
    result = accumulator.complete("chr1")

    assert result is not None
    frames, is_final = result
    assert is_final is True
    combined = pd.concat(frames, ignore_index=True)
    assert sorted(combined["read_id"]) == ["r1", "r2"]


def test_chromosome_group_accumulator_handles_zero_bucket_records():
    # A record with zero read_ids (e.g. "chr1_5mC_bottom" with no reads in a
    # real conversion dataset, per the debug trace that found this bug)
    # dispatches no work and must be marked complete with an empty frame list
    # up front, or its chromosome siblings would wait on it forever.
    record_chromosome = {
        "chr1_unconverted_top": "chr1",
        "chr1_5mC_top": "chr1",
        "chr1_5mC_bottom": "chr1",
    }
    accumulator = _ChromosomeGroupAccumulator(record_chromosome)

    assert accumulator.complete("chr1_5mC_bottom") is None
    accumulator.add_partial("chr1_unconverted_top", [_rows("r1")])
    assert accumulator.complete("chr1_unconverted_top") is None
    accumulator.add_partial("chr1_5mC_top", [_rows("r2")])
    result = accumulator.complete("chr1_5mC_top")

    assert result is not None
    frames, is_final = result
    assert is_final is True
    combined = pd.concat(frames, ignore_index=True)
    assert sorted(combined["read_id"]) == ["r1", "r2"]


def test_chromosome_group_accumulator_independent_chromosomes_dont_block_each_other():
    record_chromosome = {
        "chr1_unconverted_top": "chr1",
        "chr1_5mC_top": "chr1",
        "chr2_unconverted_top": "chr2",
    }
    accumulator = _ChromosomeGroupAccumulator(record_chromosome)

    # chr2 has only one contributing record -- completes independently of
    # chr1's still-outstanding second record.
    accumulator.add_partial("chr2_unconverted_top", [_rows("r9", reference_strand="chr2_top")])
    result = accumulator.complete("chr2_unconverted_top")
    assert result is not None
    frames, is_final = result
    assert is_final is True
    assert sorted(pd.concat(frames, ignore_index=True)["read_id"]) == ["r9"]

    accumulator.add_partial("chr1_unconverted_top", [_rows("r1")])
    assert accumulator.complete("chr1_unconverted_top") is None
    accumulator.add_partial("chr1_5mC_top", [_rows("r2")])
    result = accumulator.complete("chr1_5mC_top")
    assert result is not None
    frames, is_final = result
    assert is_final is True
    assert sorted(pd.concat(frames, ignore_index=True)["read_id"]) == ["r1", "r2"]


def test_chromosome_group_accumulator_flushes_early_past_threshold():
    # The actual memory fix: a single-record chromosome (the deaminase
    # shape) with enough accumulated rows must flush a bounded, non-final
    # batch before the record completes -- not wait and hand the whole
    # chromosome to the writer in one piece (a real 700k-read chromosome put
    # ~87GB in the parent process before this fix; see
    # dev/pipeline_scaling_audit.md).
    record_chromosome = {"chr1": "chr1"}
    accumulator = _ChromosomeGroupAccumulator(record_chromosome, flush_threshold=3)

    # Below threshold: nothing flushed yet.
    result = accumulator.add_partial("chr1", [_rows("r1", "r2", reference_strand="chr1_top")])
    assert result is None

    # Crosses threshold (2 + 2 = 4 >= 3): flushed now, marked NOT final --
    # the record (and its chromosome) isn't done yet.
    result = accumulator.add_partial("chr1", [_rows("r3", "r4", reference_strand="chr1_top")])
    assert result is not None
    frames, is_final = result
    assert is_final is False
    assert sorted(pd.concat(frames, ignore_index=True)["read_id"]) == ["r1", "r2", "r3", "r4"]

    # More data after a flush accumulates fresh, doesn't include what was
    # already flushed.
    result = accumulator.add_partial("chr1", [_rows("r5", reference_strand="chr1_top")])
    assert result is None

    # Record completes: whatever's left pending (just r5) is flushed final.
    result = accumulator.complete("chr1")
    assert result is not None
    frames, is_final = result
    assert is_final is True
    assert sorted(pd.concat(frames, ignore_index=True)["read_id"]) == ["r5"]


def test_yield_flush_result_marks_all_seen_strands_final_even_if_absent_from_last_batch():
    # A chromosome's rows can split unevenly across "_top"/"_bottom" flushes
    # -- e.g. all "_bottom" rows appear in an early partial flush, and the
    # chromosome's actual final flush only contains "_top" rows. Every
    # Reference_strand ever seen for that chromosome must still be signaled
    # final (frame=None marker for the ones absent from the last batch), or
    # a consumer waiting on that signal (e.g. plan_references) would never
    # finalize it.
    strands_seen: dict[str, set[str]] = {}

    # Partial (non-final) flush containing both strands.
    events = list(
        _yield_flush_result(
            (
                [
                    _rows("r1", reference_strand="chr1_top"),
                    _rows("r2", reference_strand="chr1_bottom"),
                ],
                False,
            ),
            "chr1",
            strands_seen,
        )
    )
    assert {(strand, is_final) for strand, _frame, is_final in events} == {
        ("chr1_bottom", False),
        ("chr1_top", False),
    }
    assert strands_seen["chr1"] == {"chr1_top", "chr1_bottom"}

    # Final flush containing only "_top" rows -- "_bottom" must still get a
    # frame=None, is_final=True marker.
    events = list(
        _yield_flush_result(
            ([_rows("r3", reference_strand="chr1_top")], True),
            "chr1",
            strands_seen,
        )
    )
    by_strand = {strand: (frame, is_final) for strand, frame, is_final in events}
    assert set(by_strand) == {"chr1_top", "chr1_bottom"}
    top_frame, top_final = by_strand["chr1_top"]
    assert top_frame is not None and list(top_frame["read_id"]) == ["r3"]
    assert top_final is True
    bottom_frame, bottom_final = by_strand["chr1_bottom"]
    assert bottom_frame is None
    assert bottom_final is True
    # Bookkeeping cleared once a chromosome is fully finalized.
    assert "chr1" not in strands_seen


def test_attach_direct_signals_converts_reference_to_query_coordinates(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "chr1",
                "reference_start": 5,
                "cigar": "2M1I2M",
            }
        ]
    )
    calls = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "ref_position": [5, 7],
            "modified_primary_base": ["A", "A"],
            "ref_strand": ["+", "+"],
            "read_id": ["read1", "read1"],
            "call_code": ["a", "-"],
            "call_prob": [0.8, 0.7],
        }
    )
    calls.to_csv(tmp_path / "calls.tsv", sep="\t", index=False)

    result, columns = _attach_direct_signals(frame, tmp_path)
    assert columns == ["modification_signal_A_plus"]
    np.testing.assert_allclose(
        result.at[0, "modification_signal"],
        [0.8, np.nan, np.nan, 0.3, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.at[0, "modification_signal_A_plus"],
        [0.8, np.nan, np.nan, 0.3, np.nan],
        equal_nan=True,
    )


def test_attach_direct_signals_reads_gzipped_modkit_extract(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "chr1",
                "reference_start": 5,
                "cigar": "2M1I2M",
            }
        ]
    )
    calls = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "ref_position": [5],
            "modified_primary_base": ["A"],
            "ref_strand": ["+"],
            "read_id": ["read1"],
            "call_code": ["a"],
            "call_prob": [0.8],
        }
    )
    with gzip.open(tmp_path / "calls.tsv.gz", "wt", encoding="utf-8") as handle:
        calls.to_csv(handle, sep="\t", index=False)

    result, columns = _attach_direct_signals(frame, tmp_path)
    assert columns == ["modification_signal_A_plus"]
    np.testing.assert_allclose(
        result.at[0, "modification_signal"],
        [0.8, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )


def test_split_modkit_tsv_by_bucket_routes_rows_by_read_id(tmp_path):
    calls = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "ref_position": [5, 6, 5, 6],
            "modified_primary_base": ["A", "A", "A", "A"],
            "ref_strand": ["+", "+", "+", "+"],
            "read_id": ["read1", "read1", "read2", "read2"],
            "call_code": ["a", "-", "-", "a"],
            "call_prob": [0.8, 0.7, 0.6, 0.9],
        }
    )
    tsv_path = tmp_path / "calls.tsv"
    calls.to_csv(tsv_path, sep="\t", index=False)
    output_dir = tmp_path / "split"

    paths = _split_modkit_tsv_by_bucket(
        [tsv_path], {"read1": 0, "read2": 1}, output_dir, chunksize=1
    )

    assert set(paths) == {0, 1}
    bucket0 = pd.read_csv(paths[0], sep="\t")
    bucket1 = pd.read_csv(paths[1], sep="\t")
    assert set(bucket0["read_id"]) == {"read1"}
    assert set(bucket1["read_id"]) == {"read2"}
    assert len(bucket0) == 2
    assert len(bucket1) == 2


def test_split_modkit_tsv_by_bucket_drops_unassigned_read_ids(tmp_path):
    calls = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "ref_position": [5, 5],
            "modified_primary_base": ["A", "A"],
            "ref_strand": ["+", "+"],
            "read_id": ["wanted", "not_wanted"],
            "call_code": ["a", "a"],
            "call_prob": [0.8, 0.8],
        }
    )
    tsv_path = tmp_path / "calls.tsv"
    calls.to_csv(tsv_path, sep="\t", index=False)
    output_dir = tmp_path / "split"

    paths = _split_modkit_tsv_by_bucket([tsv_path], {"wanted": 0}, output_dir)

    assert set(paths) == {0}
    bucket0 = pd.read_csv(paths[0], sep="\t")
    assert set(bucket0["read_id"]) == {"wanted"}


def test_map_references_parallel_sequential_matches_process_pool():
    items = [(1,), (2,), (3,), (4,)]

    sequential = list(
        _map_references_parallel(
            items, _double_record, max_workers=1, worker_kwargs={"multiplier": 10}
        )
    )
    parallel = list(
        _map_references_parallel(
            items, _double_record, max_workers=2, worker_kwargs={"multiplier": 10}
        )
    )

    # Parallel path yields as futures complete, not submission order -- compare
    # as sets/multisets, not positionally. Each entry is (args, (result, columns)).
    assert sorted(args for args, _result in sequential) == sorted(
        args for args, _result in parallel
    )
    assert {tuple(result) for _args, (result, _columns) in sequential} == {
        tuple(result) for _args, (result, _columns) in parallel
    }


def test_map_references_parallel_bounds_in_flight_submissions(monkeypatch):
    # Regression test for a real ~90GB parent-process memory blowup: the old
    # code submitted every item to the process pool upfront
    # (`{pool.submit(...): args for args in items}`), decoupling submission
    # from consumption -- with fast workers producing large per-bucket
    # results and a single-threaded consumer draining them, completed
    # results piled up in the executor's queue unbounded by max_workers (see
    # dev/pipeline_scaling_audit.md). Confirms at most max_workers items are
    # ever submitted before the first result is retrieved from the
    # generator, regardless of total item count.
    import concurrent.futures as cf

    import smftools.memory_guard as memory_guard_module

    submit_calls = []

    class _FakeExecutor:
        def __init__(self, max_workers=None, initializer=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def submit(self, fn, *args, **kwargs):
            submit_calls.append(args)
            future = cf.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    monkeypatch.setattr(cf, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        memory_guard_module, "start_worker_watchdog", lambda pool, budget, *a, **k: lambda: None
    )

    items = [(1,), (2,), (3,), (4,), (5,), (6,)]
    gen = _map_references_parallel(
        items, _double_record, max_workers=2, worker_kwargs={"multiplier": 10}
    )

    next(gen)
    assert len(submit_calls) == 2, (
        "only the initial max_workers window should be submitted before the "
        "first result is consumed"
    )

    next(gen)
    next(gen)
    assert len(submit_calls) == 4, (
        "each consumed result should trigger exactly one new submission, not "
        "a fresh burst of everything remaining"
    )

    # Draining the rest still yields every item exactly once.
    remaining = list(gen)
    assert len(remaining) == 3
    assert len(submit_calls) == 6


def test_bucket_read_ids_splits_evenly_regardless_of_clustering():
    # Reproduces the real-data shape that broke position-based windowing:
    # many reads share the exact same genomic position (PCR/library
    # duplication), so no position boundary can split them -- round-robin
    # over read identity doesn't care, and still balances exactly.
    read_ids = [f"read{i}" for i in range(1000)]

    buckets = _bucket_read_ids(read_ids, n_buckets=8)

    assert len(buckets) == 8
    counts = [len(bucket) for bucket in buckets]
    assert max(counts) - min(counts) <= 1
    # Every read assigned to exactly one bucket, none lost or duplicated.
    union = set().union(*buckets)
    assert union == set(read_ids)
    assert sum(counts) == len(read_ids)


def test_bucket_read_ids_no_reads_yields_no_buckets():
    assert _bucket_read_ids([], n_buckets=4) == []


def test_bucket_read_ids_single_bucket_covers_all_reads():
    read_ids = ["a", "b", "c"]
    assert _bucket_read_ids(read_ids, n_buckets=1) == [{"a", "b", "c"}]


def test_bucket_read_ids_more_buckets_than_reads_drops_empty_buckets():
    read_ids = ["a", "b"]
    buckets = _bucket_read_ids(read_ids, n_buckets=8)
    assert len(buckets) == 2
    assert set().union(*buckets) == {"a", "b"}


def test_n_buckets_for_reference_caps_at_max_workers_when_memory_allows():
    # Plenty of reads, few workers, but still under the memory ceiling
    # (100_000 / 4 = 25_000 <= a generous max_reads_per_bucket): capped by
    # max_workers, same as before the memory-safety fix.
    assert (
        _n_buckets_for_reference(
            100_000, max_workers=4, min_reads_per_bucket=500, max_reads_per_bucket=50_000
        )
        == 4
    )
    # Few reads: capped by min_reads_per_bucket, not max_workers.
    assert _n_buckets_for_reference(300, max_workers=8, min_reads_per_bucket=500) == 1


def test_n_buckets_for_reference_scales_past_max_workers_for_memory_safety():
    # Regression test: bucket count used to be capped at max_workers no
    # matter how large n_reads was, so bucket (and per-worker memory) size
    # scaled linearly with reference read count -- fine for a small
    # experiment, but large experiments blew past the per-worker memory
    # budget entirely (see dev/pipeline_scaling_audit.md). Bucket count must
    # now grow past max_workers once n_reads/max_workers would exceed
    # max_reads_per_bucket, keeping bucket size experiment-size-independent.
    n_buckets = _n_buckets_for_reference(
        700_000, max_workers=12, min_reads_per_bucket=500, max_reads_per_bucket=4_000
    )
    assert n_buckets == 175
    assert n_buckets > 12
    assert 700_000 // n_buckets <= 4_000


def test_n_buckets_for_reference_single_worker_still_respects_memory_ceiling():
    # A single-worker config used to always return 1 bucket regardless of
    # read count, putting an entire large reference's data in one process --
    # the memory ceiling must still apply even with no parallelism to gain.
    n_buckets = _n_buckets_for_reference(
        700_000, max_workers=1, min_reads_per_bucket=500, max_reads_per_bucket=4_000
    )
    assert n_buckets == 175


def test_resolve_direct_call_picks_highest_probability_state():
    # Canonical wins: modified probability (50/255 ~= 0.196) is lower than
    # canonical (1 - 0.196 ~= 0.804).
    code, probability = _resolve_direct_call({"m": 50})
    assert code == "-"
    np.testing.assert_allclose(probability, 1.0 - 50 / 255.0)

    # Modified wins: 200/255 ~= 0.784 beats canonical (1 - 0.784 ~= 0.216).
    code, probability = _resolve_direct_call({"m": 200})
    assert code == "m"
    np.testing.assert_allclose(probability, 200 / 255.0)

    # Multiple simultaneous codes at one position (e.g. 5mC and 5hmC on the
    # same C): highest of {canonical, m, h} wins.
    code, probability = _resolve_direct_call({"h": 2, "m": 230})
    assert code == "m"
    np.testing.assert_allclose(probability, 230 / 255.0)


def _write_direct_signal_test_bam(path, *, read_id="read1", reference="chr1"):
    """One 6bp forward-strand read ("ACGTAC") with MM/ML calls at both C's
    (query positions 1, 5) and both A's (query positions 0, 4), chosen so
    canonical wins at one position of each base and the modified call wins at
    the other -- exercises both branches of ``_resolve_direct_call`` through
    the real pysam MM/ML decode path, not just the pure function directly.
    """
    import pysam

    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": reference, "LN": 20}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = read_id
        read.query_sequence = "ACGTAC"
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 0
        read.mapping_quality = 60
        read.cigartuples = [(0, 6)]
        read.query_qualities = pysam.qualitystring_to_array("IIIIII")
        read.set_tag("MM", "C+m,0,0;A+a,0,0;", value_type="Z")
        read.set_tag("ML", [50, 200, 10, 250])
        bam.write(read)
    pysam.sort("-o", str(path.with_suffix(".sorted.bam")), str(path))
    pysam.index(str(path.with_suffix(".sorted.bam")))
    return path.with_suffix(".sorted.bam")


def test_attach_direct_signals_from_bam_matches_manual_ml_derivation(tmp_path):
    bam_path = _write_direct_signal_test_bam(tmp_path / "direct.bam")
    frame = pd.DataFrame(
        [
            {
                "read_id": "read1",
                "reference": "chr1",
                "reference_start": 0,
                "cigar": "6M",
            }
        ]
    )

    result, columns = _attach_direct_signals_from_bam(frame, bam_path)

    assert set(columns) == {"modification_signal_A_plus", "modification_signal_C_plus"}
    expected_combined = [
        10 / 255.0,  # pos 0 (A): canonical wins (96% conf.) -> P(modified) = 1 - 0.96
        50 / 255.0,  # pos 1 (C): canonical wins (80% conf.) -> P(modified) = 1 - 0.80
        np.nan,  # pos 2 (G): no MM/ML entry
        np.nan,  # pos 3 (T): no MM/ML entry
        250 / 255.0,  # pos 4 (A): modified wins -> P(modified) = winning confidence
        200 / 255.0,  # pos 5 (C): modified wins -> P(modified) = winning confidence
    ]
    np.testing.assert_allclose(
        result.at[0, "modification_signal"], expected_combined, equal_nan=True, rtol=1e-6
    )
    np.testing.assert_allclose(
        result.at[0, "modification_signal_A_plus"],
        [expected_combined[0], np.nan, np.nan, np.nan, expected_combined[4], np.nan],
        equal_nan=True,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.at[0, "modification_signal_C_plus"],
        [np.nan, expected_combined[1], np.nan, np.nan, np.nan, expected_combined[5]],
        equal_nan=True,
        rtol=1e-6,
    )


def _write_direct_signal_partial_calls_bam(path, *, read_id="read1", reference="chr1"):
    """One 6bp forward-strand read ("AAGTAC") with an explicit MM/ML call at
    the first and third A (query positions 0, 4) but none at the second
    (query position 1) -- exercises ``impute_uncalled_canonical``'s
    fill-missing behavior for a base that has some, but not all, of its
    occurrences explicitly called. Position 5 (C) has no MM/ML entries at
    all, exercising that imputation never invents a channel for a base with
    zero calls in the read.
    """
    import pysam

    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": reference, "LN": 20}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = read_id
        read.query_sequence = "AAGTAC"
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 0
        read.mapping_quality = 60
        read.cigartuples = [(0, 6)]
        read.query_qualities = pysam.qualitystring_to_array("IIIIII")
        read.set_tag("MM", "A+a,0,1;", value_type="Z")
        read.set_tag("ML", [200, 10])
        bam.write(read)
    pysam.sort("-o", str(path.with_suffix(".sorted.bam")), str(path))
    pysam.index(str(path.with_suffix(".sorted.bam")))
    return path.with_suffix(".sorted.bam")


def test_attach_direct_signals_from_bam_default_leaves_uncalled_bases_nan(tmp_path):
    bam_path = _write_direct_signal_partial_calls_bam(tmp_path / "partial.bam")
    frame = pd.DataFrame(
        [{"read_id": "read1", "reference": "chr1", "reference_start": 0, "cigar": "6M"}]
    )

    result, _ = _attach_direct_signals_from_bam(frame, bam_path)

    signal = result.at[0, "modification_signal"]
    assert np.isnan(signal[1])  # second A, no explicit call, default: NaN


def test_attach_direct_signals_from_bam_imputes_uncalled_canonical_when_enabled(tmp_path):
    bam_path = _write_direct_signal_partial_calls_bam(tmp_path / "partial.bam")
    frame = pd.DataFrame(
        [{"read_id": "read1", "reference": "chr1", "reference_start": 0, "cigar": "6M"}]
    )

    result, _ = _attach_direct_signals_from_bam(frame, bam_path, impute_uncalled_canonical=True)

    signal = result.at[0, "modification_signal"]
    assert signal[1] == 0.0  # second A, no explicit call, imputed as canonical
    np.testing.assert_allclose(signal[0], 200 / 255.0, rtol=1e-6)  # explicit call, modified wins
    np.testing.assert_allclose(signal[4], 10 / 255.0, rtol=1e-6)  # explicit call, canonical wins
    assert np.isnan(signal[2])  # non-A, untouched
    assert np.isnan(signal[3])  # non-A, untouched
    assert np.isnan(signal[5])  # C with zero calls in this read: no imputation


def test_artifact_paths_include_raw_and_dense_outputs(tmp_path):
    from smftools.cli.helpers import get_artifact_paths

    cfg = SimpleNamespace(
        output_directory=tmp_path,
        split_path=tmp_path / "split",
        bam_suffix=".bam",
        input_type="bam",
        input_data_path=tmp_path / "input.bam",
        smf_modality="conversion",
        experiment_name="experiment",
    )
    paths = get_artifact_paths(cfg)
    assert paths.raw_directory == tmp_path / "raw_outputs"
    assert paths.spine == paths.load_directory / "spine.h5ad"
    assert paths.dense_store == paths.load_directory / "store"
    assert paths.dense_catalog == paths.load_directory / "catalog.parquet"
    assert paths.sidecar_manifest.parent == paths.raw_directory


def test_cli_exposes_raw_and_optional_load_commands():
    from smftools.cli_entry import cli

    result = CliRunner().invoke(cli, ["experiment", "--help"])
    assert result.exit_code == 0
    assert "raw" in result.output
    assert "Optionally pre-build the dense zarr cache" in result.output


def test_raw_wrapper_stops_legacy_pipeline_before_dense_loading(tmp_path, monkeypatch):
    from smftools.cli import helpers
    from smftools.cli import load_adata as load_module
    from smftools.cli.raw_adata import raw_adata

    cfg = SimpleNamespace(output_directory=tmp_path, force_redo_load_adata=False)
    paths = SimpleNamespace(raw_spine=tmp_path / "raw_outputs" / "spine.h5ad")
    captured = {}

    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    def fake_core(core_cfg, core_paths, config_path=None, *, raw_only=False):
        captured.update(
            cfg=core_cfg,
            paths=core_paths,
            config_path=config_path,
            raw_only=raw_only,
        )
        return "spine", paths.raw_spine, core_cfg

    monkeypatch.setattr(load_module, "load_adata_core", fake_core)
    result = raw_adata("experiment.csv")

    assert result == ("spine", paths.raw_spine, cfg)
    assert captured["raw_only"] is True


def test_load_dense_cache_runs_raw_then_cache_builder(tmp_path, monkeypatch):
    from smftools import readwrite
    from smftools.cli import raw_adata as raw_module
    from smftools.cli.load_adata import load_dense_cache
    from smftools.informatics import partition_store

    cfg = SimpleNamespace(output_directory=tmp_path)
    source_spine = tmp_path / "spine.h5ad"
    monkeypatch.setattr(
        raw_module,
        "raw_adata",
        lambda _path: ("raw-spine", source_spine, cfg),
    )
    monkeypatch.setattr(
        partition_store,
        "write_dense_cache_from_spine",
        lambda path, output_dir=None: {"spine": path},
    )
    monkeypatch.setattr(readwrite, "safe_read_h5ad", lambda path: ("cached-spine", None))

    assert load_dense_cache("experiment.csv") == ("cached-spine", source_spine, cfg)


def test_get_adata_paths_includes_dense_cache_artifacts(tmp_path):
    from smftools.cli.helpers import get_adata_paths
    from smftools.constants import LOAD_DIR

    cfg = SimpleNamespace(
        output_directory=str(tmp_path),
        experiment_name="EXP",
        smf_modality="conversion",
    )
    p = get_adata_paths(cfg)
    load_dir = tmp_path / LOAD_DIR
    # These are the dense zarr cache paths write_dense_cache_from_spine writes
    # to via `smftools load` / load_dense_cache -- not written by the default
    # raw_adata()/full_flow path.
    assert p.store == load_dir / "store"
    assert p.spine == load_dir / "spine.h5ad"
    assert p.catalog == load_dir / "catalog.parquet"
    assert p.raw.parent.parent == load_dir


def test_preprocess_wrapper_accepts_raw_spine_source(tmp_path, monkeypatch):
    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module
    from smftools.informatics import partition_read

    raw_spine_path = tmp_path / "raw_outputs" / "spine.h5ad"
    raw_spine_path.parent.mkdir()
    raw_spine_path.touch()
    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "load_adata_outputs" / "spine.h5ad",
        raw_spine=raw_spine_path,
        pp=tmp_path / "pp.h5ad.gz",
        pp_dedup=tmp_path / "pp_dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
    )
    captured = {}
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)
    monkeypatch.setattr(partition_read, "materialize", lambda path: "materialized")

    def fake_core(**kwargs):
        captured.update(kwargs)
        return paths.pp, paths.pp_dedup

    monkeypatch.setattr(preprocess_module, "preprocess_adata_core", fake_core)
    result = preprocess_module.preprocess_adata("experiment.csv")

    assert result == (paths.pp, paths.pp_dedup)
    assert captured["adata"] == "materialized"
    assert captured["source_adata_path"] == raw_spine_path


def test_preprocess_wrapper_dispatches_planned_spine(tmp_path, monkeypatch):
    import anndata as ad

    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module
    from smftools.preprocessing import partitioned_executor

    raw_spine_path = tmp_path / "raw_outputs" / "spine.h5ad"
    raw_spine_path.parent.mkdir()
    spine = ad.AnnData(obs=pd.DataFrame(index=["read1"]))
    spine.uns["reference_plans"] = {"locus_top": {"analysis_mode": "locus"}}
    spine.write_h5ad(raw_spine_path)
    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "load_adata_outputs" / "spine.h5ad",
        raw_spine=raw_spine_path,
        pp=tmp_path / "pp.h5ad.gz",
        pp_dedup=tmp_path / "pp_dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        preprocess_execution_mode="auto",
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
    )
    output_spine = tmp_path / PREPROCESS_DIR / "spine.h5ad"
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)
    monkeypatch.setattr(
        partitioned_executor,
        "execute_partitioned_preprocessing",
        lambda source, config, output: {"spine": output_spine},
    )

    assert preprocess_module.preprocess_adata("experiment.csv") == (output_spine, None)


def test_preprocess_wrapper_returns_existing_partitioned_spine(tmp_path, monkeypatch):
    import anndata as ad

    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module
    from smftools.readwrite import safe_write_h5ad

    # A completed partitioned preprocess spine has passes_qc (and usually
    # passes_dedup) in obs -- the skip-if-exists check verifies this before
    # trusting a cached spine, since a spine.h5ad can exist on disk from a run
    # that crashed after writing it but before QC/dedup columns were merged in
    # (see reduce_duplicate_reads' staging-write fix). An empty placeholder file
    # no longer qualifies as "existing" for skip purposes.
    output_spine = tmp_path / PREPROCESS_DIR / "spine.h5ad"
    output_spine.parent.mkdir()
    safe_write_h5ad(
        ad.AnnData(obs=pd.DataFrame({"passes_qc": [True, False]}, index=["read1", "read2"])),
        output_spine,
        backup=False,
        verbose=False,
    )
    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "missing-load-spine.h5ad",
        raw_spine=tmp_path / "missing-raw-spine.h5ad",
        pp=tmp_path / "missing-pp.h5ad.gz",
        pp_dedup=tmp_path / "missing-dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    assert preprocess_module.preprocess_adata("experiment.csv") == (output_spine, None)


def test_preprocess_wrapper_reruns_when_partitioned_spine_missing_qc_dedup(tmp_path, monkeypatch):
    # Companion to test_preprocess_wrapper_returns_existing_partitioned_spine:
    # a spine.h5ad present on disk but lacking passes_qc/passes_dedup (a crash
    # inside reduce_duplicate_reads used to leave exactly this behind, in a real
    # production incident) must NOT be trusted as complete -- preprocessing
    # should re-run from the raw spine instead of silently skipping.
    import anndata as ad

    from smftools.cli import helpers
    from smftools.cli import preprocess_adata as preprocess_module
    from smftools.readwrite import safe_write_h5ad

    incomplete_spine = tmp_path / PREPROCESS_DIR / "spine.h5ad"
    incomplete_spine.parent.mkdir()
    safe_write_h5ad(
        ad.AnnData(obs=pd.DataFrame({"read_id": ["read1"]}, index=["read1"])),
        incomplete_spine,
        backup=False,
        verbose=False,
    )

    raw_spine = tmp_path / "raw-spine.h5ad"
    safe_write_h5ad(
        ad.AnnData(
            obs=pd.DataFrame({"read_id": ["read1"]}, index=["read1"]),
            uns={"reference_plans": {"ref": {"analysis_mode": "locus", "reference_length": 10}}},
        ),
        raw_spine,
        backup=False,
        verbose=False,
    )

    paths = SimpleNamespace(
        raw=tmp_path / "missing.h5ad.gz",
        spine=tmp_path / "missing-load-spine.h5ad",
        raw_spine=raw_spine,
        pp=tmp_path / "missing-pp.h5ad.gz",
        pp_dedup=tmp_path / "missing-dedup.h5ad.gz",
    )
    cfg = SimpleNamespace(
        output_directory=tmp_path,
        force_redo_preprocessing=False,
        force_redo_flag_duplicate_reads=False,
        preprocess_execution_mode="auto",
    )
    monkeypatch.setattr(helpers, "load_experiment_config", lambda _path: cfg)
    monkeypatch.setattr(helpers, "get_adata_paths", lambda _cfg: paths)

    captured = {}

    def fake_execute_partitioned_preprocessing(source_path, executor_cfg, output_dir):
        captured["source_path"] = source_path
        return {"spine": incomplete_spine}

    import smftools.preprocessing.partitioned_executor as partitioned_executor_module

    monkeypatch.setattr(
        partitioned_executor_module,
        "execute_partitioned_preprocessing",
        fake_execute_partitioned_preprocessing,
    )

    result = preprocess_module.preprocess_adata("experiment.csv")

    assert captured.get("source_path") == raw_spine
    assert result == (incomplete_spine, None)
