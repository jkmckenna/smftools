from __future__ import annotations

import json
from types import SimpleNamespace

from smftools.perf_log import (
    PerfLogger,
    get_perf_logger,
    set_perf_logger,
    summarize_perf_logs,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_perf_logger_writes_events_and_summary(tmp_path):
    path = tmp_path / "logs" / "260719_120000_perf.jsonl"
    logger = PerfLogger(path, "preprocess", sample_interval_seconds=2.0)
    logger.resource_envelope(resolved_threads=4, resolved_memory_bytes=8 * 1024**3)
    pool_id = logger.next_pool_id()
    logger.pool_start(pool_id, n_tasks=8, max_workers=4, by_memory_workers=150)
    logger.sample(pool_id, tree_rss_gb=3.5, n_live_workers=4)
    logger.sample(pool_id, tree_rss_gb=5.25, n_live_workers=4)
    logger.pool_retry(pool_id, reason="broken_pool", n_pending=6, new_max_workers=2)
    logger.pool_end(pool_id, final_max_workers=2, n_retries=1, peak_tree_rss_gb=5.25)
    logger.close()

    records = _read_jsonl(path)
    events = [r["event"] for r in records]
    assert events == [
        "resource_envelope",
        "pool_start",
        "sample",
        "sample",
        "pool_retry",
        "pool_end",
        "stage_summary",
    ]
    assert all(r["stage"] == "preprocess" for r in records)
    assert records[0]["resolved_threads"] == 4
    start = records[1]
    assert start["n_tasks"] == 8 and start["max_workers"] == 4 and start["by_memory_workers"] == 150
    summary = records[-1]
    # Peak is the max tree_rss seen across samples/pool_end; retries counted.
    assert summary["peak_tree_rss_gb"] == 5.25
    assert summary["n_retries_total"] == 1
    assert summary["max_workers_used"] == 4
    assert summary["n_pools"] == 1


def test_perf_logger_close_is_idempotent_and_silences_after(tmp_path):
    path = tmp_path / "logs" / "p_perf.jsonl"
    logger = PerfLogger(path, "spatial")
    logger.close()
    logger.close()  # must not raise
    logger.sample(1, tree_rss_gb=9.0)  # after close: dropped, not an error
    records = _read_jsonl(path)
    assert [r["event"] for r in records] == ["stage_summary"]


def test_summarize_perf_logs_rolls_up_multiple_files(tmp_path):
    # Two stages under one run tree; summary is one row per file, peak-desc.
    for stage, stem, peak in [("preprocess", "a", 5.25), ("hmm", "b", 1.5)]:
        logger = PerfLogger(tmp_path / stage / "logs" / f"{stem}_perf.jsonl", stage)
        pool_id = logger.next_pool_id()
        logger.pool_start(pool_id, n_tasks=3, max_workers=6)
        logger.sample(pool_id, tree_rss_gb=peak, n_live_workers=6)
        logger.pool_end(pool_id, final_max_workers=6, n_retries=0, peak_tree_rss_gb=peak)
        logger.close()

    summary = summarize_perf_logs(tmp_path)
    assert [row["stage"] for row in summary] == ["preprocess", "hmm"]  # peak-desc
    assert summary[0]["peak_tree_rss_gb"] == 5.25
    assert summary[0]["max_workers_used"] == 6
    assert summary[0]["n_samples"] == 1


def test_get_set_perf_logger_context(tmp_path):
    assert get_perf_logger() is None
    logger = PerfLogger(tmp_path / "x_perf.jsonl", "raw")
    set_perf_logger(logger)
    try:
        assert get_perf_logger() is logger
    finally:
        set_perf_logger(None)
        logger.close()
    assert get_perf_logger() is None


def test_stage_resource_envelope_is_written_to_perf_log(tmp_path):
    from smftools.logging_utils import _log_resource_envelope

    path = tmp_path / "resource_perf.jsonl"
    logger = PerfLogger(path, "raw")
    set_perf_logger(logger)
    envelope = SimpleNamespace(
        resolved_threads=3,
        resolved_memory_bytes=6 * 1024**3,
        enforcement_mode="worker_watchdog",
        enforcement_active=False,
        as_dict=lambda: {
            "resolved_threads": 3,
            "resolved_memory_bytes": 6 * 1024**3,
            "enforcement_mode": "worker_watchdog",
            "enforcement_active": False,
        },
    )
    try:
        _log_resource_envelope(SimpleNamespace(_resource_envelope=envelope))
    finally:
        logger.close()
        set_perf_logger(None)

    record = _read_jsonl(path)[0]
    assert record["event"] == "resource_envelope"
    assert record["resolved_threads"] == 3
