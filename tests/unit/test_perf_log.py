from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from smftools.perf_log import (
    PerfLogger,
    get_perf_logger,
    perf_substep,
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
    assert summary["outcome"] == "completed"
    assert "bytes_read" in summary
    assert "bytes_written" in summary


def test_perf_logger_emits_completion_progress_and_explicit_outcome(tmp_path):
    path = tmp_path / "raw" / "logs" / "raw_perf.jsonl"
    logger = PerfLogger(path, "raw", sample_interval_seconds=0.01)
    logger.task_complete(
        1,
        task_index=3,
        completed=1,
        total=2,
        duration_seconds=0.25,
        retry_count=1,
        rows=4,
        bases=40,
    )
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if any(record["event"] == "resource_sample" for record in _read_jsonl(path)):
            break
        time.sleep(0.01)
    logger.close(outcome="skipped", reason="already complete")

    records = _read_jsonl(path)
    progress = next(record for record in records if record["event"] == "task_progress")
    assert progress["completed"] == 1
    assert progress["total"] == 2
    assert progress["task_index"] == 3
    assert progress["retry_count"] == 1
    assert progress["rows"] == 4
    assert progress["bases"] == 40
    assert progress["eta_seconds"] is not None
    assert any(record["event"] == "resource_sample" for record in records)
    summary = records[-1]
    assert summary["outcome"] == "skipped"
    assert summary["reason"] == "already complete"
    assert summary["tasks_completed"] == 1


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
    assert summary[0]["tasks_total"] == 3
    assert summary[0]["outcome"] == "completed"


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


def test_stage_logging_lifecycle_closes_failed_stage(tmp_path):
    from smftools.logging_utils import setup_stage_logging, stage_logging_lifecycle

    @stage_logging_lifecycle
    def fail_stage():
        setup_stage_logging(SimpleNamespace(), tmp_path / "raw_outputs")
        raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        fail_stage()

    assert get_perf_logger() is None
    perf_path = next((tmp_path / "raw_outputs" / "logs").glob("*_perf.jsonl"))
    summary = _read_jsonl(perf_path)[-1]
    assert summary["event"] == "stage_summary"
    assert summary["outcome"] == "failed"
    assert summary["exception_type"] == "RuntimeError"


def test_perf_substep_records_completed_and_failed_parent_work(tmp_path):
    path = tmp_path / "spatial_perf.jsonl"
    logger = PerfLogger(path, "spatial")
    set_perf_logger(logger)
    try:
        with perf_substep("reduce_metrics", rows=10):
            pass
        with pytest.raises(ValueError, match="bad plot"):
            with perf_substep("render_plots"):
                raise ValueError("bad plot")
    finally:
        logger.close(outcome="failed")
        set_perf_logger(None)

    records = _read_jsonl(path)
    endings = [record for record in records if record["event"] == "substep_end"]
    assert [(record["name"], record["outcome"]) for record in endings] == [
        ("reduce_metrics", "completed"),
        ("render_plots", "failed"),
    ]
    assert endings[0]["rows"] == 10
    assert records[-1]["substeps_completed"] == 2
