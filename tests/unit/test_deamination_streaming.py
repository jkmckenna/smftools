"""Deamination results stream to disk instead of accumulating (`F45`)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.memory_guard import run_tasks_parallel
from smftools.preprocessing.partitioned_deamination import _StreamingParquetSink

pytestmark = pytest.mark.unit


def _identity(value):
    return value


def test_sink_writes_every_row_across_several_flushes(tmp_path):
    """Flushing must not lose or reorder rows."""
    sink = _StreamingParquetSink(tmp_path / "events.parquet", flush_rows=10)
    for start in range(0, 50, 5):
        sink.extend([{"read_id": f"r{i}", "position": i} for i in range(start, start + 5)])
    path = sink.close()

    frame = pd.read_parquet(path)
    assert sink.n_rows == 50
    assert list(frame["read_id"]) == [f"r{i}" for i in range(50)]


def test_sink_holds_no_rows_after_a_flush(tmp_path):
    """The point of the sink: rows leave memory as they are written.

    Accumulating them is what grew ~5.2 GiB/min with no plateau on a real run.
    """
    sink = _StreamingParquetSink(tmp_path / "events.parquet", flush_rows=10)
    sink.extend([{"read_id": f"r{i}"} for i in range(25)])

    assert len(sink._buffer) < 10
    sink.close()


def test_sink_keeps_the_first_schema_when_a_later_batch_is_all_null(tmp_path):
    """A batch whose column is entirely null must not change the column's type.

    Inferring per flush would let one batch decide a column is null-typed and
    make the file inconsistent with the rows already written.
    """
    sink = _StreamingParquetSink(tmp_path / "events.parquet", flush_rows=2)
    sink.extend([{"read_id": "r0", "score": 1.5}, {"read_id": "r1", "score": 2.5}])
    sink.extend([{"read_id": "r2", "score": None}, {"read_id": "r3", "score": None}])
    path = sink.close()

    frame = pd.read_parquet(path)
    assert len(frame) == 4
    assert str(frame["score"].dtype).startswith("float")
    assert frame["score"].isna().sum() == 2


def test_empty_sink_still_publishes_the_artifact(tmp_path):
    """Downstream expects the file to exist even when nothing was produced."""
    path = _StreamingParquetSink(tmp_path / "events.parquet").close()

    assert path.exists()
    assert pd.read_parquet(path).empty


def test_on_result_receives_results_and_they_are_not_retained():
    """`run_tasks_parallel` must hand results off rather than hold them.

    Streaming the arguments bounded only the input side; every completed result
    was still kept until the pool finished and then copied twice more (`F45`).
    """
    cfg = SimpleNamespace(
        threads=1,
        target_task_memory_mb=1,
        max_memory_gb=8,
        max_memory_percent=None,
        memory_reserve_gb=0.1,
        perf_log_sample_interval_seconds=2.0,
    )
    seen: list[tuple[int, int]] = []

    returned = run_tasks_parallel(
        _identity,
        [(1,), (2,), (3,)],
        cfg=cfg,
        on_result=lambda index, result: seen.append((index, result)),
    )

    assert sorted(seen) == [(0, 1), (1, 2), (2, 3)]
    assert returned == [], "results must not be retained when a sink consumes them"


def test_without_on_result_the_results_are_still_returned():
    """The sink is opt-in; existing callers are unchanged."""
    cfg = SimpleNamespace(
        threads=1,
        target_task_memory_mb=1,
        max_memory_gb=8,
        max_memory_percent=None,
        memory_reserve_gb=0.1,
        perf_log_sample_interval_seconds=2.0,
    )

    assert run_tasks_parallel(_identity, [(1,), (2,)], cfg=cfg) == [1, 2]
