"""Bounded retry before declaring a task unadmittable (`F26a`).

Admission was decided on a *single instantaneous sample* of usable headroom.
The caller treats the resulting error as fatal for that lane, so a momentary
dip silently removed an entire analysis from a published generation: on the
260820 run, deamination segmentation was refused at 60.6 MiB headroom and the
same machine had 53.9 GB free forty seconds later.

Waiting costs seconds against work measured in hours. These pin that it waits,
that it still gives up on a genuine shortage, and that it does not wait when
there is nothing to wait for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smftools import memory_guard
from smftools.memory_guard import PoolBudgetError, require_task_admission, wait_for_task_admission

pytestmark = pytest.mark.unit


def _budget(max_in_flight, headroom_mb=60.0, per_item_mb=125.0):
    return SimpleNamespace(
        max_in_flight=max_in_flight,
        n_items=10,
        per_item_memory_bytes=per_item_mb * 1024**2,
        usable_headroom_bytes=headroom_mb * 1024**2,
    )


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(memory_guard.time, "sleep", lambda s: slept.append(s))
    return slept


def test_admits_immediately_when_headroom_is_available(monkeypatch, no_sleep):
    monkeypatch.setattr(memory_guard, "resolve_pool_budget", lambda *a, **k: _budget(4))

    budget = wait_for_task_admission(SimpleNamespace(), 10)

    assert budget.max_in_flight == 4
    assert no_sleep == [], "must not wait when the first sample already fits"


def test_waits_then_admits_when_headroom_recovers(monkeypatch, no_sleep):
    """The motivating case: a transient dip, not a real shortage."""
    calls = {"n": 0}

    def resolve(*a, **k):
        calls["n"] += 1
        return _budget(0) if calls["n"] < 3 else _budget(2)

    monkeypatch.setattr(memory_guard, "resolve_pool_budget", resolve)

    budget = wait_for_task_admission(SimpleNamespace(), 10, pool_label="deamination segmentation")

    assert budget.max_in_flight == 2
    assert len(no_sleep) == 2, "should have waited twice before succeeding"


def test_gives_up_after_the_attempt_budget(monkeypatch, no_sleep):
    """A genuine shortage must still fail rather than hang forever."""
    monkeypatch.setattr(memory_guard, "resolve_pool_budget", lambda *a, **k: _budget(0))

    budget = wait_for_task_admission(SimpleNamespace(), 10, attempts=4)

    assert budget.max_in_flight == 0
    assert len(no_sleep) == 3, "attempts includes the initial sample"


def test_delay_backs_off(monkeypatch, no_sleep):
    """Polling every 2s for minutes would be its own problem."""
    monkeypatch.setattr(memory_guard, "resolve_pool_budget", lambda *a, **k: _budget(0))

    wait_for_task_admission(SimpleNamespace(), 10, attempts=5, initial_delay=2.0, max_delay=30.0)

    assert no_sleep == sorted(no_sleep), "delays must be non-decreasing"
    assert no_sleep[-1] > no_sleep[0]


def test_delay_is_capped(monkeypatch, no_sleep):
    monkeypatch.setattr(memory_guard, "resolve_pool_budget", lambda *a, **k: _budget(0))

    wait_for_task_admission(SimpleNamespace(), 10, attempts=10, initial_delay=2.0, max_delay=8.0)

    assert max(no_sleep) <= 8.0


def test_a_single_attempt_does_not_wait(monkeypatch, no_sleep):
    """Callers that want the old behaviour can still have it."""
    monkeypatch.setattr(memory_guard, "resolve_pool_budget", lambda *a, **k: _budget(0))

    wait_for_task_admission(SimpleNamespace(), 10, attempts=1)

    assert no_sleep == []


# --- the caller's contract is unchanged --------------------------------------


def test_require_still_raises_on_an_unadmittable_budget():
    with pytest.raises(PoolBudgetError, match="Cannot admit one task"):
        require_task_admission(_budget(0), pool_label="deamination segmentation")


def test_require_accepts_an_admittable_budget():
    require_task_admission(_budget(1))


def test_the_pool_loop_waits_before_raising():
    """Pins the wiring: the retry must sit on the path that used to raise."""
    import inspect

    source = inspect.getsource(memory_guard.run_tasks_parallel)
    assert "wait_for_task_admission" in source
    index_wait = source.index("wait_for_task_admission")
    index_raise = source.index("require_task_admission", index_wait)
    assert index_wait < index_raise, "must poll before declaring the task unadmittable"
