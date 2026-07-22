from __future__ import annotations

import sys
from types import SimpleNamespace

from smftools.memory_guard import (
    activate_resource_envelope,
    cgroup_cpu_quota_count,
    cgroup_memory_values,
    resolve_resource_envelope,
    scheduler_cpu_count,
    scheduler_memory_limit_bytes,
)


def _cfg(**overrides):
    values = {
        "threads": 32,
        "max_memory_percent": 60.0,
        "max_memory_gb": None,
        "memory_reserve_gb": 2.0,
        "target_task_memory_mb": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_machine(monkeypatch) -> None:
    gib = 1024**3
    monkeypatch.setattr("smftools.memory_guard.os.cpu_count", lambda: 64)
    monkeypatch.setattr("smftools.memory_guard.affinity_cpu_count", lambda: 16)
    monkeypatch.setattr("smftools.memory_guard.cgroup_cpu_quota_count", lambda: 8)
    monkeypatch.setattr("smftools.memory_guard.total_system_memory_bytes", lambda: 128 * gib)
    monkeypatch.setattr("smftools.memory_guard.available_system_memory_bytes", lambda: 24 * gib)
    monkeypatch.setattr("smftools.memory_guard.cgroup_memory_values", lambda: (16 * gib, 4 * gib))
    monkeypatch.setattr("smftools.memory_guard._cgroup_v2_self_path", lambda: None)


def test_resource_envelope_caps_cpu_by_scheduler_affinity_and_cgroup(monkeypatch):
    _patch_machine(monkeypatch)

    envelope = resolve_resource_envelope(
        _cfg(),
        environ={"SLURM_CPUS_PER_TASK": "4"},
    )

    assert envelope.requested_threads == 32
    assert envelope.logical_cpu_count == 64
    assert envelope.affinity_cpu_count == 16
    assert envelope.cgroup_cpu_count == 8
    assert envelope.scheduler_cpu_count == 4
    assert envelope.resolved_threads == 4


def test_resource_envelope_caps_memory_by_detected_headroom(monkeypatch):
    _patch_machine(monkeypatch)
    gib = 1024**3

    envelope = resolve_resource_envelope(
        _cfg(),
        environ={"SLURM_MEM_PER_NODE": "20G"},
    )

    # Available headroom: 22 GiB; cgroup: 16 - 4 - 2 = 10 GiB;
    # scheduler: 20 - 2 = 18 GiB. The cgroup headroom is restrictive.
    assert envelope.requested_memory_bytes == int(128 * gib * 0.6)
    assert envelope.resolved_memory_bytes == 10 * gib
    assert envelope.memory_reserve_bytes == 2 * gib


def test_resource_envelope_keeps_tighter_user_memory_cap(monkeypatch):
    _patch_machine(monkeypatch)
    gib = 1024**3

    envelope = resolve_resource_envelope(
        _cfg(max_memory_percent=None, max_memory_gb=6.0),
        environ={},
    )

    assert envelope.requested_memory_bytes == 6 * gib
    assert envelope.resolved_memory_bytes == 6 * gib


def test_cgroup_v2_cpu_and_memory_parsing(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text(str(12 * 1024**3), encoding="utf-8")
    (tmp_path / "memory.current").write_text(str(3 * 1024**3), encoding="utf-8")

    assert cgroup_cpu_quota_count(tmp_path) == 2
    assert cgroup_memory_values(tmp_path) == (12 * 1024**3, 3 * 1024**3)

    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    assert cgroup_cpu_quota_count(tmp_path) is None
    assert cgroup_memory_values(tmp_path) == (None, 3 * 1024**3)


def test_scheduler_resource_parsing_uses_most_restrictive_values():
    env = {
        "SLURM_CPUS_PER_TASK": "12",
        "SLURM_CPUS_ON_NODE": "16",
        "PBS_NP": "8",
        "SLURM_MEM_PER_NODE": "64G",
        "SLURM_MEM_PER_CPU": "4G",
    }

    assert scheduler_cpu_count(env) == 8
    assert scheduler_memory_limit_bytes(env, allocated_cpus=8) == 32 * 1024**3


def test_activation_records_cgroup_success_or_explicit_fallback(monkeypatch):
    _patch_machine(monkeypatch)
    envelope = resolve_resource_envelope(_cfg(), environ={})

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("smftools.memory_guard.enable_aggregate_memory_cap", lambda **kwargs: True)
    active = activate_resource_envelope(envelope)
    assert active.enforcement_mode == "cgroup_v2"
    assert active.enforcement_active is True

    monkeypatch.setattr("smftools.memory_guard.enable_aggregate_memory_cap", lambda **kwargs: False)
    fallback = activate_resource_envelope(envelope)
    assert fallback.enforcement_mode == "advisory"
    assert fallback.enforcement_active is False

    monkeypatch.setattr(sys, "platform", "win32")
    watchdog = activate_resource_envelope(envelope)
    assert watchdog.enforcement_mode == "worker_watchdog"
    assert watchdog.enforcement_active is False


def test_resolve_n_jobs_honors_detected_cpu_ceiling(monkeypatch):
    from smftools.parallel_utils import resolve_n_jobs

    monkeypatch.setattr("smftools.memory_guard.detected_usable_cpu_count", lambda: 3)

    assert resolve_n_jobs(12) == 3
    assert resolve_n_jobs(-1) == 3
    assert resolve_n_jobs(2) == 2
