from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from smftools.memory_guard import process_tree_rss_bytes

pytestmark = pytest.mark.integration


def test_process_tree_rss_includes_a_live_child():
    baseline = process_tree_rss_bytes()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "data = bytearray(16 * 1024 * 1024); print('ready', flush=True); import time; time.sleep(10)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        try:
            descendant_pids = {process.pid for process in psutil.Process().children(recursive=True)}
        except (psutil.AccessDenied, psutil.Error, OSError) as error:
            pytest.skip(f"descendant enumeration is restricted: {error}")
        if child.pid not in descendant_pids:
            pytest.skip("descendant enumeration is restricted by the runtime sandbox")
        assert process_tree_rss_bytes() >= baseline + 8 * 1024**2
    finally:
        child.terminate()
        child.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux cgroup runtime check")
def test_linux_runtime_reports_real_cgroup_activation_or_fallback(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        """
import json
from types import SimpleNamespace
from smftools.memory_guard import activate_resource_envelope, resolve_resource_envelope

cfg = SimpleNamespace(
    threads=2,
    max_memory_percent=None,
    max_memory_gb=2.0,
    memory_reserve_gb=0.0,
    target_task_memory_mb=128,
)
active = activate_resource_envelope(resolve_resource_envelope(cfg, environ={}))
print(json.dumps({
    "active": active.enforcement_active,
    "capability": active.enforcement_capability,
    "mode": active.enforcement_mode,
}))
""".strip(),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)

    assert record["mode"] in {"cgroup_v2", "advisory"}
    assert record["active"] is (record["mode"] == "cgroup_v2")
    if record["active"]:
        assert record["capability"] == "cgroup_v2"
