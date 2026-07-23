"""Sequential multi-experiment dispatch with scheduler-visible results."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..constants import HMM_DIR, PREPROCESS_DIR, RAW_DIR, SPATIAL_DIR
from ..readwrite import atomic_write_json
from .helpers import load_experiment_config

BATCH_SUMMARY_SCHEMA_VERSION = 1
STAGE_DIRECTORIES = {
    "raw": RAW_DIR,
    "preprocess": PREPROCESS_DIR,
    "spatial": SPATIAL_DIR,
    "hmm": HMM_DIR,
}


def _latest_stage_outcome(run_root: Path, task: str) -> str | None:
    stage_directory = STAGE_DIRECTORIES.get(task)
    if stage_directory is None:
        return None
    perf_paths = sorted((run_root / stage_directory / "logs").glob("*_perf.jsonl"))
    if not perf_paths:
        return None
    outcome = None
    try:
        with perf_paths[-1].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") == "stage_summary":
                    outcome = record.get("outcome")
    except (OSError, json.JSONDecodeError):
        return None
    return str(outcome) if outcome is not None else None


def _result_paths(cfg_path: Path, task: str) -> dict[str, object]:
    """Resolve best-effort output/log paths even after a failed command."""
    try:
        cfg = load_experiment_config(str(cfg_path))
        run_root = Path(cfg.output_directory)
    except Exception:
        return {
            "output_directory": None,
            "human_logs": [],
            "performance_logs": [],
            "stage_outcome": None,
        }
    return {
        "output_directory": str(run_root),
        "human_logs": [str(path) for path in sorted(run_root.rglob("*_log.log"))],
        "performance_logs": [str(path) for path in sorted(run_root.rglob("*_perf.jsonl"))],
        "stage_outcome": _latest_stage_outcome(run_root, task),
        "full_summary": (
            str(run_root / "full_summary.json")
            if (run_root / "full_summary.json").exists()
            else None
        ),
    }


def run_batch(
    task: str,
    config_paths: Sequence[Path],
    function: Callable[[str], object],
    *,
    config_table: Path,
    summary_path: Path,
    emit: Callable[[str], object],
) -> dict[str, object]:
    """Run every config, atomically write the summary, and return its payload."""
    results = []
    for index, cfg_path in enumerate(config_paths):
        started = time.monotonic()
        if not cfg_path.exists():
            emit(f"[{index + 1}/{len(config_paths)}] SKIP (missing): {cfg_path}")
            results.append(
                {
                    "index": index,
                    "config_path": str(cfg_path),
                    "task": task,
                    "status": "failed",
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "exception": {
                        "type": "FileNotFoundError",
                        "message": f"config path does not exist: {cfg_path}",
                    },
                    **_result_paths(cfg_path, task),
                }
            )
            continue

        emit(f"[{index + 1}/{len(config_paths)}] {task} → {cfg_path}")
        try:
            function(str(cfg_path))
        except Exception as exc:
            emit(f"  ERROR on {cfg_path}: {exc}")
            results.append(
                {
                    "index": index,
                    "config_path": str(cfg_path),
                    "task": task,
                    "status": "failed",
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                    **_result_paths(cfg_path, task),
                }
            )
            continue

        paths = _result_paths(cfg_path, task)
        stage_outcome = paths.pop("stage_outcome", None)
        status = (
            "skipped"
            if stage_outcome == "skipped"
            else "failed"
            if stage_outcome == "failed"
            else "completed"
        )
        results.append(
            {
                "index": index,
                "config_path": str(cfg_path),
                "task": task,
                "status": status,
                "duration_seconds": round(time.monotonic() - started, 3),
                "exception": (
                    {
                        "type": "StageOutcome",
                        "message": "stage reported a failed outcome without raising",
                    }
                    if status == "failed"
                    else None
                ),
                **paths,
            }
        )

    failed = sum(result["status"] == "failed" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    payload = {
        "schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "task": task,
        "config_table": str(config_table),
        "status": "partial_failure" if failed else "complete",
        "total": len(results),
        "completed": len(results) - failed - skipped,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }
    atomic_write_json(summary_path, payload)
    return payload
