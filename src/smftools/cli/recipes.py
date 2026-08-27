from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..constants import BASECALL_DIR, HMM_DIR, LATENT_DIR, PREPROCESS_DIR, RAW_DIR, SPATIAL_DIR

FULL_SUMMARY_FILENAME = "full_summary.json"
FULL_SUMMARY_SCHEMA_VERSION = 2
FULL_STAGE_DIRECTORIES = {
    "basecall": BASECALL_DIR,
    "raw": RAW_DIR,
    "preprocess": PREPROCESS_DIR,
    "spatial": SPATIAL_DIR,
    "hmm": HMM_DIR,
    "latent": LATENT_DIR,
}


def _relative_log(run_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _latest_log(directory: Path, pattern: str) -> Path | None:
    paths = list(directory.glob(pattern))
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def _perf_outcome(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        outcome = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") == "stage_summary":
                    outcome = record.get("outcome")
    except (OSError, json.JSONDecodeError):
        return None
    return str(outcome) if outcome is not None else None


def _write_full_summary(cfg, *, outcome: str, error: BaseException | None = None) -> Path:
    """Write the relocatable top-level index for child stage logs and outcomes."""
    from ..informatics.experiment_manifest import read_experiment_manifest
    from ..readwrite import atomic_write_json

    run_root = Path(cfg.output_directory)
    manifest_stages = read_experiment_manifest(run_root).get("stages", {})
    stages = []
    for stage, directory_name in FULL_STAGE_DIRECTORIES.items():
        logs = run_root / directory_name / "logs"
        human_log = _latest_log(logs, "*_log.log")
        perf_log = _latest_log(logs, "*_perf.jsonl")
        manifest_entry = manifest_stages.get(stage, {})
        stage_outcome = (
            "disabled"
            if stage == "latent" and not getattr(cfg, "full_run_latent", True)
            else _perf_outcome(perf_log)
            or (manifest_entry.get("state") if isinstance(manifest_entry, dict) else None)
            or "not_started"
        )
        stages.append(
            {
                "stage": stage,
                "outcome": stage_outcome,
                "manifest_state": (
                    manifest_entry.get("state") if isinstance(manifest_entry, dict) else None
                ),
                "human_log": _relative_log(run_root, human_log),
                "performance_log": _relative_log(run_root, perf_log),
            }
        )
    payload = {
        "schema_version": FULL_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "outcome": outcome,
        "stages": stages,
    }
    if error is not None:
        payload["exception"] = {"type": type(error).__name__, "message": str(error)}
    return atomic_write_json(run_root / FULL_SUMMARY_FILENAME, payload)


def raw_adata(config_path: str):
    from ..cli.raw_adata import raw_adata as _raw_adata

    return _raw_adata(config_path)


def preprocess_adata(config_path: str):
    from ..cli.preprocess_adata import preprocess_adata as _preprocess_adata

    return _preprocess_adata(config_path)


def spatial_adata(config_path: str):
    from ..cli.spatial_adata import spatial_adata as _spatial_adata

    return _spatial_adata(config_path)


def hmm_adata(config_path: str):
    from ..cli.hmm_adata import hmm_adata as _hmm_adata

    return _hmm_adata(config_path)


def latent_adata(config_path: str):
    from ..cli.latent_adata import latent_adata as _latent_adata

    return _latent_adata(config_path)


def _stage_runners():
    """Return current wrapper callables so tests and callers can replace them independently."""
    return {
        "raw": raw_adata,
        "preprocess": preprocess_adata,
        "spatial": spatial_adata,
        "hmm": hmm_adata,
        "latent": latent_adata,
    }


def run_experiment_target(config_path: str, target: str):
    """Execute one public experiment target through the semantic graph."""
    from ..pipeline.experiment_graph import execute_experiment_target

    execution = execute_experiment_target(
        config_path,
        target,
        stage_runners=_stage_runners(),
    )
    return execution.final_result


def full_flow(config_path: str):
    """Run the configured final experiment target through the semantic graph."""
    from smftools.constants import PARTITIONED_STAGE_REQUIRED_ARTIFACTS

    from ..pipeline.experiment_graph import execute_experiment_target
    from .basecall import basecall_stage
    from .helpers import (
        get_adata_paths,
        load_experiment_config,
        partitioned_stage_is_complete,
        publish_stage_outputs,
        stage_lifecycle,
    )

    cfg = load_experiment_config(config_path)
    run_latent = bool(getattr(cfg, "full_run_latent", True))
    paths = get_adata_paths(cfg)
    with stage_lifecycle(cfg, "full") as lifecycle:
        try:
            basecall_stage(cfg)
            execution = execute_experiment_target(
                config_path,
                "full",
                cfg=cfg,
                paths=paths,
                stage_runners=_stage_runners(),
            )
            result = execution.final_result
            outputs = {}
            required = ()
            result_path = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            final_spine = paths.latent_spine if run_latent else paths.hmm_spine
            if result_path is not None and final_spine is not None:
                result_path = Path(result_path)
                if result_path == Path(final_spine):
                    required_stages = tuple(
                        decision.analysis_id.split(".")[1] for decision in execution.plan.decisions
                    )
                    incomplete = [
                        stage
                        for stage in required_stages
                        if not partitioned_stage_is_complete(
                            cfg,
                            stage,
                            required=PARTITIONED_STAGE_REQUIRED_ARTIFACTS[stage],
                        )
                    ]
                    if incomplete:
                        raise RuntimeError(
                            "full workflow cannot publish completion; incomplete stage record(s): "
                            f"{incomplete}"
                        )
                    spine_key = "latent_spine" if run_latent else "hmm_spine"
                    outputs[spine_key] = result_path
                    required = (spine_key,)
            summary_path = _write_full_summary(cfg, outcome="completed")
            outputs["summary"] = summary_path
            required = (*required, "summary")
            publish_stage_outputs(
                lifecycle,
                outputs,
                required=required,
                task_catalog_key=None,
                checksum_keys=("summary",),
                schema_versions={"full_workflow": FULL_SUMMARY_SCHEMA_VERSION},
                task_count=len(execution.plan.decisions),
            )
        except BaseException as exc:
            _write_full_summary(cfg, outcome="failed", error=exc)
            raise
    return result
