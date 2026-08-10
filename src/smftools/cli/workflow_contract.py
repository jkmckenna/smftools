"""Engine-facing execution, result, version, and validation contracts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterator, Mapping
from urllib.parse import unquote, urlparse
from uuid import uuid4

import pandas as pd

from .._version import __version__
from ..constants import PARTITIONED_STAGE_REQUIRED_ARTIFACTS, PREPROCESS_DIR
from ..informatics.experiment_manifest import (
    MANIFEST_FILENAME,
    read_experiment_manifest,
    resolve_artifact_record,
    stage_is_complete,
)
from ..pipeline.experiment_graph import EXPERIMENT_NODE_IDS, EXPERIMENT_STAGES
from ..readwrite import atomic_write_json

WORKFLOW_RESULT_FILENAME = "workflow_result.json"
WORKFLOW_VERSIONS_FILENAME = "software_versions.json"
WORKFLOW_RUNTIME_DIRECTORY = ".smftools-workflow"
WORKFLOW_RUNTIME_CONFIG_FILENAME = "runtime_config.csv"
WORKFLOW_LOCK_FILENAME = "run.lock"
WORKFLOW_RESULT_SCHEMA_VERSION = 1
WORKFLOW_VERSIONS_SCHEMA_VERSION = 2
WORKFLOW_VALIDATION_SCHEMA_VERSION = 1
_SUCCESS_OUTCOMES = frozenset({"success", "compatible_skip"})
_RESULT_OUTCOMES = _SUCCESS_OUTCOMES | {"failed"}
_RESULT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "result_id",
        "run_id",
        "command",
        "target",
        "outcome",
        "started_at",
        "completed_at",
        "timings",
        "run_root",
        "output_root",
        "runtime_config",
        "plan",
        "post_plan",
        "generation_ids",
        "artifacts",
        "schemas",
        "resources",
        "sources",
        "strict",
        "failure",
    }
)
_TOOL_VERSION_COMMANDS = {
    "bedGraphToBigWig": ("bedGraphToBigWig",),
    "bedtools": ("bedtools", "--version"),
    "dorado": ("dorado", "--version"),
    "gzip": ("gzip", "--version"),
    "minimap2": ("minimap2", "--version"),
    "modkit": ("modkit", "--version"),
    "multiqc": ("multiqc", "--version"),
    "pod5": ("pod5", "--version"),
    "samtools": ("samtools", "--version"),
}
_CONTAINER_ENVIRONMENT_FIELDS = {
    "image": "SMFTOOLS_CONTAINER_IMAGE",
    "tag": "SMFTOOLS_CONTAINER_TAG",
    "digest": "SMFTOOLS_CONTAINER_DIGEST",
    "revision": "SMFTOOLS_CONTAINER_REVISION",
    "profile": "SMFTOOLS_CONTAINER_PROFILE",
}


class WorkflowContractError(RuntimeError):
    """Raised when an engine-facing workflow contract cannot be satisfied."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_path(value: str | Path, *, label: str) -> Path:
    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise WorkflowContractError(
            f"{label} uses unsupported URI scheme {parsed.scheme!r}; "
            "supported inputs are local paths and file:// URIs"
        )
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise WorkflowContractError(f"{label} file URI must refer to the local host")
        raw = unquote(parsed.path)
    return Path(raw).expanduser().resolve()


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise WorkflowContractError(f"{label} must be inside the declared output root: {root}")
    return resolved


def _source_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if path.is_dir():
        digest = hashlib.sha256()
        count = 0
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            stat = child.stat()
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8"))
            count += 1
        return {"kind": "directory", "file_count": count, "metadata_sha256": digest.hexdigest()}
    raise WorkflowContractError(f"staged input does not exist: {path}")


def _snapshot_sources(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "fingerprint": _source_fingerprint(path),
        }
        for name, path in sorted(paths.items())
    }


def _assert_sources_unchanged(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    changed = []
    for name, record in snapshot.items():
        path = Path(str(record["path"]))
        if not path.exists() or _source_fingerprint(path) != record["fingerprint"]:
            changed.append(name)
    if changed:
        raise WorkflowContractError(f"staged input source(s) were modified: {changed}")


def _set_config_value(frame: pd.DataFrame, name: str, value: Any, value_type: str) -> None:
    frame.drop(frame.index[frame["variable"].astype(str).str.strip() == name], inplace=True)
    next_index = 0 if frame.empty else int(max(frame.index)) + 1
    frame.loc[next_index] = {"variable": name, "value": value, "type": value_type}


def _stage_readonly_alias(source: Path, runtime_dir: Path, *, stem: str) -> Path:
    if not source.exists():
        raise WorkflowContractError(f"staged source does not exist: {source}")
    if not source.is_file():
        raise WorkflowContractError(
            f"workflow staging requires a concrete file for {stem}; "
            "stage directory inputs to one file before invocation"
        )
    if source.is_relative_to(runtime_dir.parent):
        return source
    suffix = "".join(source.suffixes)
    alias = runtime_dir / f"{stem}{suffix}"
    if alias.is_symlink():
        alias.unlink()
    elif alias.exists():
        raise WorkflowContractError(f"workflow input alias is occupied: {alias}")
    alias.symlink_to(source)
    return alias


def _resolved_accelerator(cfg: Any, requested: str | None) -> str:
    configured = str(getattr(cfg, "device", "auto") or "auto").strip().lower()
    requested = configured if requested is None else str(requested).strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise WorkflowContractError("accelerator must be one of: auto, cpu, cuda, mps")
    if configured == "cpu" and requested not in {"auto", "cpu"}:
        raise WorkflowContractError("accelerator override exceeds the config's CPU-only ceiling")
    if requested in {"cuda", "mps"}:
        try:
            import torch
        except ImportError as exc:
            raise WorkflowContractError(f"accelerator {requested!r} requires torch") from exc
        available = (
            torch.cuda.is_available()
            if requested == "cuda"
            else bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        )
        if not available:
            raise WorkflowContractError(f"requested accelerator {requested!r} is unavailable")
    return requested


def _write_runtime_config(
    source_config: Path,
    output_root: Path,
    *,
    input_path: Path | None,
    fasta_path: Path | None,
    cpus: int | None,
    memory_gb: float | None,
    accelerator: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    from ..cli.helpers import load_experiment_config
    from ..config import LoadExperimentConfig

    source_cfg = load_experiment_config(str(source_config))
    source_envelope = source_cfg._resource_envelope
    resolved_cpus = source_envelope.resolved_threads
    if cpus is not None:
        if int(cpus) < 1:
            raise WorkflowContractError("cpus must be positive")
        resolved_cpus = min(resolved_cpus, int(cpus))
    ceiling_gb = source_envelope.resolved_memory_bytes / (1024**3)
    resolved_memory_gb = ceiling_gb if memory_gb is None else min(ceiling_gb, float(memory_gb))
    if resolved_memory_gb <= 0:
        raise WorkflowContractError("memory_gb must be positive")
    resolved_accelerator = _resolved_accelerator(source_cfg, accelerator)

    runtime_dir = output_root / WORKFLOW_RUNTIME_DIRECTORY
    runtime_dir.mkdir(parents=True, exist_ok=True)
    effective_sources: dict[str, Path] = {}
    configured_manifest = (
        _local_path(source_cfg.input_manifest_path, label="configured input manifest")
        if input_path is None and getattr(source_cfg, "input_manifest_path", None)
        else None
    )
    configured_input = input_path or (
        _local_path(source_cfg.input_data_path, label="configured input")
        if configured_manifest is None and getattr(source_cfg, "input_data_path", None)
        else None
    )
    configured_fasta = fasta_path or (
        _local_path(source_cfg.fasta, label="configured FASTA")
        if getattr(source_cfg, "fasta", None)
        else None
    )
    aliased_input = None
    aliased_manifest = None
    if configured_manifest is not None:
        effective_sources["input_manifest"] = configured_manifest
        for index, source in enumerate(getattr(source_cfg, "input_files", ()) or ()):
            effective_sources[f"input:{index:06d}"] = Path(source).resolve()
        aliased_manifest = _stage_readonly_alias(
            configured_manifest, runtime_dir, stem="input_manifest"
        )
    if configured_input is not None:
        effective_sources["input"] = configured_input
        aliased_input = _stage_readonly_alias(configured_input, runtime_dir, stem="input")
    aliased_fasta = None
    if configured_fasta is not None:
        effective_sources["fasta"] = configured_fasta
        aliased_fasta = _stage_readonly_alias(configured_fasta, runtime_dir, stem="reference")

    frame = LoadExperimentConfig(source_config).df.copy()
    _set_config_value(frame, "output_directory", str(output_root), "string")
    _set_config_value(frame, "threads", resolved_cpus, "int")
    _set_config_value(frame, "max_memory_gb", resolved_memory_gb, "float")
    _set_config_value(frame, "device", resolved_accelerator, "string")
    _set_config_value(frame, "hmm_device", resolved_accelerator, "string")
    if aliased_manifest is not None:
        _set_config_value(frame, "input_data_path", "", "string")
        _set_config_value(frame, "input_manifest_path", str(aliased_manifest), "string")
    elif aliased_input is not None:
        _set_config_value(frame, "input_manifest_path", "", "string")
        _set_config_value(frame, "input_data_path", str(aliased_input), "string")
    if aliased_fasta is not None:
        _set_config_value(frame, "fasta", str(aliased_fasta), "string")

    runtime_config = runtime_dir / WORKFLOW_RUNTIME_CONFIG_FILENAME
    frame.to_csv(runtime_config, index=False)
    return (
        runtime_config,
        {
            "requested": {
                "cpus": cpus,
                "memory_gb": memory_gb,
                "accelerator": accelerator,
            },
            "ceiling": source_envelope.as_dict(),
            "resolved": {
                "cpus": resolved_cpus,
                "memory_bytes": int(resolved_memory_gb * (1024**3)),
                "accelerator": resolved_accelerator,
            },
        },
        effective_sources,
    )


def _required_external_tools(cfg: Any, *, raw_will_run: bool) -> tuple[str, ...]:
    if not raw_will_run:
        return ()
    tools: set[str] = set()
    input_type = str(getattr(cfg, "input_type", "") or "").lower()
    if input_type in {"pod5", "fast5"}:
        tools.add("dorado")
    if input_type == "fast5":
        tools.add("pod5")
    aligner = str(getattr(cfg, "aligner", "") or "").lower()
    if aligner in {"dorado", "minimap2"}:
        tools.add(aligner)
    if (
        str(getattr(cfg, "demux_backend", "") or "").lower() == "dorado"
        and getattr(cfg, "barcode_kit", None)
        and not bool(getattr(cfg, "input_already_demuxed", False))
    ):
        tools.add("dorado")
    if (
        str(getattr(cfg, "smf_modality", "") or "").lower() == "direct"
        and str(getattr(cfg, "direct_signal_backend", "") or "").lower() == "modkit"
    ):
        tools.update(("gzip", "modkit"))
    samtools_backend = str(getattr(cfg, "samtools_backend", "") or "").lower()
    if samtools_backend == "cli" or (samtools_backend == "auto" and shutil.which("samtools")):
        tools.add("samtools")
    if shutil.which("samtools"):
        tools.add("samtools")
    if not bool(getattr(cfg, "skip_bam_qc", True)):
        tools.add("multiqc")
    if bool(getattr(cfg, "make_beds", False)) and bool(getattr(cfg, "make_bigwigs", False)):
        bedtools_backend = str(getattr(cfg, "bedtools_backend", "auto") or "auto").lower()
        if bedtools_backend == "cli" or (bedtools_backend == "auto" and shutil.which("bedtools")):
            tools.add("bedtools")
        bigwig_backend = str(getattr(cfg, "bigwig_backend", "auto") or "auto").lower()
        if bigwig_backend == "cli" or (
            bigwig_backend == "auto" and shutil.which("bedGraphToBigWig")
        ):
            tools.add("bedGraphToBigWig")
        if shutil.which("samtools"):
            tools.add("samtools")
    return tuple(sorted(tools))


def _target_stage(cfg: Any, target: str) -> str:
    if target == "full":
        return "latent" if bool(getattr(cfg, "full_run_latent", True)) else "hmm"
    if target == "variant":
        return "preprocess"
    return str(target)


def _require_completed_target(output_root: Path, cfg: Any, target: str) -> None:
    stage = _target_stage(cfg, target)
    if stage not in EXPERIMENT_STAGES:
        raise WorkflowContractError(f"unsupported workflow target stage: {stage!r}")
    if not stage_is_complete(
        output_root,
        stage,
        required_artifacts=PARTITIONED_STAGE_REQUIRED_ARTIFACTS[stage],
    ):
        raise WorkflowContractError(
            f"target stage {stage!r} did not publish a complete validated lifecycle record"
        )
    if stage != "preprocess":
        return
    from ..preprocessing.preprocess_generation import (
        PreprocessGenerationError,
        resolve_current_preprocess_generation,
    )

    try:
        current = resolve_current_preprocess_generation(output_root / PREPROCESS_DIR)
    except PreprocessGenerationError as exc:
        raise WorkflowContractError(str(exc)) from exc
    expected = (
        read_experiment_manifest(output_root)
        .get("stages", {})
        .get("preprocess", {})
        .get("generation_id")
    )
    if current is None or current[1].get("generation_id") != expected:
        raise WorkflowContractError(
            "preprocess current pointer does not match the published stage generation"
        )


def _tool_version(tool: str) -> dict[str, Any]:
    command = _TOOL_VERSION_COMMANDS[tool]
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "version": None}
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "version": None, "error": type(exc).__name__}
    output = (completed.stdout.strip() or completed.stderr.strip()).splitlines()
    return {
        "available": True,
        "version": output[0] if output else None,
        "version_probe_exit_code": completed.returncode,
    }


def software_versions(
    *,
    tools: tuple[str, ...] = (),
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Return stable package, interpreter, external-tool, and model versions."""
    unknown = sorted(set(tools).difference(_TOOL_VERSION_COMMANDS))
    if unknown:
        raise WorkflowContractError(f"unsupported external tool version request(s): {unknown}")
    models = {}
    if cfg is not None:
        for field in ("model", "dorado_model", "dorado_modification_model"):
            value = getattr(cfg, field, None)
            if value not in (None, ""):
                models[field] = str(value)
    container = {
        field: os.environ.get(environment_name) or None
        for field, environment_name in _CONTAINER_ENVIRONMENT_FIELDS.items()
    }
    return {
        "schema_version": WORKFLOW_VERSIONS_SCHEMA_VERSION,
        "smftools": __version__,
        "python": platform.python_version(),
        "platform": sys.platform,
        "container": container,
        "external_tools": {tool: _tool_version(tool) for tool in sorted(tools)},
        "models": models,
    }


def _write_versions(
    output_root: Path,
    *,
    tools: tuple[str, ...],
    cfg: Any,
    strict: bool,
) -> tuple[Path, dict[str, Any]]:
    payload = software_versions(tools=tools, cfg=cfg)
    if strict:
        missing = [
            name
            for name, record in payload["external_tools"].items()
            if not record.get("available")
        ]
        if missing:
            raise WorkflowContractError(
                f"strict workflow mode requires unavailable external tool(s): {missing}"
            )
    path = atomic_write_json(output_root / WORKFLOW_VERSIONS_FILENAME, payload)
    return path, payload


@contextmanager
def _exclusive_run(output_root: Path) -> Iterator[None]:
    runtime_dir = output_root / WORKFLOW_RUNTIME_DIRECTORY
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / WORKFLOW_LOCK_FILENAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise WorkflowContractError(
            f"output root is already owned by another workflow invocation: {output_root}"
        ) from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _relative_artifact(
    output_root: Path,
    path: Path,
    *,
    artifact_id: str,
    schema_version: int | None = None,
    checksum: str | None = None,
) -> dict[str, Any]:
    path = _require_within(path, output_root, label=f"artifact {artifact_id!r}")
    payload = {
        "artifact_id": artifact_id,
        "path": path.relative_to(output_root).as_posix(),
        "kind": "directory" if path.is_dir() else "file",
        "checksum": checksum,
    }
    if schema_version is not None:
        payload["schema_version"] = int(schema_version)
    if path.is_file():
        payload["size_bytes"] = path.stat().st_size
    return payload


def _collect_artifacts(
    output_root: Path,
    *,
    runtime_config: Path | None,
    versions_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    manifest = read_experiment_manifest(output_root)
    stages = manifest.get("stages", {})
    artifacts = []
    if runtime_config is not None and runtime_config.is_file():
        artifacts.append(
            _relative_artifact(
                output_root,
                runtime_config,
                artifact_id="workflow:runtime_config",
                checksum=_sha256(runtime_config),
            )
        )
    if versions_path is not None and versions_path.is_file():
        artifacts.append(
            _relative_artifact(
                output_root,
                versions_path,
                artifact_id="workflow:versions",
                schema_version=WORKFLOW_VERSIONS_SCHEMA_VERSION,
                checksum=_sha256(versions_path),
            )
        )
    manifest_path = output_root / MANIFEST_FILENAME
    if manifest_path.is_file():
        artifacts.append(
            _relative_artifact(
                output_root,
                manifest_path,
                artifact_id="experiment:manifest",
                checksum=_sha256(manifest_path),
            )
        )
    schemas: dict[str, int] = {}
    generation_ids: dict[str, str] = {}
    if isinstance(stages, Mapping):
        for stage, entry in sorted(stages.items()):
            if not isinstance(entry, Mapping):
                continue
            if entry.get("generation_id") is not None:
                generation_ids[str(stage)] = str(entry["generation_id"])
            for name, version in dict(entry.get("schema_versions", {})).items():
                schemas[f"{stage}:{name}"] = int(version)
            stage_artifacts = entry.get("artifacts", {})
            if not isinstance(stage_artifacts, Mapping):
                continue
            for name, record in sorted(stage_artifacts.items()):
                if not isinstance(record, dict):
                    continue
                path = resolve_artifact_record(output_root, record)
                if path is None or not path.exists():
                    continue
                schema_version = None
                if name in dict(entry.get("schema_versions", {})):
                    schema_version = int(entry["schema_versions"][name])
                artifacts.append(
                    _relative_artifact(
                        output_root,
                        path,
                        artifact_id=f"{stage}:{name}",
                        schema_version=schema_version,
                        checksum=record.get("sha256"),
                    )
                )
    deduplicated = {(item["artifact_id"], item["path"]): item for item in artifacts}
    return list(deduplicated.values()), schemas, generation_ids


def _result_path(output_root: Path, value: str | Path | None) -> Path:
    if value is None:
        return output_root / WORKFLOW_RESULT_FILENAME
    path = Path(value)
    if not path.is_absolute():
        path = output_root / path
    path = _require_within(path, output_root, label="result JSON")
    if path.parent != output_root:
        raise WorkflowContractError("result JSON must be a direct child of the output root")
    if path.name in {MANIFEST_FILENAME, WORKFLOW_VERSIONS_FILENAME}:
        raise WorkflowContractError(f"result JSON uses reserved workflow name: {path.name}")
    return path


def workflow_result_contract_issues(payload: Mapping[str, Any]) -> list[str]:
    """Return stable structural errors for one workflow-result payload."""
    issues = []
    missing = sorted(_RESULT_REQUIRED_FIELDS.difference(payload))
    if missing:
        issues.append(f"missing required field(s): {missing}")
    if payload.get("schema_version") != WORKFLOW_RESULT_SCHEMA_VERSION:
        issues.append(f"schema_version must be {WORKFLOW_RESULT_SCHEMA_VERSION}")
    if payload.get("outcome") not in _RESULT_OUTCOMES:
        issues.append(f"outcome must be one of {sorted(_RESULT_OUTCOMES)}")
    for field in ("result_id", "run_id", "command", "target"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            issues.append(f"{field} must be a non-empty string")
    if payload.get("run_root") != "." or payload.get("output_root") != ".":
        issues.append("run_root and output_root must both be '.'")
    for field in ("timings", "generation_ids", "schemas", "resources", "sources"):
        if not isinstance(payload.get(field), Mapping):
            issues.append(f"{field} must be an object")
    if not isinstance(payload.get("artifacts"), list):
        issues.append("artifacts must be an array")
    if payload.get("outcome") == "failed" and not isinstance(payload.get("failure"), Mapping):
        issues.append("failed results require a structured failure object")
    if payload.get("outcome") in _SUCCESS_OUTCOMES and payload.get("failure") is not None:
        issues.append("successful results must not contain a failure")
    if payload.get("outcome") in _SUCCESS_OUTCOMES and not isinstance(
        payload.get("post_plan"), Mapping
    ):
        issues.append("successful results require a post-execution plan")
    return issues


def run_experiment_workflow(
    config_path: str | Path,
    *,
    target: str,
    output_root: str | Path,
    input_path: str | Path | None = None,
    fasta_path: str | Path | None = None,
    result_json: str | Path | None = None,
    cpus: int | None = None,
    memory_gb: float | None = None,
    accelerator: str | None = None,
    strict: bool = False,
) -> Path:
    """Run one experiment under a task-local, machine-readable workflow contract."""
    from ..cli.helpers import load_experiment_config
    from ..cli.recipes import full_flow, run_experiment_target
    from ..cli.variant_adata import variant_adata
    from ..pipeline.experiment_graph import plan_experiment

    source_config = _local_path(config_path, label="config")
    run_root = _local_path(output_root, label="output root")
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(run_root, result_json)
    staged_input = None if input_path is None else _local_path(input_path, label="staged input")
    staged_fasta = None if fasta_path is None else _local_path(fasta_path, label="staged FASTA")
    for label, path in (("staged input", staged_input), ("staged FASTA", staged_fasta)):
        if path is not None and run_root.is_relative_to(path):
            raise WorkflowContractError(f"output root must not be nested inside {label}: {path}")
    source_snapshot = _snapshot_sources({"config": source_config})
    started = perf_counter()
    started_at = _now()
    run_id = uuid4().hex
    runtime_config: Path | None = None
    versions_path: Path | None = None
    resources: dict[str, Any] = {}
    plan_payload: dict[str, Any] | None = None
    post_plan_payload: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    outcome = "failed"

    with _exclusive_run(run_root):
        try:
            runtime_config, resources, effective_sources = _write_runtime_config(
                source_config,
                run_root,
                input_path=staged_input,
                fasta_path=staged_fasta,
                cpus=cpus,
                memory_gb=memory_gb,
                accelerator=accelerator,
            )
            source_snapshot.update(_snapshot_sources(effective_sources))
            cfg = load_experiment_config(str(runtime_config))
            plan = plan_experiment(runtime_config, target)
            plan_payload = plan.to_dict()
            raw_id = EXPERIMENT_NODE_IDS["raw"]
            raw_will_run = any(
                decision.analysis_id == raw_id and decision.state.value != "compatible"
                for decision in plan.decisions
            )
            tools = _required_external_tools(cfg, raw_will_run=raw_will_run)
            versions_path, _ = _write_versions(
                run_root,
                tools=tools,
                cfg=cfg,
                strict=strict,
            )
            if target == "full":
                full_flow(str(runtime_config))
            elif target == "variant":
                variant_adata(str(runtime_config))
            else:
                run_experiment_target(str(runtime_config), target)
            _require_completed_target(run_root, cfg, target)
            post_plan = plan_experiment(runtime_config, target)
            post_plan_payload = post_plan.to_dict()
            incompatible = [
                decision for decision in post_plan.decisions if decision.state.value != "compatible"
            ]
            if incompatible:
                raise WorkflowContractError(
                    "target completed but its post-execution semantic plan is not compatible: "
                    f"{[decision.analysis_id for decision in incompatible]}"
                )
            outcome = (
                "compatible_skip"
                if plan.decisions
                and all(decision.state.value == "compatible" for decision in plan.decisions)
                else "success"
            )
        except Exception as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "workflow",
            }
        try:
            _assert_sources_unchanged(source_snapshot)
        except Exception as exc:
            outcome = "failed"
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "source_integrity",
            }
        try:
            artifacts, schemas, generation_ids = _collect_artifacts(
                run_root,
                runtime_config=runtime_config,
                versions_path=versions_path,
            )
        except Exception as exc:
            outcome = "failed"
            if failure is None:
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": "artifact_publication",
                }
            artifacts, schemas, generation_ids = [], {}, {}
        payload: dict[str, Any] = {
            "schema_version": WORKFLOW_RESULT_SCHEMA_VERSION,
            "result_id": uuid4().hex,
            "run_id": run_id,
            "command": "experiment.run",
            "target": str(target),
            "outcome": outcome,
            "started_at": started_at,
            "completed_at": _now(),
            "timings": {"elapsed_seconds": perf_counter() - started},
            "run_root": ".",
            "output_root": ".",
            "runtime_config": (
                runtime_config.relative_to(run_root).as_posix()
                if runtime_config is not None
                else None
            ),
            "plan": plan_payload,
            "post_plan": post_plan_payload,
            "generation_ids": generation_ids,
            "artifacts": artifacts,
            "schemas": schemas,
            "resources": resources,
            "sources": {
                name: {
                    "kind": record["fingerprint"]["kind"],
                    "fingerprint": record["fingerprint"],
                }
                for name, record in source_snapshot.items()
            },
            "strict": bool(strict),
            "failure": failure,
        }
        contract_issues = workflow_result_contract_issues(payload)
        if contract_issues:
            raise WorkflowContractError(
                f"refusing to publish invalid workflow result: {contract_issues}"
            )
        atomic_write_json(result_path, payload)
        if failure is not None:
            raise WorkflowContractError(
                f"workflow failed; structured result written to {result_path}: "
                f"{failure['type']}: {failure['message']}"
            )
    return result_path


def _project_resource_decision(
    *,
    cpus: int | None,
    memory_gb: float | None,
    memory_percent: float | None,
) -> dict[str, Any]:
    from ..memory_guard import resolve_resource_envelope

    cfg = SimpleNamespace(
        threads=cpus,
        max_memory_gb=memory_gb,
        max_memory_percent=memory_percent,
        memory_reserve_gb=1.0,
    )
    envelope = resolve_resource_envelope(cfg)
    return {
        "requested": {
            "cpus": cpus,
            "memory_gb": memory_gb,
            "memory_percent": memory_percent,
        },
        "ceiling": envelope.as_dict(),
        "resolved": {
            "cpus": envelope.resolved_threads,
            "memory_bytes": envelope.resolved_memory_bytes,
            "accelerator": "cpu",
        },
    }


def _project_request(
    canonical_reference: str,
    *,
    output_name: str,
    set_name: str | None,
    modality: str | None,
    experiments: tuple[str, ...],
    stage: str | None,
    start: int | None,
    end: int | None,
    layers: list[str] | None,
    read_metrics: bool,
    partitioned: bool,
) -> dict[str, Any]:
    return {
        "canonical_reference": str(canonical_reference),
        "output_name": output_name,
        "set_name": set_name,
        "modality": modality,
        "experiments": list(experiments) if experiments else None,
        "stage": stage,
        "start": start,
        "end": end,
        "layers": layers,
        "read_metrics": bool(read_metrics),
        "partitioned": bool(partitioned),
    }


def _project_output_is_compatible(
    result_path: Path,
    output_root: Path,
    *,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        result.get("command") != "project.materialize"
        or result.get("request") != dict(request)
        or result.get("plan") != dict(plan)
        or result.get("outcome") not in _SUCCESS_OUTCOMES
    ):
        return False
    for artifact in result.get("artifacts", []):
        if artifact.get("artifact_id") != "project:materialization":
            continue
        path = (output_root / str(artifact.get("path", ""))).resolve()
        if (
            not path.is_relative_to(output_root)
            or not path.exists()
            or (artifact.get("checksum") is not None and _sha256(path) != artifact.get("checksum"))
        ):
            return False
        return True
    return False


def run_project_materialization_workflow(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    output_root: str | Path,
    output_name: str | None = None,
    result_json: str | Path | None = None,
    set_name: str | None = None,
    modality: str | None = None,
    experiments: tuple[str, ...] = (),
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers: list[str] | None = None,
    read_metrics: bool = False,
    allow_large: bool = False,
    partitioned: bool = False,
    cpus: int | None = None,
    memory_gb: float | None = None,
    memory_percent: float | None = 60.0,
) -> Path:
    """Materialize one project selection under the shared workflow contract."""
    from ..cli.project_cmd import project_materialize, project_plan

    source_project = _local_path(project_dir, label="project")
    run_root = _local_path(output_root, label="output root")
    if run_root.is_relative_to(source_project):
        raise WorkflowContractError(
            "project workflow output root must be outside the source project directory"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(run_root, result_json)
    name = output_name or ("materialized" if partitioned else "materialized.h5ad.gz")
    materialized_path = _require_within(
        run_root / name,
        run_root,
        label="project materialization output",
    )
    reserved = (
        result_path,
        run_root / WORKFLOW_VERSIONS_FILENAME,
        run_root / WORKFLOW_RUNTIME_DIRECTORY,
    )
    if any(
        materialized_path == path
        or materialized_path.is_relative_to(path)
        or path.is_relative_to(materialized_path)
        for path in reserved
    ):
        raise WorkflowContractError(
            f"project output name overlaps workflow-owned control artifacts: {name!r}"
        )
    request = _project_request(
        canonical_reference,
        output_name=name,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layers,
        read_metrics=read_metrics,
        partitioned=partitioned,
    )
    started = perf_counter()
    started_at = _now()
    run_id = uuid4().hex
    failure: dict[str, Any] | None = None
    plan_payload: dict[str, Any] | None = None
    versions_path: Path | None = None
    resources: dict[str, Any] = {}
    outcome = "failed"

    with _exclusive_run(run_root):
        try:
            plan = project_plan(
                source_project,
                "materialization",
                canonical_reference,
                set_name=set_name,
                modality=modality,
                experiments=experiments,
                stage=stage,
                start=start,
                end=end,
                layers=layers,
                read_metrics=read_metrics,
                partitioned=partitioned,
            )
            plan_payload = plan.to_dict()
            resources = _project_resource_decision(
                cpus=cpus,
                memory_gb=memory_gb,
                memory_percent=memory_percent,
            )
            versions_path, _ = _write_versions(
                run_root,
                tools=(),
                cfg=SimpleNamespace(),
                strict=False,
            )
            if _project_output_is_compatible(
                result_path,
                run_root,
                request=request,
                plan=plan_payload,
            ):
                outcome = "compatible_skip"
            else:
                project_materialize(
                    source_project,
                    canonical_reference,
                    materialized_path,
                    set_name=set_name,
                    modality=modality,
                    stage=stage,
                    start=start,
                    end=end,
                    layers=layers,
                    read_metrics=read_metrics,
                    allow_large=allow_large,
                    partitioned=partitioned,
                    max_memory_gb=resources["resolved"]["memory_bytes"] / (1024**3),
                    max_memory_percent=None,
                )
                if not materialized_path.exists():
                    raise WorkflowContractError(
                        "project materialization returned without publishing its output"
                    )
                outcome = "success"
        except Exception as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "project.materialization",
            }
        artifacts = []
        if materialized_path.exists():
            artifacts.append(
                _relative_artifact(
                    run_root,
                    materialized_path,
                    artifact_id="project:materialization",
                    schema_version=1,
                    checksum=_sha256(materialized_path),
                )
            )
        if versions_path is not None and versions_path.is_file():
            artifacts.append(
                _relative_artifact(
                    run_root,
                    versions_path,
                    artifact_id="workflow:versions",
                    schema_version=WORKFLOW_VERSIONS_SCHEMA_VERSION,
                    checksum=_sha256(versions_path),
                )
            )
        payload: dict[str, Any] = {
            "schema_version": WORKFLOW_RESULT_SCHEMA_VERSION,
            "result_id": uuid4().hex,
            "run_id": run_id,
            "command": "project.materialize",
            "target": "materialization",
            "outcome": outcome,
            "started_at": started_at,
            "completed_at": _now(),
            "timings": {"elapsed_seconds": perf_counter() - started},
            "run_root": ".",
            "output_root": ".",
            "runtime_config": None,
            "plan": plan_payload,
            "post_plan": plan_payload,
            "generation_ids": {},
            "artifacts": artifacts,
            "schemas": {"project:materialization": 1},
            "resources": resources,
            "sources": {
                "project": {
                    "kind": "project_registry",
                    "fingerprint": _project_plan_identity(plan_payload),
                }
            },
            "strict": False,
            "failure": failure,
            "request": request,
        }
        contract_issues = workflow_result_contract_issues(payload)
        if contract_issues:
            raise WorkflowContractError(
                f"refusing to publish invalid workflow result: {contract_issues}"
            )
        atomic_write_json(result_path, payload)
        if failure is not None:
            raise WorkflowContractError(
                f"project workflow failed; structured result written to {result_path}: "
                f"{failure['type']}: {failure['message']}"
            )
    return result_path


def _project_plan_identity(plan: Mapping[str, Any] | None) -> str | None:
    if not isinstance(plan, Mapping):
        return None
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_issue(code: str, message: str, **fields: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **fields}


def validate_workflow_output(
    output_root: str | Path,
    *,
    result_json: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a workflow result, semantic compatibility, pointers, and checksums."""
    run_root = _local_path(output_root, label="output root")
    path = _result_path(run_root, result_json)
    issues: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        result = {}
        issues.append(_validation_issue("result_unreadable", str(exc)))
    for message in workflow_result_contract_issues(result):
        issues.append(_validation_issue("result_contract_invalid", message))
    if result.get("outcome") not in _SUCCESS_OUTCOMES:
        issues.append(
            _validation_issue(
                "result_not_successful",
                f"workflow outcome is {result.get('outcome')!r}",
            )
        )
    if result.get("run_root") != "." or result.get("output_root") != ".":
        issues.append(
            _validation_issue(
                "root_pointer_mismatch",
                "run_root and output_root must be relocation-safe '.' pointers",
            )
        )
    for artifact in result.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            issues.append(_validation_issue("artifact_record_invalid", "artifact is not an object"))
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            issues.append(
                _validation_issue(
                    "artifact_pointer_invalid",
                    f"artifact path must be relative: {raw_path!r}",
                )
            )
            continue
        artifact_path = (run_root / raw_path).resolve()
        if not artifact_path.is_relative_to(run_root):
            issues.append(
                _validation_issue(
                    "artifact_pointer_escape",
                    f"artifact escapes output root: {raw_path!r}",
                )
            )
            continue
        if not artifact_path.exists():
            issues.append(
                _validation_issue(
                    "artifact_missing",
                    f"artifact is missing: {raw_path}",
                    artifact_id=artifact.get("artifact_id"),
                )
            )
            continue
        checksum = artifact.get("checksum")
        if checksum is not None and _sha256(artifact_path) != checksum:
            issues.append(
                _validation_issue(
                    "artifact_checksum_mismatch",
                    f"artifact checksum mismatch: {raw_path}",
                    artifact_id=artifact.get("artifact_id"),
                )
            )
    command = result.get("command")
    if command == "experiment.run":
        runtime_value = result.get("runtime_config")
        runtime_config = (
            run_root / runtime_value
            if isinstance(runtime_value, str) and not Path(runtime_value).is_absolute()
            else None
        )
        if runtime_config is None or not runtime_config.is_file():
            issues.append(
                _validation_issue("runtime_config_missing", "runtime config is unavailable")
            )
        for decision in result.get("post_plan", {}).get("decisions", []):
            if decision.get("state") != "compatible":
                issues.append(
                    _validation_issue(
                        "publication_incompatible",
                        "post-execution plan did not classify the result as compatible",
                        analysis_id=decision.get("analysis_id"),
                        state=decision.get("state"),
                        reason_code=decision.get("reason_code"),
                    )
                )
        manifest = read_experiment_manifest(run_root)
        target = str(result.get("target", "full"))
        requested_target = result.get("plan", {}).get("requested_target")
        final_stage = next(
            (
                stage
                for stage, analysis_id in EXPERIMENT_NODE_IDS.items()
                if analysis_id == requested_target
            ),
            "preprocess" if target == "variant" else ("latent" if target == "full" else target),
        )
        if final_stage in EXPERIMENT_STAGES and not stage_is_complete(
            run_root,
            final_stage,
            required_artifacts=PARTITIONED_STAGE_REQUIRED_ARTIFACTS[final_stage],
        ):
            issues.append(
                _validation_issue(
                    "stage_incomplete",
                    f"target stage {final_stage!r} failed lifecycle validation",
                )
            )
        if final_stage == "preprocess":
            from ..preprocessing.preprocess_generation import (
                PreprocessGenerationError,
                resolve_current_preprocess_generation,
            )

            try:
                current = resolve_current_preprocess_generation(run_root / PREPROCESS_DIR)
            except PreprocessGenerationError as exc:
                issues.append(_validation_issue("current_pointer_invalid", str(exc)))
            else:
                expected = manifest.get("stages", {}).get("preprocess", {}).get("generation_id")
                if current is None or current[1].get("generation_id") != expected:
                    issues.append(
                        _validation_issue(
                            "current_pointer_mismatch",
                            "preprocess current pointer does not match the published stage generation",
                        )
                    )
    elif command == "project.materialize":
        if not any(
            artifact.get("artifact_id") == "project:materialization"
            for artifact in result.get("artifacts", [])
            if isinstance(artifact, Mapping)
        ):
            issues.append(
                _validation_issue(
                    "project_output_missing",
                    "project result does not declare its materialized artifact",
                )
            )
        if project_dir is not None:
            from ..cli.project_cmd import project_plan

            request = result.get("request", {})
            try:
                current = project_plan(
                    project_dir,
                    "materialization",
                    request["canonical_reference"],
                    set_name=request.get("set_name"),
                    modality=request.get("modality"),
                    experiments=request.get("experiments"),
                    stage=request.get("stage"),
                    start=request.get("start"),
                    end=request.get("end"),
                    layers=request.get("layers"),
                    read_metrics=bool(request.get("read_metrics")),
                    partitioned=bool(request.get("partitioned")),
                )
            except Exception as exc:
                issues.append(_validation_issue("project_plan_failed", str(exc)))
            else:
                if _project_plan_identity(current.to_dict()) != _project_plan_identity(
                    result.get("plan")
                ):
                    issues.append(
                        _validation_issue(
                            "project_source_stale",
                            "current project selection no longer matches the published result",
                        )
                    )
    else:
        issues.append(
            _validation_issue(
                "command_unsupported",
                f"unsupported workflow command contract: {command!r}",
            )
        )
    return {
        "schema_version": WORKFLOW_VALIDATION_SCHEMA_VERSION,
        "valid": not issues,
        "result": path.relative_to(run_root).as_posix(),
        "issues": issues,
    }
