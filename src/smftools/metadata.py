from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Optional

from ._version import __version__
from .schema import SCHEMA_REGISTRY_RESOURCE, SCHEMA_REGISTRY_VERSION

_DEPENDENCIES = ("anndata", "numpy", "pandas", "scanpy", "torch")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _safe_version(package_name: str) -> Optional[str]:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _find_git_root(start: Path) -> Optional[Path]:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _get_git_commit() -> Optional[str]:
    root = _find_git_root(Path(__file__).resolve())
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _hash_file(path: Path, *, max_full_bytes: int = 50 * 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    hasher = hashlib.sha256()
    hash_mode = "full"
    hash_bytes = 0
    chunk_size = 1024 * 1024

    with path.open("rb") as handle:
        if size <= max_full_bytes:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
                hash_bytes += len(chunk)
        else:
            hash_mode = "head_tail_1mb"
            head = handle.read(chunk_size)
            hasher.update(head)
            hash_bytes += len(head)
            if size > chunk_size:
                handle.seek(max(size - chunk_size, 0))
                tail = handle.read(chunk_size)
                hasher.update(tail)
                hash_bytes += len(tail)

    return {
        "size": size,
        "mtime": mtime,
        "hash": hasher.hexdigest(),
        "hash_algorithm": "sha256",
        "hash_mode": hash_mode,
        "hash_bytes": hash_bytes,
    }


def _path_record(path: Path, role: Optional[str] = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    if role:
        record["role"] = role
    if not path.exists():
        record["exists"] = False
        return record

    record["exists"] = True
    if path.is_dir():
        stat = path.stat()
        record["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        record["type"] = "directory"
        return record

    record["type"] = "file"
    record.update(_hash_file(path))
    return record


def _normalize_paths(paths: Optional[Iterable[Path | str]]) -> list[Path]:
    if not paths:
        return []
    normalized = []
    for path in paths:
        if path is None:
            continue
        normalized.append(Path(path))
    return normalized


def _environment_snapshot() -> dict[str, Any]:
    dependencies = {name: version for name in _DEPENDENCIES if (version := _safe_version(name))}
    return {
        "smftools_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "dependencies": dependencies,
    }


def _infer_dtype(value: Any) -> str:
    if hasattr(value, "dtype"):
        return str(value.dtype)
    if hasattr(value, "dtypes"):
        try:
            return ",".join(str(dt) for dt in value.dtypes)
        except TypeError:
            return str(value.dtypes)
    return type(value).__name__


def _schema_snapshot(adata) -> dict[str, Any]:
    layers = {
        name: {"dtype": _infer_dtype(matrix), "shape": list(matrix.shape)}
        for name, matrix in adata.layers.items()
    }
    obsm = {
        name: {"dtype": _infer_dtype(matrix), "shape": list(matrix.shape)}
        for name, matrix in adata.obsm.items()
    }
    obsp = {
        name: {"dtype": _infer_dtype(matrix), "shape": list(matrix.shape)}
        for name, matrix in adata.obsp.items()
    }
    return {
        "layers": layers,
        "obs": {name: {"dtype": str(adata.obs[name].dtype)} for name in adata.obs.columns},
        "var": {name: {"dtype": str(adata.var[name].dtype)} for name in adata.var.columns},
        "obsm": obsm,
        "obsp": obsp,
        "uns_keys": sorted(adata.uns.keys()),
    }


def _runtime_schema_entries(items: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "dtype": _infer_dtype(value),
            "created_by": "runtime_snapshot",
            "modified_by": [],
            "notes": "",
            "requires": [],
            "optional_inputs": [],
        }
        for key, value in items.items()
    }


def _runtime_schema_dict(adata, step_name: str, output_path: Optional[Path] = None) -> dict:
    return {
        "schema_version": "runtime-1",
        "description": "Runtime AnnData schema snapshot (auto-generated).",
        "generated_at": _iso_timestamp(),
        "output_path": str(output_path) if output_path else None,
        "stages": {
            step_name: {
                "stage_requires": [],
                "obs": _runtime_schema_entries(
                    {name: adata.obs[name] for name in adata.obs.columns}
                ),
                "var": _runtime_schema_entries(
                    {name: adata.var[name] for name in adata.var.columns}
                ),
                "obsm": _runtime_schema_entries(dict(adata.obsm.items())),
                "varm": _runtime_schema_entries(dict(adata.varm.items())),
                "layers": _runtime_schema_entries(dict(adata.layers.items())),
                "obsp": _runtime_schema_entries(dict(adata.obsp.items())),
                "uns": _runtime_schema_entries(dict(adata.uns.items())),
            }
        },
    }


def append_runtime_schema_entry(
    adata,
    *,
    stage: str,
    location: str,
    key: str,
    created_by: str,
    used_structures: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> None:
    """Append a runtime schema entry describing a newly created structure.

    Args:
        adata: AnnData object to annotate.
        stage: Pipeline stage name (e.g. "load", "preprocess").
        location: AnnData slot ("obs", "var", "layers", "obsm", "varm", "obsp", "uns").
        key: Name of the structure within the slot.
        created_by: Function or module responsible for creating the structure.
        used_structures: List of structures consumed to create this structure.
        notes: Optional notes (e.g., first line of a docstring).
    """
    smftools_uns = adata.uns.setdefault("smftools", {})
    runtime_schema = smftools_uns.setdefault(
        "runtime_schema",
        {
            "schema_version": "runtime-1",
            "description": "Runtime AnnData schema annotations (recorded during execution).",
            "generated_at": _iso_timestamp(),
            "stages": {},
        },
    )
    stages = runtime_schema.setdefault("stages", {})
    stage_block = stages.setdefault(stage, {})
    slot_block = stage_block.setdefault(location, {})

    value = None
    if location == "obs" and key in adata.obs:
        value = adata.obs[key]
    elif location == "var" and key in adata.var:
        value = adata.var[key]
    elif location == "layers" and key in adata.layers:
        value = adata.layers[key]
    elif location == "obsm" and key in adata.obsm:
        value = adata.obsm[key]
    elif location == "varm" and key in adata.varm:
        value = adata.varm[key]
    elif location == "obsp" and key in adata.obsp:
        value = adata.obsp[key]
    elif location == "uns" and key in adata.uns:
        value = adata.uns[key]

    slot_block[key] = {
        "dtype": _infer_dtype(value) if value is not None else "unknown",
        "created_by": created_by,
        "used_structures": used_structures or [],
        "notes": notes or "",
        "recorded_at": _iso_timestamp(),
    }


def _format_yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return f'"{str(value)}"'


def _dump_yaml(data: Any, indent: int = 0) -> str:
    space = "  " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_dump_yaml(value, indent + 1))
            else:
                lines.append(f"{space}{key}: {_format_yaml_value(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(_dump_yaml(item, indent + 1))
            else:
                lines.append(f"{space}- {_format_yaml_value(item)}")
        return "\n".join(lines)
    return f"{space}{_format_yaml_value(data)}"


def _schema_sidecar_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".h5ad.gz"):
        base = name[: -len(".h5ad.gz")]
    elif name.endswith(".h5ad"):
        base = name[: -len(".h5ad")]
    else:
        base = output_path.stem
    return output_path.with_name(f"{base}.schema.yaml")


def write_runtime_schema_yaml(adata, output_path: Path, step_name: str) -> Path:
    runtime_schema = adata.uns.get("smftools", {}).get("runtime_schema")
    if isinstance(runtime_schema, dict):
        schema_dict = dict(runtime_schema)
        schema_dict.setdefault("output_path", str(output_path))
        schema_dict.setdefault("generated_at", _iso_timestamp())
        schema_dict.setdefault("schema_version", "runtime-1")
        schema_dict.setdefault(
            "description", "Runtime AnnData schema annotations (recorded during execution)."
        )
    else:
        schema_dict = _runtime_schema_dict(adata, step_name, output_path=output_path)
    yaml_text = _dump_yaml(schema_dict)
    schema_path = _schema_sidecar_path(output_path)
    schema_path.write_text(yaml_text + "\n", encoding="utf-8")
    return schema_path


def _append_unique_inputs(existing: list[dict[str, Any]], new_inputs: list[dict[str, Any]]) -> None:
    seen = {
        (item.get("path"), item.get("hash"), item.get("hash_mode"))
        for item in existing
        if item.get("path")
    }
    for item in new_inputs:
        key = (item.get("path"), item.get("hash"), item.get("hash_mode"))
        if key in seen:
            continue
        existing.append(item)
        seen.add(key)


def record_smftools_metadata(
    adata,
    *,
    step_name: str,
    cfg: Optional[Any] = None,
    config_path: Optional[str | Path] = None,
    input_paths: Optional[Iterable[Path | str]] = None,
    output_path: Optional[Path | str] = None,
    status: str = "ok",
    cli_argv: Optional[list[str]] = None,
) -> None:
    """Record structured smftools metadata into AnnData.uns.

    Args:
        adata: AnnData object to update.
        step_name: Pipeline step name (e.g. "load", "preprocess").
        cfg: Optional ExperimentConfig to capture resolved params.
        config_path: Path to the experiment config file used.
        input_paths: Optional iterable of input artifacts (e.g. h5ad inputs).
        output_path: Optional output path written by this step.
        status: Step status string ("ok" or "failed").
        cli_argv: Optional command argument vector for provenance.
    """
    smftools_uns = adata.uns.setdefault("smftools", {})
    timestamp = _iso_timestamp()

    if "created_by" not in smftools_uns:
        smftools_uns["created_by"] = {
            "version": __version__,
            "time": timestamp,
            "git_commit": _get_git_commit(),
        }

    smftools_uns.setdefault("environment", _environment_snapshot())
    smftools_uns.setdefault("schema_version", "1")
    smftools_uns.setdefault("schema_registry_version", SCHEMA_REGISTRY_VERSION)
    smftools_uns.setdefault(
        "schema_registry_resource",
        f"smftools.schema:{SCHEMA_REGISTRY_RESOURCE}",
    )
    smftools_uns["schema"] = _schema_snapshot(adata)

    provenance = smftools_uns.setdefault("provenance", {})
    inputs = provenance.setdefault("inputs", [])

    input_records: list[dict[str, Any]] = []
    if config_path:
        input_records.append(_path_record(Path(config_path), role="config"))
    if cfg is not None:
        cfg_paths = _normalize_paths(
            [
                cfg.input_data_path,
                cfg.fasta,
                cfg.sample_sheet_path,
                cfg.summary_file,
            ]
        )
        input_records.extend(_path_record(path, role="input") for path in cfg_paths)
        input_records.extend(
            _path_record(path, role="input") for path in _normalize_paths(cfg.input_files)
        )
    if input_paths:
        input_records.extend(_path_record(Path(path), role="input") for path in input_paths)

    _append_unique_inputs(inputs, input_records)

    outputs: dict[str, Any] = {
        "layers": sorted(adata.layers.keys()),
        "obs_columns": sorted(adata.obs.columns),
        "var_columns": sorted(adata.var.columns),
        "uns_keys": sorted(adata.uns.keys()),
        "obsm_keys": sorted(adata.obsm.keys()),
        "obsp_keys": sorted(adata.obsp.keys()),
    }
    if output_path is not None:
        out_path = Path(output_path)
        outputs["h5ad_path"] = str(out_path)
        outputs["schema_yaml_path"] = str(_schema_sidecar_path(out_path))

    runtime = {"device": getattr(cfg, "device", None), "threads": getattr(cfg, "threads", None)}
    if cli_argv is None:
        cli_argv = list(sys.argv)

    step_record = {
        "id": hashlib.sha1(f"{step_name}-{timestamp}".encode("utf-8")).hexdigest(),
        "time": timestamp,
        "step": step_name,
        "smftools_version": __version__,
        "params": cfg.to_dict() if cfg is not None else None,
        "inputs": input_records,
        "outputs": outputs,
        "runtime": runtime,
        "status": status,
        "cli_argv": cli_argv,
    }

    history = smftools_uns.setdefault("history", [])
    history.append(step_record)
