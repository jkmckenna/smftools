"""Resolve installed Dorado capabilities and immutable model-bundle identity."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..optional_imports import require
from .raw_intermediate_manifest import RawIntermediateError, artifact_checksum

_FLAG_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])(--[a-z0-9][a-z0-9-]*)")
_MODEL_ALIAS_PATTERN = re.compile(r"^(fast|hac|sup)(?:@(latest|v[0-9][0-9.]*))?$")
_VERSION_PATTERN = re.compile(r"(?:^|\s)(\d+\.\d+\.\d+(?:[+._-][A-Za-z0-9._-]+)?)")


class DoradoModelError(RuntimeError):
    """Raised when an installed Dorado/model identity cannot be resolved."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class DoradoBasecallOptions:
    """Normalized scientific and output options for one Dorado invocation."""

    model: str
    modified_bases: tuple[str, ...] = ()
    read_splitting: str = "preserve"
    trim: str = "none"
    emit_moves: bool = True
    min_qscore: float = 0.0
    barcode_kit: str | None = None
    barcode_both_ends: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("Dorado model selector must not be empty")
        if self.read_splitting not in {"preserve", "disable"}:
            raise ValueError("Dorado read_splitting must be 'preserve' or 'disable'")
        if self.trim not in {"none", "all"}:
            raise ValueError("Dorado trim must be 'none' or 'all'")
        if not math.isfinite(float(self.min_qscore)) or float(self.min_qscore) < 0:
            raise ValueError("Dorado min_qscore must be finite and nonnegative")
        if len(set(self.modified_bases)) != len(self.modified_bases):
            raise ValueError("Dorado modified_bases must be unique")
        if self.barcode_both_ends and not self.barcode_kit:
            raise ValueError("Dorado barcode_both_ends requires barcode_kit")

    def semantic_payload(self) -> dict[str, object]:
        """Return path-neutral options that may affect basecall outputs."""
        return {
            "model": self.model,
            "modified_bases": list(self.modified_bases),
            "read_splitting": self.read_splitting,
            "trim": self.trim,
            "emit_moves": self.emit_moves,
            "min_qscore": float(self.min_qscore),
            "barcode_kit": self.barcode_kit,
            "barcode_both_ends": self.barcode_both_ends,
        }


@dataclass(frozen=True)
class DoradoRunCondition:
    """POD5 run metadata used to select one Dorado chemistry catalog entry."""

    flow_cell: str
    kit: str
    sampling_rate: int

    def to_dict(self) -> dict[str, object]:
        return {
            "flow_cell": self.flow_cell,
            "kit": self.kit,
            "sampling_rate": self.sampling_rate,
        }


@dataclass(frozen=True)
class DoradoModelArtifact:
    """Content identity for one installed Dorado model directory."""

    name: str
    sha256: str
    file_count: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class DoradoBasecallResolution:
    """One path-neutral resolved Dorado executable and model-bundle plan."""

    selector: str
    dorado_version: str
    chemistry: str
    run_conditions: tuple[DoradoRunCondition, ...]
    simplex_model: DoradoModelArtifact
    modification_models: tuple[DoradoModelArtifact, ...]
    model_bundle_digest: str
    supported_flags: tuple[str, ...]
    capability_digest: str
    options: DoradoBasecallOptions
    normalized_argv: tuple[str, ...]
    executable_path: Path
    model_directory: Path
    simplex_path: Path
    modification_paths: tuple[Path, ...]

    def semantic_payload(self) -> dict[str, object]:
        """Return the exact path-independent reuse identity."""
        return {
            "selector": self.selector,
            "dorado_version": self.dorado_version,
            "chemistry": self.chemistry,
            "run_conditions": [condition.to_dict() for condition in self.run_conditions],
            "simplex_model": self.simplex_model.to_dict(),
            "modification_models": [model.to_dict() for model in self.modification_models],
            "model_bundle_digest": self.model_bundle_digest,
            "capability_digest": self.capability_digest,
            "options": self.options.semantic_payload(),
            "normalized_argv": list(self.normalized_argv),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "resolution_status": "resolved",
            **self.semantic_payload(),
            "supported_flags": list(self.supported_flags),
        }


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_probe(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            list(command),
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise DoradoModelError(
            "dorado_probe_failed",
            f"could not run Dorado capability probe: {type(exc).__name__}: {exc}",
        ) from exc


def _probe_dorado(
    executable: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    executable_finder: Callable[[str], str | None],
) -> tuple[Path, str, tuple[str, ...], Mapping[str, object]]:
    raw_executable = str(executable)
    located = (
        str(Path(raw_executable).expanduser().resolve(strict=False))
        if Path(raw_executable).expanduser().is_absolute()
        else executable_finder(raw_executable)
    )
    if located is None or not Path(located).is_file():
        raise DoradoModelError(
            "dorado_executable_unavailable",
            f"Dorado executable {raw_executable!r} is unavailable",
        )
    executable_path = Path(located).resolve()

    version_result = _run_probe((str(executable_path), "--version"), runner=runner)
    version_text = (version_result.stdout or version_result.stderr or "").strip()
    version_match = _VERSION_PATTERN.search(version_text)
    if version_result.returncode != 0 or version_match is None:
        raise DoradoModelError(
            "dorado_version_unavailable",
            "installed Dorado did not report a parseable semantic version",
        )
    version = version_match.group(1)

    help_result = _run_probe(
        (str(executable_path), "basecaller", "--help"),
        runner=runner,
    )
    help_text = "\n".join((help_result.stdout or "", help_result.stderr or ""))
    if help_result.returncode != 0 or not help_text.strip():
        raise DoradoModelError(
            "dorado_capabilities_unavailable",
            "installed Dorado did not expose basecaller help",
        )
    supported_flags = tuple(sorted(set(_FLAG_PATTERN.findall(help_text))))

    catalog_result = _run_probe(
        (str(executable_path), "download", "--list-structured"),
        runner=runner,
    )
    try:
        catalog = json.loads(catalog_result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DoradoModelError(
            "dorado_model_catalog_unavailable",
            "installed Dorado returned an invalid structured model catalog",
        ) from exc
    if catalog_result.returncode != 0 or not isinstance(catalog, Mapping) or not catalog:
        raise DoradoModelError(
            "dorado_model_catalog_unavailable",
            "installed Dorado returned no structured model catalog",
        )
    return executable_path, version, supported_flags, catalog


def read_pod5_run_conditions(source_paths: Sequence[Path]) -> tuple[DoradoRunCondition, ...]:
    """Read distinct model-selection conditions from POD5 run-info tables."""
    pod5 = require("pod5", extra="ont", purpose="Dorado model resolution")
    conditions: set[DoradoRunCondition] = set()
    for source_path in source_paths:
        try:
            with pod5.Reader(Path(source_path)) as reader:
                run_rows = reader.run_info_table.read_all().to_pylist()
        except Exception as exc:
            raise DoradoModelError(
                "dorado_chemistry_unavailable",
                f"could not read POD5 run metadata from source {Path(source_path).name!r}",
            ) from exc
        for row in run_rows:
            context_tags = dict(row.get("context_tags") or ())
            flow_cell = str(row.get("flow_cell_product_code") or "").strip().upper()
            kit = (
                str(
                    row.get("sequencing_kit")
                    or context_tags.get("sequencing_kit")
                    or context_tags.get("barcoding_kits")
                    or ""
                )
                .strip()
                .upper()
            )
            sampling_rate_raw = row.get("sample_rate") or context_tags.get("sample_frequency")
            try:
                sampling_rate = int(sampling_rate_raw)
            except (TypeError, ValueError) as exc:
                raise DoradoModelError(
                    "dorado_chemistry_unavailable",
                    "POD5 run metadata has no valid sampling rate",
                ) from exc
            if not flow_cell or not kit:
                raise DoradoModelError(
                    "dorado_chemistry_unavailable",
                    "POD5 run metadata lacks flow-cell or sequencing-kit identity",
                )
            conditions.add(DoradoRunCondition(flow_cell, kit, sampling_rate))
    if not conditions:
        raise DoradoModelError(
            "dorado_chemistry_unavailable",
            "resolved POD5 sources contain no run metadata",
        )
    return tuple(
        sorted(conditions, key=lambda item: (item.flow_cell, item.kit, item.sampling_rate))
    )


def _chemistry_for_condition(
    catalog: Mapping[str, object],
    condition: DoradoRunCondition,
) -> str:
    matches = []
    for chemistry, raw_record in catalog.items():
        if not isinstance(raw_record, Mapping):
            continue
        raw_flow_cells = raw_record.get("flowcells")
        raw_kits = raw_record.get("kits")
        if not isinstance(raw_flow_cells, (list, tuple)) or not isinstance(raw_kits, (list, tuple)):
            continue
        flow_cells = {str(value).upper() for value in raw_flow_cells}
        kits = {str(value).upper() for value in raw_kits}
        try:
            sampling_rate = int(raw_record.get("sampling_rate"))
        except (TypeError, ValueError):
            continue
        if (
            condition.flow_cell in flow_cells
            and condition.kit in kits
            and condition.sampling_rate == sampling_rate
        ):
            matches.append(str(chemistry))
    if len(matches) != 1:
        qualifier = "no" if not matches else "multiple"
        raise DoradoModelError(
            "dorado_chemistry_unavailable" if not matches else "dorado_chemistry_ambiguous",
            f"POD5 condition {condition.to_dict()} matches {qualifier} Dorado chemistry entries",
        )
    return matches[0]


def _model_version_key(model_name: str) -> tuple[int, ...]:
    match = re.search(r"@v([0-9]+(?:\.[0-9]+)*)$", model_name)
    return () if match is None else tuple(int(part) for part in match.group(1).split("."))


def _resolve_simplex_name(
    selector: str,
    chemistry: str,
    chemistry_record: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    raw_models = chemistry_record.get("simplex_models")
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise DoradoModelError(
            "dorado_simplex_model_unavailable",
            f"Dorado catalog has no simplex models for chemistry {chemistry!r}",
        )
    models = {
        str(name): record for name, record in raw_models.items() if isinstance(record, Mapping)
    }
    if selector in models:
        return selector, models[selector]

    alias_match = _MODEL_ALIAS_PATTERN.fullmatch(selector)
    if alias_match is None:
        raise DoradoModelError(
            "dorado_model_selector_unsupported",
            f"Dorado model selector {selector!r} is neither an alias nor a catalog model name",
        )
    variant, requested_version = alias_match.groups()
    candidates = [
        (name, record) for name, record in models.items() if str(record.get("variant")) == variant
    ]
    if requested_version not in {None, "latest"}:
        candidates = [
            (name, record) for name, record in candidates if name.endswith(f"@{requested_version}")
        ]
    else:
        current = [(name, record) for name, record in candidates if not record.get("outdated")]
        candidates = current or candidates
    if not candidates:
        raise DoradoModelError(
            "dorado_simplex_model_unavailable",
            f"no Dorado simplex model satisfies {selector!r} for chemistry {chemistry!r}",
        )
    return max(candidates, key=lambda item: (_model_version_key(item[0]), item[0]))


def _resolve_modification_names(
    simplex_name: str,
    simplex_record: Mapping[str, object],
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    raw_models = simplex_record.get("modified_models", {})
    models = (
        {str(name): record for name, record in raw_models.items() if isinstance(record, Mapping)}
        if isinstance(raw_models, Mapping)
        else {}
    )
    resolved = []
    for variant in requested:
        candidates = [
            name for name, record in models.items() if str(record.get("variant")) == variant
        ]
        if not candidates:
            raise DoradoModelError(
                "dorado_modification_model_unavailable",
                f"simplex model {simplex_name!r} has no compatible {variant!r} model",
            )
        resolved.append(max(candidates, key=lambda name: (_model_version_key(name), name)))
    return tuple(resolved)


def _model_artifact(model_directory: Path, name: str) -> tuple[DoradoModelArtifact, Path]:
    path = model_directory / name
    if not path.is_dir():
        raise DoradoModelError(
            "dorado_model_not_installed",
            f"resolved Dorado model {name!r} is not installed in the configured model directory",
        )
    try:
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise DoradoModelError(
                "dorado_model_invalid",
                f"installed Dorado model {name!r} contains no files",
            )
        size_bytes = sum(item.stat().st_size for item in files)
        digest = artifact_checksum(path)
    except DoradoModelError:
        raise
    except RawIntermediateError as exc:
        raise DoradoModelError("dorado_model_invalid", str(exc)) from exc
    except OSError as exc:
        raise DoradoModelError(
            "dorado_model_invalid",
            f"could not inspect installed Dorado model {name!r}: {exc}",
        ) from exc
    return (
        DoradoModelArtifact(
            name=name,
            sha256=digest,
            file_count=len(files),
            size_bytes=size_bytes,
        ),
        path.resolve(),
    )


def _number_text(value: float) -> str:
    return format(float(value), ".15g")


def _basecaller_arguments(
    options: DoradoBasecallOptions,
    supported_flags: set[str],
    *,
    executable: str,
    simplex: str,
    modifications: tuple[str, ...],
    data: str,
    read_ids: str,
    output_directory: str,
) -> tuple[str, ...]:
    arguments = [
        executable,
        "basecaller",
        "--device",
        options.device,
        "--batchsize",
        "0",
        "--read-ids",
        read_ids,
        "--min-qscore",
        _number_text(options.min_qscore),
        "--emit-summary",
        "--output-dir",
        output_directory,
    ]
    if options.read_splitting == "disable":
        arguments.append("--disable-read-splitting")
    if "--trim" in supported_flags:
        arguments.extend(("--trim", options.trim))
    elif options.trim == "none":
        arguments.append("--no-trim")
    if options.emit_moves:
        arguments.append("--emit-moves")
    if options.barcode_kit:
        arguments.extend(("--kit-name", options.barcode_kit))
    if options.barcode_both_ends:
        arguments.append("--barcode-both-ends")
    if modifications:
        arguments.extend(("--modified-bases-models", ",".join(modifications)))
    arguments.extend((simplex, data))
    return tuple(arguments)


def _required_flags(options: DoradoBasecallOptions, supported_flags: set[str]) -> set[str]:
    required = {
        "--batchsize",
        "--device",
        "--emit-summary",
        "--min-qscore",
        "--output-dir",
        "--read-ids",
    }
    if "--trim" not in supported_flags and options.trim == "none":
        required.add("--no-trim")
    if options.read_splitting == "disable":
        required.add("--disable-read-splitting")
    if options.emit_moves:
        required.add("--emit-moves")
    if options.modified_bases:
        required.add("--modified-bases-models")
    if options.barcode_kit:
        required.add("--kit-name")
    if options.barcode_both_ends:
        required.add("--barcode-both-ends")
    return required


def resolve_dorado_basecall(
    options: DoradoBasecallOptions,
    source_paths: Sequence[str | Path],
    model_directory: str | Path | None,
    *,
    executable: str | Path = "dorado",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    executable_finder: Callable[[str], str | None] = shutil.which,
    condition_reader: Callable[[Sequence[Path]], tuple[DoradoRunCondition, ...]] = (
        read_pod5_run_conditions
    ),
) -> DoradoBasecallResolution:
    """Resolve exact installed models and a supported path-neutral invocation."""
    if model_directory is None:
        raise DoradoModelError(
            "dorado_model_directory_unavailable",
            "no Dorado model directory is configured",
        )
    model_directory = Path(model_directory).expanduser().resolve(strict=False)
    if not model_directory.is_dir():
        raise DoradoModelError(
            "dorado_model_directory_unavailable",
            "the configured Dorado model directory is unavailable",
        )
    normalized_sources = tuple(Path(path).resolve() for path in source_paths)
    if not normalized_sources:
        raise DoradoModelError(
            "dorado_chemistry_unavailable",
            "no resolved POD5 sources are available for Dorado model selection",
        )

    executable_path, version, flags, catalog = _probe_dorado(
        executable,
        runner=runner,
        executable_finder=executable_finder,
    )
    supported_flags = set(flags)
    missing_flags = sorted(_required_flags(options, supported_flags).difference(supported_flags))
    if missing_flags:
        raise DoradoModelError(
            "dorado_capability_missing",
            f"installed Dorado lacks required basecaller flag(s): {', '.join(missing_flags)}",
        )

    conditions = condition_reader(normalized_sources)
    if not conditions:
        raise DoradoModelError(
            "dorado_chemistry_unavailable",
            "resolved POD5 sources contain no run metadata",
        )
    chemistries = {_chemistry_for_condition(catalog, condition) for condition in conditions}
    if len(chemistries) != 1:
        raise DoradoModelError(
            "dorado_chemistry_ambiguous",
            f"resolved POD5 sources require multiple Dorado chemistries: {sorted(chemistries)}",
        )
    chemistry = next(iter(chemistries))
    chemistry_record = catalog[chemistry]
    assert isinstance(chemistry_record, Mapping)
    simplex_name, simplex_record = _resolve_simplex_name(
        options.model,
        chemistry,
        chemistry_record,
    )
    modification_names = _resolve_modification_names(
        simplex_name,
        simplex_record,
        options.modified_bases,
    )
    simplex_artifact, simplex_path = _model_artifact(model_directory, simplex_name)
    modification_pairs = tuple(
        _model_artifact(model_directory, name) for name in modification_names
    )
    modification_artifacts = tuple(pair[0] for pair in modification_pairs)
    modification_paths = tuple(pair[1] for pair in modification_pairs)
    bundle_payload = {
        "simplex_model": simplex_artifact.to_dict(),
        "modification_models": [model.to_dict() for model in modification_artifacts],
    }
    bundle_digest = _json_digest(bundle_payload)
    capability_digest = _json_digest({"dorado_version": version, "supported_flags": list(flags)})
    normalized_argv = _basecaller_arguments(
        options,
        supported_flags,
        executable="dorado",
        simplex=simplex_name,
        modifications=modification_names,
        data="<signal-input>",
        read_ids="<read-ids>",
        output_directory="<output-directory>",
    )
    return DoradoBasecallResolution(
        selector=options.model,
        dorado_version=version,
        chemistry=chemistry,
        run_conditions=conditions,
        simplex_model=simplex_artifact,
        modification_models=modification_artifacts,
        model_bundle_digest=bundle_digest,
        supported_flags=flags,
        capability_digest=capability_digest,
        options=options,
        normalized_argv=normalized_argv,
        executable_path=executable_path,
        model_directory=model_directory,
        simplex_path=simplex_path,
        modification_paths=modification_paths,
    )


def build_dorado_basecaller_argv(
    resolution: DoradoBasecallResolution,
    data_path: str | Path,
    read_ids_path: str | Path,
    output_directory: str | Path,
) -> tuple[str, ...]:
    """Build the exact supported Dorado argv for a resolved immutable bundle."""
    return _basecaller_arguments(
        resolution.options,
        set(resolution.supported_flags),
        executable=str(resolution.executable_path),
        simplex=str(resolution.simplex_path),
        modifications=tuple(map(str, resolution.modification_paths)),
        data=str(Path(data_path)),
        read_ids=str(Path(read_ids_path)),
        output_directory=str(Path(output_directory)),
    )
