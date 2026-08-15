"""Immutable generation publication for partitioned preprocessing outputs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from ..informatics.experiment_manifest import (
    artifact_record,
    read_experiment_manifest,
)
from ..informatics.generation import (
    CURRENT_FILENAME,
    CURRENT_SCHEMA_VERSION,
    GENERATION_MANIFEST,
    GENERATIONS_SUBDIR,
    STAGING_SUBDIR,
    GenerationError,
    resolve_current_generation,
    staged_generation,
)
from ..informatics.partition_read import load_spine, relative_uns_path
from ..informatics.sidecar_manifest import resolve_sidecar, sidecar_manifest_path
from ..pipeline import PlanState
from ..readwrite import atomic_write_json, safe_write_h5ad
from .partitioned_executor import (
    PREPROCESS_OBS_SIDECAR,
    PREPROCESS_PARTITION_CATALOG,
    PREPROCESS_SPINE_FILENAME,
    PREPROCESS_STAGE_OBS,
    PREPROCESS_STORE_SUBDIR,
    PREPROCESS_TASK_CATALOG,
    PREPROCESS_VAR_CATALOG,
)
from .variant_reporting import VARIANT_REPORTING_SUBDIR

PREPROCESS_GENERATIONS_SUBDIR = GENERATIONS_SUBDIR
PREPROCESS_STAGING_SUBDIR = STAGING_SUBDIR
PREPROCESS_CURRENT_FILENAME = CURRENT_FILENAME
PREPROCESS_GENERATION_MANIFEST = GENERATION_MANIFEST
PREPROCESS_GENERATION_SCHEMA_VERSION = 1
PREPROCESS_CURRENT_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION
PREPROCESS_TASK_CATALOG_SCHEMA_VERSION = 1
PREPROCESS_OUTPUT_SCHEMA_VERSION = 2
PREPROCESS_READ_INDEX_SCHEMA_VERSION = 1

_GENERATION_ARTIFACTS = {
    "spine": PREPROCESS_SPINE_FILENAME,
    "store": PREPROCESS_STORE_SUBDIR,
    "task_catalog": PREPROCESS_TASK_CATALOG,
    "catalog": PREPROCESS_PARTITION_CATALOG,
    "read_index": "read_index",
    "var": PREPROCESS_VAR_CATALOG,
    "obs": PREPROCESS_OBS_SIDECAR,
    "stage_obs": PREPROCESS_STAGE_OBS,
    "plots": "plots",
    "plot_catalog": "plots/catalog.parquet",
    "manifest": "sidecar_manifest.json",
}
_VARIANT_GENERATION_ARTIFACTS = {
    "variant_task_store": f"{VARIANT_REPORTING_SUBDIR}/task_store",
    "variant_task_catalog": f"{VARIANT_REPORTING_SUBDIR}/task_catalog.parquet",
    "variant_obs": f"{VARIANT_REPORTING_SUBDIR}/variant_obs",
    "variant_read_index": f"{VARIANT_REPORTING_SUBDIR}/read_index",
    "variant_reference_catalog": f"{VARIANT_REPORTING_SUBDIR}/reference_catalog.json",
    "variant_generation_manifest": f"{VARIANT_REPORTING_SUBDIR}/generation_manifest.json",
}
_VARIANT_METRIC_ARTIFACTS = {
    "variant_qc_metrics": f"{VARIANT_REPORTING_SUBDIR}/variant_qc_metrics.parquet",
    "variant_qc_summary": f"{VARIANT_REPORTING_SUBDIR}/variant_qc_summary.json",
    "variant_qc_summary_tsv": f"{VARIANT_REPORTING_SUBDIR}/variant_qc_summary.tsv",
}
_REQUIRED_SIDECARS = (
    "preprocess_store",
    "preprocess_catalog",
    "preprocess_task_catalog",
    "preprocess_read_index",
    "preprocess_var",
    "preprocess_obs",
    "preprocess_stage_obs",
    "preprocess_spine",
    "preprocess_plot_catalog",
)
_VARIANT_REQUIRED_SIDECARS = tuple(f"preprocess_{key}" for key in _VARIANT_GENERATION_ARTIFACTS)
_VARIANT_METRIC_SIDECARS = tuple(f"preprocess_{key}" for key in _VARIANT_METRIC_ARTIFACTS)


class PreprocessGenerationError(RuntimeError):
    """Raised when an immutable preprocess generation is unsafe to publish or read."""


def _checksum(path: Path) -> str:
    return str(artifact_record(path, path.parent, checksum=True)["sha256"])


def _generation_artifact_record(path: Path, generation_root: Path) -> dict[str, Any]:
    record = artifact_record(path, generation_root, checksum=True)
    record["anchor"] = "generation_root"
    return record


def _resolve_generation_artifact(
    generation_root: Path,
    record: dict[str, Any],
) -> Path:
    raw_path = record.get("path")
    relative = Path(str(raw_path or ""))
    resolved = (generation_root / relative).resolve()
    if (
        record.get("path_kind") != "relative"
        or record.get("anchor") != "generation_root"
        or not raw_path
        or relative.is_absolute()
        or not resolved.is_relative_to(generation_root.resolve())
    ):
        raise PreprocessGenerationError("preprocess generation artifact path is not portable")
    return resolved


def _source_provenance(spine_path: Path, run_root: Path) -> dict[str, Any]:
    source_stage = {
        "raw_outputs": "raw",
        "load_adata_outputs": "raw",
    }.get(spine_path.parent.name)
    stage_entry = (
        read_experiment_manifest(run_root).get("stages", {}).get(source_stage, {})
        if source_stage is not None
        else {}
    )
    return {
        "artifact": artifact_record(spine_path, run_root, checksum=True),
        "stage": source_stage,
        "generation_id": (
            stage_entry.get("generation_id") if isinstance(stage_entry, dict) else None
        ),
        "config_hash": (stage_entry.get("config_hash") if isinstance(stage_entry, dict) else None),
        "input_artifact_ids": (
            stage_entry.get("input_artifact_ids") if isinstance(stage_entry, dict) else None
        ),
    }


def _atomic_publish_spine(source: Path, destination: Path) -> None:
    spine = load_spine(source, verbose=False)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp.h5ad")
    try:
        safe_write_h5ad(spine, temporary, backup=False, verbose=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _bind_generation_spine(
    spine_path: Path,
    *,
    generation_id: str,
    publication_dir: Path,
    run_root: Path,
) -> None:
    """Bind copied or newly computed spine pointers to a new generation."""
    spine = load_spine(spine_path, verbose=False)
    pointers = {
        "preprocess_store": publication_dir / PREPROCESS_STORE_SUBDIR,
        "preprocess_catalog": publication_dir / PREPROCESS_PARTITION_CATALOG,
        "preprocess_task_catalog": publication_dir / PREPROCESS_TASK_CATALOG,
        "preprocess_read_index": publication_dir / "read_index",
        "preprocess_var": publication_dir / PREPROCESS_VAR_CATALOG,
        "preprocess_obs": publication_dir / PREPROCESS_OBS_SIDECAR,
        "preprocess_stage_obs": publication_dir / PREPROCESS_STAGE_OBS,
        "preprocess_plot_catalog": publication_dir / "plots" / "catalog.parquet",
    }
    if (spine_path.parent / VARIANT_REPORTING_SUBDIR).is_dir():
        pointers.update(
            {
                f"preprocess_{key}": publication_dir / relative
                for key, relative in _VARIANT_GENERATION_ARTIFACTS.items()
            }
        )
    if all(
        (spine_path.parent / relative).exists() for relative in _VARIANT_METRIC_ARTIFACTS.values()
    ):
        pointers.update(
            {
                f"preprocess_{key}": publication_dir / relative
                for key, relative in _VARIANT_METRIC_ARTIFACTS.items()
            }
        )
    for key, path in pointers.items():
        spine.uns[key] = relative_uns_path(path, run_root)
    spine.uns["preprocess_generation_id"] = generation_id
    safe_write_h5ad(spine, spine_path, backup=False, verbose=False)


def _regenerate_preprocess_plots(
    generation_dir: Path,
    source_spine: Path,
    cfg: Any,
) -> None:
    from ..cli.stage_artifacts import prepare_analysis_plot_layout
    from .partitioned_plots import generate_preprocess_summary_plots

    plots = generation_dir / "plots"
    shutil.rmtree(plots, ignore_errors=True)
    layout = prepare_analysis_plot_layout(
        generation_dir,
        stage="preprocess",
        source_spine=source_spine,
    )
    if bool(getattr(cfg, "emit_automated_plots", True)):
        generate_preprocess_summary_plots(
            generation_dir / PREPROCESS_OBS_SIDECAR,
            generation_dir / PREPROCESS_VAR_CATALOG,
            layout,
            cfg=cfg,
            spine_path=generation_dir / PREPROCESS_SPINE_FILENAME,
            task_catalog=generation_dir / PREPROCESS_TASK_CATALOG,
            read_index=generation_dir / "read_index",
        )
        metrics_path = generation_dir / _VARIANT_METRIC_ARTIFACTS["variant_qc_metrics"]
        if metrics_path.is_file():
            from .variant_metrics import generate_variant_qc_plots

            generate_variant_qc_plots(metrics_path, layout)


def _regenerate_variant_metrics(
    generation_dir: Path,
    *,
    source_generation_id: str,
) -> None:
    """Rebuild cohort metrics from compatible evidence and reducer artifacts."""
    from .variant_metrics import write_variant_qc_metric_artifacts

    write_variant_qc_metric_artifacts(
        generation_dir / PREPROCESS_OBS_SIDECAR,
        generation_dir / VARIANT_REPORTING_SUBDIR,
        source_generation_id=source_generation_id,
    )


def validate_preprocess_generation(
    generation_dir: str | Path,
    *,
    expected_generation_id: str | None = None,
    final_dir: str | Path | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one complete preprocess generation without mutating it."""
    generation_dir = Path(generation_dir)
    manifest_path = generation_dir / PREPROCESS_GENERATION_MANIFEST
    if not manifest_path.is_file():
        raise PreprocessGenerationError("preprocess generation manifest is missing")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessGenerationError("preprocess generation manifest is unreadable") from exc
    if int(manifest.get("schema_version", -1)) != PREPROCESS_GENERATION_SCHEMA_VERSION:
        raise PreprocessGenerationError("preprocess generation schema is incompatible")
    if manifest.get("status") != "complete":
        raise PreprocessGenerationError("preprocess generation is not complete")
    generation_id = str(manifest.get("generation_id", ""))
    if not generation_id or (
        expected_generation_id is not None and generation_id != expected_generation_id
    ):
        raise PreprocessGenerationError("preprocess generation ID does not match")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PreprocessGenerationError("preprocess generation artifact manifest is missing")
    variant_artifact_keys = set(_VARIANT_GENERATION_ARTIFACTS).intersection(artifacts)
    if variant_artifact_keys and variant_artifact_keys != set(_VARIANT_GENERATION_ARTIFACTS):
        raise PreprocessGenerationError("preprocess variant artifacts are incomplete")
    expected_artifacts = dict(_GENERATION_ARTIFACTS)
    if variant_artifact_keys:
        expected_artifacts.update(_VARIANT_GENERATION_ARTIFACTS)
    variant_metric_keys = set(_VARIANT_METRIC_ARTIFACTS).intersection(artifacts)
    if variant_metric_keys and variant_metric_keys != set(_VARIANT_METRIC_ARTIFACTS):
        raise PreprocessGenerationError("preprocess variant metric artifacts are incomplete")
    if variant_metric_keys and not variant_artifact_keys:
        raise PreprocessGenerationError("preprocess variant metrics lack evidence artifacts")
    if variant_metric_keys:
        expected_artifacts.update(_VARIANT_METRIC_ARTIFACTS)
    for key, expected_relative in expected_artifacts.items():
        record = artifacts.get(key)
        if not isinstance(record, dict):
            raise PreprocessGenerationError(f"preprocess generation artifact is missing: {key}")
        path = _resolve_generation_artifact(generation_dir, record)
        if Path(str(record.get("path"))) != Path(expected_relative):
            raise PreprocessGenerationError(
                f"preprocess generation artifact path is invalid: {key}"
            )
        if not path.exists() or str(record.get("sha256", "")) != _checksum(path):
            raise PreprocessGenerationError(
                f"preprocess generation artifact is missing or corrupt: {key}"
            )
        expected_kind = record.get("kind")
        if expected_kind == "file" and not path.is_file():
            raise PreprocessGenerationError(f"preprocess generation artifact is not a file: {key}")
        if expected_kind == "directory" and (
            not path.is_dir() or (key in {"store", "read_index"} and not any(path.iterdir()))
        ):
            raise PreprocessGenerationError(
                f"preprocess generation artifact directory is invalid: {key}"
            )

    if variant_metric_keys:
        from ..constants import VARIANT_QC_METRICS_SCHEMA_VERSION

        metric_columns = {
            "schema_version",
            "analysis_version",
            "source_generation_id",
            "variant_reference_set_id",
            "cohort",
            "grouping",
            "reference",
            "sample",
            "level",
            "measure",
            "numerator",
            "denominator",
            "value",
        }
        metrics = pd.read_parquet(generation_dir / _VARIANT_METRIC_ARTIFACTS["variant_qc_metrics"])
        if not metric_columns.issubset(metrics.columns):
            raise PreprocessGenerationError("preprocess variant metric schema is incomplete")
        source_generation_ids = set(metrics["source_generation_id"].dropna().astype(str))
        if (
            set(metrics["schema_version"].dropna().astype(int))
            != {VARIANT_QC_METRICS_SCHEMA_VERSION}
            or len(source_generation_ids) != 1
        ):
            raise PreprocessGenerationError("preprocess variant metric provenance is invalid")
        metric_source_generation_id = next(iter(source_generation_ids))
        summary_path = generation_dir / _VARIANT_METRIC_ARTIFACTS["variant_qc_summary"]
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreprocessGenerationError(
                "preprocess variant metric summary is unreadable"
            ) from exc
        if (
            int(summary.get("schema_version", -1)) != VARIANT_QC_METRICS_SCHEMA_VERSION
            or str(summary.get("source_generation_id", "")) != metric_source_generation_id
        ):
            raise PreprocessGenerationError(
                "preprocess variant metric summary provenance is invalid"
            )
        compact = pd.read_csv(
            generation_dir / _VARIANT_METRIC_ARTIFACTS["variant_qc_summary_tsv"],
            sep="\t",
        )
        if not metric_columns.issubset(compact.columns):
            raise PreprocessGenerationError("preprocess variant TSV summary schema is incomplete")

    if "node_results" in manifest:
        from .semantic_upgrade import (
            PREPROCESS_PLOTS_NODE,
            PREPROCESS_REDUCERS_NODE,
            PREPROCESS_TASKS_NODE,
            PREPROCESS_VARIANT_EVIDENCE_NODE,
            PREPROCESS_VARIANT_METRICS_NODE,
            PREPROCESS_VARIANT_REFERENCE_NODE,
            load_preprocess_node_results,
            preprocess_registry,
        )

        try:
            node_results = load_preprocess_node_results(manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessGenerationError(
                "preprocess generation node results are malformed"
            ) from exc
        expected_nodes = {
            PREPROCESS_TASKS_NODE,
            PREPROCESS_REDUCERS_NODE,
            PREPROCESS_PLOTS_NODE,
        }
        variant_enabled = bool(variant_artifact_keys)
        if variant_enabled:
            expected_nodes.update(
                {PREPROCESS_VARIANT_REFERENCE_NODE, PREPROCESS_VARIANT_EVIDENCE_NODE}
            )
        if variant_metric_keys:
            expected_nodes.add(PREPROCESS_VARIANT_METRICS_NODE)
        if set(node_results) != expected_nodes:
            raise PreprocessGenerationError("preprocess generation node results are incomplete")
        registry = preprocess_registry(generation_dir, variant_enabled=variant_enabled)
        for analysis_id, result in node_results.items():
            validation = registry.validator_for(registry.node(analysis_id))(result)
            if not validation.valid:
                raise PreprocessGenerationError(
                    f"preprocess generation node result is invalid: {analysis_id}: "
                    f"{validation.reason}"
                )

    task_catalog = pd.read_parquet(generation_dir / PREPROCESS_TASK_CATALOG)
    result_catalog = pd.read_parquet(generation_dir / PREPROCESS_PARTITION_CATALOG)
    task_count = int(manifest.get("task_count", -1))
    if task_count <= 0 or len(task_catalog) != task_count or len(result_catalog) != task_count:
        raise PreprocessGenerationError("preprocess generation task counts do not match")
    for record in result_catalog.to_dict("records"):
        relative_group = Path(str(record.get("group_path", "")))
        if (
            not str(relative_group)
            or relative_group.is_absolute()
            or ".." in relative_group.parts
            or not (generation_dir / relative_group).is_dir()
        ):
            raise PreprocessGenerationError("preprocess task catalog has an invalid group path")

    final_dir = Path(final_dir) if final_dir is not None else generation_dir
    run_root = Path(run_root) if run_root is not None else final_dir.parents[2]
    spine = load_spine(generation_dir / PREPROCESS_SPINE_FILENAME, verbose=False)
    expected_pointers = {
        "preprocess_store": final_dir / PREPROCESS_STORE_SUBDIR,
        "preprocess_catalog": final_dir / PREPROCESS_PARTITION_CATALOG,
        "preprocess_var": final_dir / PREPROCESS_VAR_CATALOG,
        "preprocess_obs": final_dir / PREPROCESS_OBS_SIDECAR,
        "preprocess_read_index": final_dir / "read_index",
        "preprocess_stage_obs": final_dir / PREPROCESS_STAGE_OBS,
        "preprocess_task_catalog": final_dir / PREPROCESS_TASK_CATALOG,
        "preprocess_plot_catalog": final_dir / "plots" / "catalog.parquet",
    }
    if variant_artifact_keys:
        expected_pointers.update(
            {
                f"preprocess_{key}": final_dir / relative
                for key, relative in _VARIANT_GENERATION_ARTIFACTS.items()
            }
        )
    if variant_metric_keys:
        expected_pointers.update(
            {
                f"preprocess_{key}": final_dir / relative
                for key, relative in _VARIANT_METRIC_ARTIFACTS.items()
            }
        )
    for key, path in expected_pointers.items():
        if spine.uns.get(key) != relative_uns_path(path, run_root):
            raise PreprocessGenerationError(f"preprocess spine pointer is unsafe: {key}")
    if str(spine.uns.get("preprocess_generation_id", "")) != generation_id:
        raise PreprocessGenerationError("preprocess spine generation ID does not match")

    sidecars = sidecar_manifest_path(generation_dir)
    missing_sidecars = [key for key in _REQUIRED_SIDECARS if resolve_sidecar(sidecars, key) is None]
    if variant_artifact_keys:
        missing_sidecars.extend(
            key for key in _VARIANT_REQUIRED_SIDECARS if resolve_sidecar(sidecars, key) is None
        )
    if variant_metric_keys:
        missing_sidecars.extend(
            key for key in _VARIANT_METRIC_SIDECARS if resolve_sidecar(sidecars, key) is None
        )
    if missing_sidecars:
        raise PreprocessGenerationError(
            f"preprocess sidecar manifest is incomplete: {missing_sidecars}"
        )
    return manifest


def resolve_current_preprocess_generation(
    output_dir: str | Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve and validate the generation selected by preprocess ``current.json``."""
    output_dir = Path(output_dir)
    try:
        selected = resolve_current_generation(
            output_dir,
            manifest_checksum=_checksum,
            require_generation_id=True,
        )
    except GenerationError as exc:
        raise PreprocessGenerationError(str(exc)) from exc
    if selected is None:
        return None
    generation, pointer_manifest = selected
    manifest = validate_preprocess_generation(
        generation,
        expected_generation_id=str(pointer_manifest.get("generation_id", "")),
        final_dir=generation,
        run_root=output_dir.parent,
    )
    return generation, manifest


def publish_preprocess_generation(
    spine_path: str | Path,
    cfg: Any,
    output_dir: str | Path,
    *,
    executor: Callable[..., dict[str, Path]] | None = None,
) -> dict[str, Path | str | int]:
    """Build, validate, and atomically select one immutable preprocess generation."""
    from ..cli.helpers import stage_config_hash
    from ..informatics.experiment_spine import write_experiment_spine
    from .partitioned_executor import execute_partitioned_preprocessing
    from .semantic_upgrade import (
        PREPROCESS_PLOTS_NODE,
        PREPROCESS_REDUCERS_NODE,
        PREPROCESS_TASKS_NODE,
        PREPROCESS_VARIANT_METRICS_NODE,
        build_preprocess_node_results,
        plan_preprocess_upgrade,
        preprocess_force_targets,
    )

    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    run_root = output_dir.parent
    canonical_spine = output_dir / PREPROCESS_SPINE_FILENAME
    current_generation = resolve_current_preprocess_generation(output_dir)
    upgrade_plan = plan_preprocess_upgrade(
        spine_path,
        cfg,
        output_dir,
        current_generation=current_generation,
        force_targets=preprocess_force_targets(cfg),
    )
    decisions = {decision.analysis_id: decision for decision in upgrade_plan.decisions}
    reusable = {
        analysis_id
        for analysis_id, decision in decisions.items()
        if decision.state is PlanState.COMPATIBLE
    }
    previous_generation_id = (
        str(current_generation[1]["generation_id"]) if current_generation is not None else None
    )

    def validate(staging: Path, final: Path, root: Path) -> None:
        validate_preprocess_generation(
            staging,
            expected_generation_id=staged.generation_id,
            final_dir=final,
            run_root=root,
        )

    def publish_spine(_staging: Path, final: Path, _root: Path) -> None:
        _atomic_publish_spine(final / PREPROCESS_SPINE_FILENAME, canonical_spine)

    try:
        with staged_generation(
            output_dir,
            run_root=run_root,
            validate=validate,
            manifest_checksum=_checksum,
            write_json=atomic_write_json,
            after_current=publish_spine,
        ) as staged:
            generation_id = staged.generation_id
            staging_dir = staged.staging_dir
            final_dir = staged.final_dir
            compute_nodes = {PREPROCESS_TASKS_NODE, PREPROCESS_REDUCERS_NODE}
            if current_generation is not None and compute_nodes.issubset(reusable):
                shutil.copytree(current_generation[0], staging_dir, dirs_exist_ok=True)
                (staging_dir / PREPROCESS_GENERATION_MANIFEST).unlink(missing_ok=True)
                if (
                    str(getattr(cfg, "variant_analysis_mode", "off")).lower()
                    in {"report", "filter"}
                    and PREPROCESS_VARIANT_METRICS_NODE not in reusable
                ):
                    _regenerate_variant_metrics(
                        staging_dir,
                        source_generation_id=generation_id,
                    )
                if PREPROCESS_PLOTS_NODE not in reusable:
                    _regenerate_preprocess_plots(staging_dir, spine_path, cfg)
                staged_spine = staging_dir / PREPROCESS_SPINE_FILENAME
            else:
                execute = executor or execute_partitioned_preprocessing
                execute_kwargs: dict[str, Any] = {
                    "publication_dir": final_dir,
                    "run_root": run_root,
                    "refresh_experiment_spine": False,
                }
                if executor is None:
                    execute_kwargs["analysis_generation_id"] = generation_id
                if (
                    current_generation is not None
                    and PREPROCESS_TASKS_NODE in reusable
                    and executor is None
                ):
                    execute_kwargs["reuse_task_artifacts_from"] = current_generation[0]
                staged_outputs = execute(
                    spine_path,
                    cfg,
                    staging_dir,
                    **execute_kwargs,
                )
                staged_spine = Path(staged_outputs["spine"])
            _bind_generation_spine(
                staged_spine,
                generation_id=generation_id,
                publication_dir=final_dir,
                run_root=run_root,
            )

            artifact_paths = dict(_GENERATION_ARTIFACTS)
            if (staging_dir / VARIANT_REPORTING_SUBDIR).is_dir():
                artifact_paths.update(_VARIANT_GENERATION_ARTIFACTS)
            if all(
                (staging_dir / relative).exists() for relative in _VARIANT_METRIC_ARTIFACTS.values()
            ):
                artifact_paths.update(_VARIANT_METRIC_ARTIFACTS)
            artifacts = {
                key: _generation_artifact_record(staging_dir / relative, staging_dir)
                for key, relative in artifact_paths.items()
            }
            task_count = len(pd.read_parquet(staging_dir / PREPROCESS_TASK_CATALOG))
            reused_nodes = (
                reusable
                if current_generation is not None and compute_nodes.issubset(reusable)
                else reusable.intersection({PREPROCESS_TASKS_NODE})
            )
            node_results = build_preprocess_node_results(
                staging_dir,
                spine_path,
                cfg,
                generation_id=generation_id,
                reused_nodes=reused_nodes,
                reused_from_generation_id=previous_generation_id,
            )
            staged.record_manifest(
                {
                    "schema_version": PREPROCESS_GENERATION_SCHEMA_VERSION,
                    "status": "complete",
                    "generation_id": generation_id,
                    "compute_config_hash": stage_config_hash(cfg, "preprocess"),
                    "source": _source_provenance(spine_path, run_root),
                    "output_schema_version": PREPROCESS_OUTPUT_SCHEMA_VERSION,
                    "task_catalog_schema_version": PREPROCESS_TASK_CATALOG_SCHEMA_VERSION,
                    "read_index_schema_version": PREPROCESS_READ_INDEX_SCHEMA_VERSION,
                    "task_count": task_count,
                    "upgrade_plan": upgrade_plan.to_dict(),
                    "node_results": [result.to_dict() for result in node_results],
                    "artifacts": artifacts,
                }
            )
    except GenerationError as exc:
        raise PreprocessGenerationError(str(exc)) from exc

    write_experiment_spine(run_root)
    current_path = output_dir / PREPROCESS_CURRENT_FILENAME

    outputs: dict[str, Path | str | int] = {
        key: final_dir / relative for key, relative in _GENERATION_ARTIFACTS.items()
    }
    if (final_dir / VARIANT_REPORTING_SUBDIR).is_dir():
        outputs.update(
            {key: final_dir / relative for key, relative in _VARIANT_GENERATION_ARTIFACTS.items()}
        )
    if all((final_dir / relative).exists() for relative in _VARIANT_METRIC_ARTIFACTS.values()):
        outputs.update(
            {key: final_dir / relative for key, relative in _VARIANT_METRIC_ARTIFACTS.items()}
        )
    outputs.update(
        {
            "spine": canonical_spine,
            "generation_spine": final_dir / PREPROCESS_SPINE_FILENAME,
            "generation_manifest": final_dir / PREPROCESS_GENERATION_MANIFEST,
            "generation": final_dir,
            "current": current_path,
            "generation_id": generation_id,
            "task_count": task_count,
        }
    )
    return outputs
