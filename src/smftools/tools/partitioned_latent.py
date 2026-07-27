"""Bounded latent-representation fitting over partitioned experiment spines.

Each output group owns an independent coordinate system for one reference locus or
genome core. Embeddings from different groups must not be compared as though their
component axes were shared.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import pandas as pd

from smftools.constants import LATENT_DIR, REFERENCE_STRAND, SEQUENCE_INTEGER_ENCODING
from smftools.latent_resource import (
    LATENT_RESOURCE_ESTIMATOR_VERSION,
    LatentResourceError,
    resolve_latent_operation,
    resource_envelope_id,
)
from smftools.logging_utils import get_logger

from ..cli.latent_adata import (
    _build_mod_sites_var_filter_mask,
    _build_reference_position_mask,
    _build_shared_valid_non_mod_sites_mask,
)
from ..cli.stage_artifacts import (
    PLOT_CATALOG_COLUMNS,
    prepare_analysis_plot_layout,
    register_plot_artifact,
)
from ..informatics.analysis_region_plan import plan_analysis_cores
from ..informatics.experiment_manifest import artifact_record, read_experiment_manifest
from ..informatics.partition_read import (
    load_spine,
    materialize,
    relative_uns_path,
    resolve_relative_path,
)
from ..informatics.sidecar_manifest import (
    register_sidecar,
    resolve_sidecar,
    sidecar_manifest_path,
)
from ..memory_guard import process_tree_rss_bytes, resource_envelope_for_config
from ..optional_imports import require
from ..readwrite import (
    atomic_write_json,
    safe_read_h5ad,
    safe_write_h5ad,
    safe_write_zarr,
)

logger = get_logger(__name__)

LATENT_SPINE_FILENAME = "spine.h5ad"
LATENT_TASK_CATALOG = "task_catalog.parquet"
LATENT_STORE_SUBDIR = "store"
LATENT_GENERATIONS_SUBDIR = "generations"
LATENT_CURRENT_FILENAME = "current.json"
LATENT_GENERATION_MANIFEST = "generation_manifest.json"
LATENT_RESOURCE_PLAN = "resource_plan.json"
LATENT_TASK_CATALOG_SCHEMA_VERSION = 3
LATENT_GENERATION_SCHEMA_VERSION = 1
_LATENT_SOURCE_ESTIMATE_DTYPE = "float64"
_LATENT_TASK_REQUIRED_COLUMNS = {
    "reference",
    "analysis_mode",
    "core_start",
    "core_end",
    "n_reads",
    "fit_reads",
    "group_path",
    "group_sha256",
    "obsm_keys",
    "varm_keys",
    "obs_columns",
    "n_positions",
    "resource_estimator_version",
    "resource_envelope_id",
    "requested_fit_reads",
    "effective_fit_reads",
    "requested_transform_chunk_reads",
    "effective_transform_chunk_reads",
    "requested_plot_reads",
    "effective_plot_reads",
    "predicted_peak_bytes",
    "measured_peak_bytes",
    "limiting_operation",
    "cp_skip_reason",
    "resource_decisions",
}


def _component(value: object) -> str:
    return quote(str(value), safe="._-")


def _task_path(output_dir: Path, reference: str, start: int, end: int) -> Path:
    return (
        output_dir
        / LATENT_STORE_SUBDIR
        / f"reference={_component(reference)}"
        / f"core={start:012d}-{end:012d}"
    )


def _content_sha256(path: Path) -> str:
    return str(artifact_record(path, path.parent, checksum=True)["sha256"])


def _memory_sample_bytes() -> int:
    """Return best-effort process-tree RSS for persisted peak calibration."""
    try:
        return max(0, int(process_tree_rss_bytes()))
    except Exception:
        logger.debug("Could not sample latent process-tree RSS", exc_info=True)
        return 0


def _resource_record_defaults(
    record: dict[str, object],
    *,
    cfg,
    envelope_id: str,
) -> dict[str, object]:
    """Fill resource fields for reused or injected task records."""
    n_reads = int(record.get("n_reads", 0))
    fit_reads = int(record.get("fit_reads", n_reads))
    n_positions = int(
        record.get(
            "n_positions",
            max(1, int(record.get("core_end", 1)) - int(record.get("core_start", 0))),
        )
    )
    record.setdefault("n_positions", n_positions)
    record.setdefault("analysis_core_id", "")
    record.setdefault("resource_estimator_version", LATENT_RESOURCE_ESTIMATOR_VERSION)
    record.setdefault("resource_envelope_id", envelope_id)
    record.setdefault("requested_fit_reads", fit_reads)
    record.setdefault("effective_fit_reads", fit_reads)
    record.setdefault(
        "requested_transform_chunk_reads",
        int(getattr(cfg, "latent_transform_chunk_reads", 2000)),
    )
    record.setdefault("effective_transform_chunk_reads", 0)
    record.setdefault(
        "requested_plot_reads",
        min(n_reads, max(1, int(getattr(cfg, "latent_plot_max_reads", 10000)))),
    )
    record.setdefault("effective_plot_reads", 0)
    record.setdefault("predicted_peak_bytes", 0)
    record.setdefault("measured_peak_bytes", 0)
    record.setdefault("limiting_operation", "")
    record.setdefault("cp_skip_reason", "")
    record.setdefault(
        "resource_decisions",
        json.dumps(
            {
                "estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
                "decisions": [],
            },
            sort_keys=True,
        ),
    )
    return record


def _append_resource_decision(
    record: dict[str, object],
    decision,
) -> None:
    """Append a decision to one task's portable JSON resource record."""
    try:
        payload = json.loads(str(record.get("resource_decisions", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    decisions = list(payload.get("decisions", []))
    decisions.append(decision.as_dict())
    record["resource_decisions"] = json.dumps(
        {
            "estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
            "decisions": decisions,
        },
        sort_keys=True,
    )
    record["predicted_peak_bytes"] = max(
        int(record.get("predicted_peak_bytes", 0)),
        int(decision.predicted_peak_bytes),
    )
    if decision.limiting_operation:
        record["limiting_operation"] = str(decision.limiting_operation)
    elif not str(record.get("limiting_operation", "")):
        record["limiting_operation"] = str(
            max(
                decisions,
                key=lambda item: int(item.get("predicted_peak_bytes", 0)),
            ).get("operation", "")
        )


def _reset_plot_resource_decisions(record: dict[str, object]) -> None:
    """Drop plot decisions copied from a reused compute generation."""
    try:
        payload = json.loads(str(record.get("resource_decisions", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    decisions = [
        decision for decision in payload.get("decisions", []) if decision.get("operation") != "plot"
    ]
    record["resource_decisions"] = json.dumps(
        {
            "estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
            "decisions": decisions,
        },
        sort_keys=True,
    )
    record["predicted_peak_bytes"] = max(
        (int(decision.get("predicted_peak_bytes", 0)) for decision in decisions),
        default=0,
    )
    limited = next(
        (
            str(decision["limiting_operation"])
            for decision in decisions
            if decision.get("limiting_operation")
        ),
        "",
    )
    record["limiting_operation"] = limited


def _atomic_publish_spine(adata, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.h5ad")
    try:
        safe_write_h5ad(adata, temporary, backup=False, verbose=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_provenance(spine, spine_path: Path, run_root: Path) -> dict[str, object]:
    source_stage = {
        "preprocess_adata_outputs": "preprocess",
        "spatial_adata_outputs": "spatial",
        "hmm_adata_outputs": "hmm",
    }.get(spine_path.parent.name)
    stage_entry = (
        read_experiment_manifest(run_root).get("stages", {}).get(source_stage, {})
        if source_stage is not None
        else {}
    )
    region_catalogs = {}
    configured_catalogs = spine.uns.get("region_catalogs", {})
    if isinstance(configured_catalogs, dict):
        for scope, value in sorted(configured_catalogs.items()):
            path = resolve_relative_path(value, run_root)
            if path is not None and path.is_file():
                region_catalogs[str(scope)] = artifact_record(path, run_root, checksum=True)
    return {
        "source_spine": artifact_record(spine_path, run_root, checksum=True),
        "source_stage": source_stage,
        "source_stage_config_hash": (
            stage_entry.get("config_hash") if isinstance(stage_entry, dict) else None
        ),
        "source_stage_generation_id": (
            stage_entry.get("generation_id") if isinstance(stage_entry, dict) else None
        ),
        "source_stage_completed_at": (
            stage_entry.get("completed_at") if isinstance(stage_entry, dict) else None
        ),
        "region_catalogs": region_catalogs,
    }


def _validate_latent_generation(
    generation_dir: Path,
    *,
    final_dir: Path,
    run_root: Path,
) -> int:
    catalog_path = generation_dir / LATENT_TASK_CATALOG
    catalog = pd.read_parquet(catalog_path)
    missing = sorted(_LATENT_TASK_REQUIRED_COLUMNS.difference(catalog.columns))
    if missing:
        raise RuntimeError(f"latent task catalog is missing required columns: {missing}")
    if catalog.empty:
        raise RuntimeError("latent task catalog contains no successful tasks")

    for record in catalog.to_dict("records"):
        relative_group = Path(str(record["group_path"]))
        if relative_group.is_absolute() or ".." in relative_group.parts:
            raise RuntimeError(f"invalid latent task group path: {relative_group}")
        group_path = generation_dir / relative_group
        if not group_path.is_dir():
            raise RuntimeError(f"latent task group is missing: {group_path}")
        import anndata as ad

        result = ad.experimental.read_lazy(group_path)
        if int(record["n_reads"]) != int(result.n_obs):
            raise RuntimeError(f"latent task row count mismatch for {relative_group}")
        expected = {
            "obsm_keys": sorted(map(str, record["obsm_keys"])),
            "varm_keys": sorted(map(str, record["varm_keys"])),
            "obs_columns": sorted(map(str, record["obs_columns"])),
        }
        observed = {
            "obsm_keys": sorted(map(str, result.obsm.keys())),
            "varm_keys": sorted(map(str, result.varm.keys())),
            "obs_columns": sorted(map(str, result.obs.columns)),
        }
        if expected != observed:
            raise RuntimeError(f"latent task schema mismatch for {relative_group}")
        if str(record["group_sha256"]) != _content_sha256(group_path):
            raise RuntimeError(f"latent task checksum mismatch for {relative_group}")
        try:
            decisions = json.loads(str(record["resource_decisions"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"latent task resource decisions are invalid for {relative_group}"
            ) from exc
        if decisions.get("estimator_version") != LATENT_RESOURCE_ESTIMATOR_VERSION:
            raise RuntimeError(
                f"latent task resource estimator version mismatch for {relative_group}"
            )
        if int(record["effective_fit_reads"]) != int(record["fit_reads"]):
            raise RuntimeError(f"latent task effective fit count mismatch for {relative_group}")
        if int(record["predicted_peak_bytes"]) < 0 or int(record["measured_peak_bytes"]) < 0:
            raise RuntimeError(f"latent task resource peaks are invalid for {relative_group}")

    plot_catalog = pd.read_parquet(generation_dir / "plots" / "catalog.parquet")
    missing_plot_columns = sorted(set(PLOT_CATALOG_COLUMNS).difference(plot_catalog.columns))
    if missing_plot_columns:
        raise RuntimeError(
            f"latent plot catalog is missing required columns: {missing_plot_columns}"
        )
    for record in plot_catalog.to_dict("records"):
        relative_plot = Path(str(record["path"]))
        if relative_plot.is_absolute() or ".." in relative_plot.parts:
            raise RuntimeError(f"invalid latent plot path: {relative_plot}")
        plot_path = generation_dir / relative_plot
        if not plot_path.is_file():
            raise RuntimeError(f"latent plot artifact is missing: {plot_path}")
        source_manifest = record.get("source_manifest")
        if pd.notna(source_manifest):
            relative_manifest = Path(str(source_manifest))
            if relative_manifest.is_absolute() or ".." in relative_manifest.parts:
                raise RuntimeError(f"invalid latent plot source manifest path: {relative_manifest}")
            if not (generation_dir / relative_manifest).is_file():
                raise RuntimeError(f"latent plot source manifest is missing: {source_manifest}")

    spine_path = generation_dir / LATENT_SPINE_FILENAME
    latent_spine, _ = safe_read_h5ad(spine_path)
    expected_pointers = {
        "latent_task_catalog": relative_uns_path(final_dir / LATENT_TASK_CATALOG, run_root),
        "latent_store": relative_uns_path(final_dir / LATENT_STORE_SUBDIR, run_root),
        "latent_resource_plan": relative_uns_path(final_dir / LATENT_RESOURCE_PLAN, run_root),
    }
    for key, expected_value in expected_pointers.items():
        if latent_spine.uns.get(key) != expected_value:
            raise RuntimeError(f"latent spine pointer {key!r} is not publication-safe")

    manifest = sidecar_manifest_path(generation_dir)
    for key in (
        "latent_spine",
        "latent_source_spine",
        "latent_task_catalog",
        "latent_store",
        "latent_plot_catalog",
        "latent_resource_plan",
    ):
        if resolve_sidecar(manifest, key) is None:
            raise RuntimeError(f"latent sidecar manifest cannot resolve {key!r}")
    with (generation_dir / LATENT_GENERATION_MANIFEST).open("r", encoding="utf-8") as handle:
        generation_manifest = json.load(handle)
    if generation_manifest.get("generation_id") != final_dir.name:
        raise RuntimeError("latent generation manifest ID does not match publication path")
    if int(generation_manifest.get("task_count", -1)) != len(catalog):
        raise RuntimeError("latent generation manifest task count does not match catalog")
    with (generation_dir / LATENT_RESOURCE_PLAN).open("r", encoding="utf-8") as handle:
        resource_plan = json.load(handle)
    if resource_plan.get("estimator_version") != LATENT_RESOURCE_ESTIMATOR_VERSION:
        raise RuntimeError("latent resource plan estimator version does not match")
    if int(resource_plan.get("task_count", -1)) != len(catalog):
        raise RuntimeError("latent resource plan task count does not match catalog")
    if generation_manifest.get("resource_envelope_id") != resource_plan.get("resource_envelope_id"):
        raise RuntimeError("latent resource plan envelope ID does not match generation")
    resource_units = resource_plan.get("units")
    if not isinstance(resource_units, list) or len(resource_units) != len(catalog):
        raise RuntimeError("latent resource plan unit records do not match task catalog")
    for record, resource_unit in zip(
        catalog.to_dict("records"),
        resource_units,
        strict=True,
    ):
        for key in (
            "reference",
            "core_start",
            "core_end",
            "effective_fit_reads",
            "effective_transform_chunk_reads",
            "effective_plot_reads",
            "predicted_peak_bytes",
            "measured_peak_bytes",
            "limiting_operation",
            "cp_skip_reason",
        ):
            if resource_unit.get(key) != record[key]:
                raise RuntimeError(
                    f"latent resource plan field {key!r} does not match task catalog"
                )
    return len(catalog)


def _analysis_units(
    spine,
    filter_mask: str | None,
    *,
    spine_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Plan one independent latent space per non-empty reference/core."""
    obs = spine.obs
    if filter_mask is not None:
        obs = obs.loc[obs[filter_mask].astype(bool)]
    units = []
    for core in plan_analysis_cores(spine, spine_path=spine_path):
        reference = core.reference
        reference_obs = obs.loc[obs[REFERENCE_STRAND].astype(str) == str(reference)]
        selected = reference_obs.loc[
            (reference_obs["reference_start"].astype("int64") < core.core_end)
            & (reference_obs["reference_end"].astype("int64") > core.core_start)
        ]
        if not selected.empty:
            units.append(
                {
                    "reference": str(reference),
                    "analysis_mode": core.analysis_mode,
                    "core_start": core.core_start,
                    "core_end": core.core_end,
                    "read_ids": list(map(str, selected.index)),
                    "analysis_core_id": core.analysis_core_id,
                    "analysis_region_ids": core.analysis_region_ids,
                    "original_reference": core.original_reference,
                    "original_start": core.original_start,
                    "original_end": core.original_end,
                    "analysis_planner_version": core.planner_version,
                }
            )
    return units


def _matrix(adata, layer: str, mask: np.ndarray, *, non_negative: bool) -> np.ndarray:
    if layer not in adata.layers:
        raise KeyError(f"latent input layer {layer!r} is unavailable")
    values = np.asarray(adata.layers[layer][:, mask], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.5)
    if non_negative:
        values = np.clip(values, 0.0, None)
    return values


def _fit_indices(n_reads: int, limit: int, seed: int) -> np.ndarray:
    if n_reads <= limit:
        return np.arange(n_reads, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_reads, size=limit, replace=False))


def _nearest_labels(points, reference_points, reference_labels) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=1).fit(reference_points)
    _, indices = neighbors.kneighbors(points)
    return np.asarray(reference_labels)[indices[:, 0]]


def _record_memory_sample(label: str) -> None:
    """Record best-effort process-tree memory for the sequential executor."""
    from ..perf_log import get_perf_logger

    perf = get_perf_logger()
    if perf is None:
        return
    try:
        import psutil

        parent = psutil.Process()
        parent_rss = parent.memory_info().rss
        try:
            children = parent.children(recursive=True)
            child_rss = sum(child.memory_info().rss for child in children if child.is_running())
        except (OSError, psutil.Error):
            children = []
            child_rss = 0
        fields = {}
        try:
            virtual = psutil.virtual_memory()
            fields.update(
                system_used_gb=round(virtual.used / (1024**3), 3),
                system_available_gb=round(virtual.available / (1024**3), 3),
            )
        except (OSError, psutil.Error):
            pass
        perf.sample(
            None,
            tree_rss_gb=(parent_rss + child_rss) / (1024**3),
            parent_rss_gb=round(parent_rss / (1024**3), 3),
            workers_rss_gb=round(child_rss / (1024**3), 3),
            n_live_workers=len(children),
            sample_label=label,
            **fields,
        )
    except Exception:
        logger.debug("Could not record latent memory sample", exc_info=True)


def _fit_matrix_representations(
    adata,
    *,
    layer: str,
    mask: np.ndarray,
    suffix: str,
    cfg,
    fit_indices: np.ndarray,
) -> dict[str, object]:
    """Fit PCA/UMAP/Leiden and NMF, then transform every selected read."""
    from sklearn.decomposition import NMF, PCA

    if not np.asarray(mask, dtype=bool).any():
        logger.warning("Skipping latent representation %s: no eligible positions", suffix)
        return {"layer": layer, "mask": mask, "suffix": suffix}
    matrix = _matrix(adata, layer, mask, non_negative=False)
    fit_matrix = matrix[fit_indices]
    random_state = int(getattr(cfg, "latent_random_state", 0))

    fitted: dict[str, object] = {"layer": layer, "mask": mask, "suffix": suffix}
    if bool(getattr(cfg, "latent_run_pca_umap", True)):
        n_pcs = min(
            int(getattr(cfg, "latent_n_pcs", 10)),
            fit_matrix.shape[0],
            fit_matrix.shape[1],
        )
        if n_pcs >= 1:
            pca = PCA(n_components=n_pcs, svd_solver="auto", random_state=random_state)
            fit_pca = pca.fit_transform(fit_matrix)
            all_pca = pca.transform(matrix).astype(np.float32, copy=False)
            adata.obsm[f"X_pca_{suffix}"] = all_pca
            full_loadings = np.zeros((adata.n_vars, n_pcs), dtype=np.float32)
            full_loadings[mask] = pca.components_.T.astype(np.float32, copy=False)
            adata.varm[f"PCs_{suffix}"] = full_loadings
            adata.uns[f"pca_{suffix}"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "fit_read_count": int(len(fit_indices)),
                "layer": layer,
            }
            fitted.update(pca=pca, fit_pca=fit_pca)

            umap = require("umap", extra="umap", purpose="partitioned latent UMAP")
            n_neighbors = min(
                int(getattr(cfg, "latent_knn_neighbors", 15)),
                max(2, len(fit_indices) - 1),
            )
            if len(fit_indices) >= 3:
                model = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                    metric="euclidean",
                    random_state=random_state,
                    n_jobs=max(1, int(getattr(cfg, "threads", 1) or 1)),
                )
                fit_umap = model.fit_transform(fit_pca)
                all_umap = model.transform(all_pca).astype(np.float32, copy=False)
                all_umap[fit_indices] = fit_umap
                adata.obsm[f"X_umap_{suffix}"] = all_umap

                fit_labels = np.zeros(len(fit_indices), dtype=str)
                graph = getattr(model, "graph_", None)
                if graph is not None:
                    try:
                        import anndata as ad

                        from .calculate_leiden import calculate_leiden

                        fit = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names[fit_indices]))
                        fit.obsp["connectivities"] = graph.tocsr()
                        calculate_leiden(
                            fit,
                            resolution=float(getattr(cfg, "latent_leiden_resolution", 0.1)),
                            key_added="leiden",
                            connectivities_key="connectivities",
                        )
                        fit_labels = fit.obs["leiden"].astype(str).to_numpy()
                    except Exception as exc:
                        logger.warning("Leiden failed for %s: %s", suffix, exc)
                labels = _nearest_labels(all_pca, fit_pca, fit_labels)
                labels[fit_indices] = fit_labels
                adata.obs[f"leiden_{suffix}"] = pd.Categorical(labels)
                adata.uns[f"umap_{suffix}"] = {
                    "n_neighbors": n_neighbors,
                    "fit_read_count": int(len(fit_indices)),
                }
                fitted.update(
                    umap=model,
                    fit_umap=fit_umap,
                    fit_labels=fit_labels,
                )

    if bool(getattr(cfg, "latent_run_nmf", True)):
        non_negative = np.clip(matrix, 0.0, None)
        fit_non_negative = non_negative[fit_indices]
        n_components = min(
            int(getattr(cfg, "latent_nmf_components", 2)),
            fit_non_negative.shape[0],
            fit_non_negative.shape[1],
        )
        if n_components >= 1:
            nmf = NMF(
                n_components=n_components,
                init="nndsvda",
                max_iter=int(getattr(cfg, "latent_nmf_max_iter", 500)),
                random_state=random_state,
            )
            nmf.fit(fit_non_negative)
            adata.obsm[f"X_nmf_{suffix}"] = nmf.transform(non_negative).astype(
                np.float32, copy=False
            )
            full_components = np.zeros((adata.n_vars, n_components), dtype=np.float32)
            full_components[mask] = nmf.components_.T.astype(np.float32, copy=False)
            adata.varm[f"H_nmf_{suffix}"] = full_components
            adata.uns[f"nmf_{suffix}"] = {
                "fit_read_count": int(len(fit_indices)),
                "layer": layer,
                "n_components": n_components,
            }
            fitted["nmf"] = nmf
    return fitted


def _transform_matrix_representations(adata, fitted: dict[str, object]) -> dict[str, np.ndarray]:
    """Project a materialized read chunk through one fitted latent space."""
    if not any(key in fitted for key in ("pca", "nmf")):
        return {}
    layer = str(fitted["layer"])
    mask = np.asarray(fitted["mask"], dtype=bool)
    suffix = str(fitted["suffix"])
    matrix = _matrix(adata, layer, mask, non_negative=False)
    outputs: dict[str, np.ndarray] = {}
    pca = fitted.get("pca")
    if pca is not None:
        pca_values = pca.transform(matrix).astype(np.float32, copy=False)
        outputs[f"X_pca_{suffix}"] = pca_values
        umap_model = fitted.get("umap")
        if umap_model is not None:
            outputs[f"X_umap_{suffix}"] = umap_model.transform(pca_values).astype(
                np.float32, copy=False
            )
            outputs[f"leiden_{suffix}"] = _nearest_labels(
                pca_values,
                np.asarray(fitted["fit_pca"]),
                np.asarray(fitted["fit_labels"]),
            )
    nmf = fitted.get("nmf")
    if nmf is not None:
        outputs[f"X_nmf_{suffix}"] = nmf.transform(np.clip(matrix, 0.0, None)).astype(
            np.float32, copy=False
        )
    return outputs


def _fit_cp_representations(adata, *, mod_mask, non_mod_mask, valid_mask, cfg) -> None:
    """Preserve the six legacy CP variants when the complete unit is bounded."""
    if not bool(getattr(cfg, "latent_run_cp", True)):
        return
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning("Skipping latent CP: %s is unavailable", SEQUENCE_INTEGER_ENCODING)
        return
    from .tensor_factorization import calculate_sequence_cp_decomposition

    specs = (
        ("shared_valid_mod_sites_ohe_sequence_N_masked", mod_mask, False),
        ("shared_valid_mod_sites_ohe_sequence_N_masked_non_negative", mod_mask, True),
        ("non_mod_site_ohe_sequence_N_masked", non_mod_mask, False),
        ("non_mod_site_ohe_sequence_N_masked_non_negative", non_mod_mask, True),
        ("full_ohe_sequence_N_masked", valid_mask, False),
        ("full_ohe_sequence_N_masked_non_negative", valid_mask, True),
    )
    for suffix, mask, non_negative in specs:
        if not np.asarray(mask, dtype=bool).any():
            logger.warning("Skipping CP %s: no eligible positions", suffix)
            continue
        calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=mask,
            var_mask_name=suffix,
            rank=int(getattr(cfg, "latent_cp_rank", 2)),
            n_iter_max=int(getattr(cfg, "latent_cp_iterations", 100)),
            random_state=int(getattr(cfg, "latent_random_state", 0)),
            embedding_key=f"X_cp_{suffix}",
            components_key=f"H_cp_{suffix}",
            uns_key=f"cp_{suffix}",
            non_negative=non_negative,
        )


def _plot_colors(result, basis: str, cfg) -> list[str]:
    """Return informative colors, including the cluster labels for this basis."""
    candidates = [
        getattr(cfg, "sample_name_col_for_plotting", "Sample"),
        *list(getattr(cfg, "umap_layers_to_plot", []) or []),
    ]
    for prefix in ("pca_", "umap_", "nmf_"):
        if basis.startswith(prefix):
            candidates.append(f"leiden_{basis.removeprefix(prefix)}")
            break

    colors = []
    for color in dict.fromkeys(map(str, candidates)):
        if color in result.obs and result.obs[color].nunique(dropna=False) > 1:
            colors.append(color)
    return colors


def _plot_task(result, record, cfg, layout) -> None:
    from ..plotting import (
        plot_cp_sequence_components,
        plot_embedding_grid,
        plot_nmf_components,
        plot_pca_components,
    )

    task_label = (
        f"reference={_component(record['reference'])}__"
        f"core={int(record['core_start']):012d}-{int(record['core_end']):012d}"
    )
    for key in list(result.obsm.keys()):
        if np.asarray(result.obsm[key]).shape[1] < 2:
            continue
        basis = key.removeprefix("X_")
        colors = _plot_colors(result, basis, cfg)
        path = plot_embedding_grid(
            result,
            basis=basis,
            color=colors,
            output_dir=layout.categories["embeddings"] / task_label,
            prefix=basis,
        )
        if path is not None:
            register_plot_artifact(
                layout,
                path,
                stage="latent",
                category="embeddings",
                plot_type=basis,
                reference=str(record["reference"]),
                core_start=int(record["core_start"]),
                core_end=int(record["core_end"]),
            )
    for key in list(result.varm.keys()):
        output = layout.categories["loadings"] / task_label / _component(key)
        if key.startswith("PCs_"):
            paths = plot_pca_components(result, output_dir=output, components_key=key)
        elif key.startswith("H_nmf_"):
            paths = plot_nmf_components(result, output_dir=output, components_key=key)
        elif key.startswith("H_cp_"):
            paths = plot_cp_sequence_components(
                result,
                output_dir=output,
                components_key=key,
                uns_key=key.replace("H_cp_", "cp_", 1),
            )
        else:
            continue
        for path in paths.values():
            register_plot_artifact(
                layout,
                path,
                stage="latent",
                category="loadings",
                plot_type=key,
                reference=str(record["reference"]),
                core_start=int(record["core_start"]),
                core_end=int(record["core_end"]),
            )


def _read_plot_subset(group_path: Path, *, max_reads: int, seed: int):
    """Lazily materialize only the deterministic rows admitted for plotting."""
    import anndata as ad

    lazy = ad.experimental.read_lazy(group_path)
    if lazy.n_obs <= max_reads:
        return lazy.to_memory()
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(lazy.n_obs, size=max_reads, replace=False))
    return lazy[chosen, :].to_memory()


def execute_latent_unit(spine_path, unit, cfg, output_dir) -> dict[str, object] | None:
    """Fit and persist one reference/core-local latent space."""
    reference = str(unit["reference"])
    start, end = int(unit["core_start"]), int(unit["core_end"])
    read_ids = list(map(str, unit["read_ids"]))
    min_reads = max(2, int(getattr(cfg, "latent_min_reads", 3)))
    if len(read_ids) < min_reads:
        logger.warning(
            "Skipping latent unit %s:%d-%d: %d reads is below latent_min_reads=%d",
            reference,
            start,
            end,
            len(read_ids),
            min_reads,
        )
        return None

    envelope = resource_envelope_for_config(cfg)
    envelope_identity = resource_envelope_id(envelope)
    measured_peak = _memory_sample_bytes()
    estimated_positions = max(1, end - start)
    requested_fit_reads = min(
        len(read_ids),
        max(min_reads, int(getattr(cfg, "latent_max_fit_reads", 5000))),
    )
    fit_decision = resolve_latent_operation(
        cfg,
        "fit",
        requested_reads=requested_fit_reads,
        n_positions=estimated_positions,
        minimum_reads=min_reads,
        source_dtype=_LATENT_SOURCE_ESTIMATE_DTYPE,
    )
    decisions = [fit_decision]

    layers = list(
        dict.fromkeys(
            [str(getattr(cfg, "layer_for_umap_plotting", "nan_half")), SEQUENCE_INTEGER_ENCODING]
        )
    )
    fit_positions = _fit_indices(
        len(read_ids),
        fit_decision.effective_reads,
        int(getattr(cfg, "latent_random_state", 0)),
    )
    fit_read_ids = [read_ids[index] for index in fit_positions]
    fit_adata = materialize(
        spine_path,
        references=reference,
        read_ids=fit_read_ids,
        start=start,
        end=end,
        layers=layers,
    )
    measured_peak = max(measured_peak, _memory_sample_bytes())
    references = [reference]
    modality = str(getattr(cfg, "smf_modality", "conversion"))
    deaminase = modality != "conversion"
    mod_mask = _build_mod_sites_var_filter_mask(fit_adata, references, cfg, modality, deaminase)
    non_mod_mask = _build_shared_valid_non_mod_sites_mask(
        fit_adata, references, cfg, modality, deaminase
    )
    valid_mask = _build_reference_position_mask(fit_adata, references)
    fit_indices = np.arange(fit_adata.n_obs, dtype=np.int64)
    signal_layer = str(getattr(cfg, "layer_for_umap_plotting", "nan_half"))
    fitted = [
        _fit_matrix_representations(
            fit_adata,
            layer=signal_layer,
            mask=mod_mask,
            suffix="shared_valid_mod_sites_binary_mod_arrays",
            cfg=cfg,
            fit_indices=fit_indices,
        ),
        _fit_matrix_representations(
            fit_adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            mask=valid_mask,
            suffix="shared_valid_ref_sites_integer_sequence_encodings",
            cfg=cfg,
            fit_indices=fit_indices,
        ),
    ]
    measured_peak = max(measured_peak, _memory_sample_bytes())
    cp_skip_reason = ""
    if bool(getattr(cfg, "latent_run_cp", True)):
        if len(read_ids) > int(getattr(cfg, "latent_max_fit_reads", 5000)):
            cp_skip_reason = "unit_exceeds_fit_ceiling"
        elif len(fit_read_ids) != len(read_ids):
            cp_skip_reason = "fit_reduced_by_memory"
        else:
            cp_decision = resolve_latent_operation(
                cfg,
                "cp",
                requested_reads=len(read_ids),
                n_positions=fit_adata.n_vars,
                minimum_reads=0,
            )
            decisions.append(cp_decision)
            if cp_decision.effective_reads < len(read_ids):
                cp_skip_reason = (
                    "minimum_unit_exceeds_memory"
                    if cp_decision.effective_reads < min_reads
                    else "complete_unit_exceeds_memory"
                )
            else:
                _fit_cp_representations(
                    fit_adata,
                    mod_mask=mod_mask,
                    non_mod_mask=non_mod_mask,
                    valid_mask=valid_mask,
                    cfg=cfg,
                )
                measured_peak = max(measured_peak, _memory_sample_bytes())
        if cp_skip_reason:
            message = (
                f"Skipping CP for {reference}:{start}-{end}: {cp_skip_reason} "
                f"(policy={getattr(cfg, 'latent_cp_memory_policy', 'skip')})"
            )
            if str(getattr(cfg, "latent_cp_memory_policy", "skip")).strip().lower() == "fail":
                raise LatentResourceError(message)
            logger.warning(message)

    if not fit_adata.obsm:
        logger.warning(
            "Skipping latent unit %s:%d-%d: no latent representations could be computed",
            reference,
            start,
            end,
        )
        return None

    if len(read_ids) == len(fit_read_ids):
        adata = fit_adata
        effective_transform_chunk_reads = 0
    else:
        import anndata as ad

        result_decision = resolve_latent_operation(
            cfg,
            "result",
            requested_reads=len(read_ids),
            n_positions=fit_adata.n_vars,
            minimum_reads=len(read_ids),
        )
        decisions.append(result_decision)
        spine = load_spine(spine_path, verbose=False)
        adata = ad.AnnData(
            obs=spine.obs.loc[read_ids].copy(),
            var=fit_adata.var.copy(),
        )
        for key, value in fit_adata.varm.items():
            adata.varm[key] = np.asarray(value)
        adata.uns.update(dict(fit_adata.uns))
        embedding_shapes = {
            key: np.asarray(value).shape[1] for key, value in fit_adata.obsm.items()
        }
        for key, width in embedding_shapes.items():
            adata.obsm[key] = np.full((len(read_ids), width), np.nan, dtype=np.float32)
        label_keys = [key for key in fit_adata.obs if str(key).startswith("leiden_")]
        label_values = {key: np.full(len(read_ids), None, dtype=object) for key in label_keys}
        row_lookup = {read_id: index for index, read_id in enumerate(read_ids)}
        requested_transform_chunk_reads = min(
            len(read_ids),
            max(1, int(getattr(cfg, "latent_transform_chunk_reads", 2000))),
        )
        transform_decision = resolve_latent_operation(
            cfg,
            "transform",
            requested_reads=requested_transform_chunk_reads,
            n_positions=fit_adata.n_vars,
            minimum_reads=1,
            source_dtype=_LATENT_SOURCE_ESTIMATE_DTYPE,
        )
        decisions.append(transform_decision)
        chunk_size = transform_decision.effective_reads
        effective_transform_chunk_reads = chunk_size
        for chunk_start in range(0, len(read_ids), chunk_size):
            chunk_ids = read_ids[chunk_start : chunk_start + chunk_size]
            chunk = materialize(
                spine_path,
                references=reference,
                read_ids=chunk_ids,
                start=start,
                end=end,
                layers=layers,
            )
            target_rows = np.asarray([row_lookup[str(name)] for name in chunk.obs_names])
            for model in fitted:
                transformed = _transform_matrix_representations(chunk, model)
                for key, values in transformed.items():
                    if key.startswith("leiden_"):
                        label_values[key][target_rows] = values
                    else:
                        adata.obsm[key][target_rows] = values
            measured_peak = max(measured_peak, _memory_sample_bytes())
        for key, values in label_values.items():
            adata.obs[key] = pd.Categorical(values)

        # Preserve the exact fitted coordinates/labels rather than UMAP-transforming
        # its own training observations.
        fit_target_rows = np.asarray([row_lookup[str(name)] for name in fit_adata.obs_names])
        for key, values in fit_adata.obsm.items():
            adata.obsm[key][fit_target_rows] = np.asarray(values)
        for key in label_keys:
            adata.obs.loc[fit_adata.obs_names, key] = fit_adata.obs[key].astype(str).to_numpy()

    adata.uns["latent_coordinate_scope"] = {
        "reference": reference,
        "core_start": start,
        "core_end": end,
        "independent_coordinate_system": True,
        "analysis_core_id": str(unit.get("analysis_core_id", "")),
        "analysis_region_ids": list(unit.get("analysis_region_ids", ())),
        "analysis_planner_version": int(unit.get("analysis_planner_version", 1)),
    }
    write_decision = resolve_latent_operation(
        cfg,
        "write",
        requested_reads=adata.n_obs,
        n_positions=adata.n_vars,
        minimum_reads=adata.n_obs,
        source_dtype=_LATENT_SOURCE_ESTIMATE_DTYPE,
    )
    decisions.append(write_decision)
    output_path = _task_path(Path(output_dir), reference, start, end)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="zarr v3 autosharding will be the default.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part.*",
            category=UserWarning,
        )
        safe_write_zarr(adata, output_path, backup=False, verbose=False, zarr_format=3)
    measured_peak = max(measured_peak, _memory_sample_bytes())
    limiting_operation = next(
        (
            str(decision.limiting_operation)
            for decision in decisions
            if decision.limiting_operation is not None
        ),
        "",
    )
    if not limiting_operation:
        limiting_operation = str(
            max(decisions, key=lambda decision: decision.predicted_peak_bytes).operation
        )
    return {
        "reference": reference,
        "analysis_mode": str(unit["analysis_mode"]),
        "core_start": start,
        "core_end": end,
        "n_reads": adata.n_obs,
        "fit_reads": int(len(fit_read_ids)),
        "n_positions": int(adata.n_vars),
        "group_path": output_path.relative_to(output_dir).as_posix(),
        "group_sha256": _content_sha256(output_path),
        "obsm_keys": list(adata.obsm.keys()),
        "analysis_core_id": str(unit.get("analysis_core_id", "")),
        "analysis_region_ids": tuple(unit.get("analysis_region_ids", ())),
        "original_reference": unit.get("original_reference"),
        "original_start": unit.get("original_start"),
        "original_end": unit.get("original_end"),
        "analysis_planner_version": int(unit.get("analysis_planner_version", 1)),
        "varm_keys": list(adata.varm.keys()),
        "obs_columns": list(adata.obs.columns),
        "resource_estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
        "resource_envelope_id": envelope_identity,
        "requested_fit_reads": requested_fit_reads,
        "effective_fit_reads": int(len(fit_read_ids)),
        "requested_transform_chunk_reads": int(getattr(cfg, "latent_transform_chunk_reads", 2000)),
        "effective_transform_chunk_reads": effective_transform_chunk_reads,
        "requested_plot_reads": min(
            adata.n_obs,
            max(1, int(getattr(cfg, "latent_plot_max_reads", 10000))),
        ),
        "effective_plot_reads": 0,
        "predicted_peak_bytes": max(int(decision.predicted_peak_bytes) for decision in decisions),
        "measured_peak_bytes": measured_peak,
        "limiting_operation": limiting_operation,
        "cp_skip_reason": cp_skip_reason,
        "resource_decisions": json.dumps(
            {
                "estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
                "decisions": [decision.as_dict() for decision in decisions],
            },
            sort_keys=True,
        ),
    }


def execute_partitioned_latent(
    spine_path,
    cfg,
    output_dir,
    *,
    reuse_generation: str | Path | None = None,
) -> dict[str, Path | str | int]:
    """Build, validate, and atomically publish one immutable latent generation."""
    from ..cli.helpers import stage_config_hash, stage_plot_config_hash

    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    if output_dir.name != LATENT_DIR:
        logger.debug("Using non-canonical latent output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = output_dir.parent
    generation_id = uuid4().hex
    staging_dir = output_dir / ".staging" / generation_id
    final_dir = output_dir / LATENT_GENERATIONS_SUBDIR / generation_id
    staging_dir.mkdir(parents=True)
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        spine = load_spine(spine_path)
        envelope = resource_envelope_for_config(cfg)
        envelope_identity = resource_envelope_id(envelope)
        filter_mask = next(
            (column for column in ("passes_dedup", "passes_qc") if column in spine.obs),
            None,
        )
        records: list[dict[str, object]]
        if reuse_generation is not None:
            reuse_generation = Path(reuse_generation)
            reuse_catalog = pd.read_parquet(reuse_generation / LATENT_TASK_CATALOG)
            records = reuse_catalog.to_dict("records")
            for record in records:
                _reset_plot_resource_decisions(record)
            shutil.copytree(
                reuse_generation / LATENT_STORE_SUBDIR,
                staging_dir / LATENT_STORE_SUBDIR,
                copy_function=shutil.copy2,
            )
            pd.DataFrame(records).to_parquet(staging_dir / LATENT_TASK_CATALOG, index=False)
        else:
            units = _analysis_units(spine, filter_mask, spine_path=spine_path)
            if not units:
                raise RuntimeError("partitioned latent analysis has no non-empty units")
            records = []
            _record_memory_sample("executor_start")
            # Independently fitted units execute sequentially to avoid multiplying
            # model and materialization memory.
            for unit in units:
                record = execute_latent_unit(spine_path, unit, cfg, staging_dir)
                if record is not None:
                    group_path = staging_dir / str(record["group_path"])
                    record.setdefault("group_sha256", _content_sha256(group_path))
                    record.setdefault("analysis_planner_version", 1)
                    _resource_record_defaults(
                        record,
                        cfg=cfg,
                        envelope_id=envelope_identity,
                    )
                    records.append(record)
                _record_memory_sample(
                    f"unit_complete:{unit['reference']}:{unit['core_start']}-{unit['core_end']}"
                )
            if not records:
                raise RuntimeError(
                    "partitioned latent analysis has no units meeting latent_min_reads"
                )
            pd.DataFrame(records).to_parquet(staging_dir / LATENT_TASK_CATALOG, index=False)

        for record in records:
            _resource_record_defaults(
                record,
                cfg=cfg,
                envelope_id=envelope_identity,
            )
        layout = prepare_analysis_plot_layout(staging_dir, stage="latent", source_spine=spine_path)
        pd.DataFrame(columns=PLOT_CATALOG_COLUMNS).to_parquet(layout.catalog, index=False)
        for record in records:
            requested_plot_reads = min(
                int(record["n_reads"]),
                max(1, int(getattr(cfg, "latent_plot_max_reads", 10000))),
            )
            plot_decision = resolve_latent_operation(
                cfg,
                "plot",
                requested_reads=requested_plot_reads,
                n_positions=int(record["n_positions"]),
                minimum_reads=1,
            )
            _append_resource_decision(record, plot_decision)
            record["requested_plot_reads"] = requested_plot_reads
            record["effective_plot_reads"] = plot_decision.effective_reads
            result = _read_plot_subset(
                staging_dir / str(record["group_path"]),
                max_reads=plot_decision.effective_reads,
                seed=int(getattr(cfg, "plot_subsample_seed", 0)),
            )
            _plot_task(result, record, cfg, layout)
            record["measured_peak_bytes"] = max(
                int(record["measured_peak_bytes"]),
                _memory_sample_bytes(),
            )
            _record_memory_sample(
                f"plot_complete:{record['reference']}:{record['core_start']}-{record['core_end']}"
            )
        pd.DataFrame(records).to_parquet(staging_dir / LATENT_TASK_CATALOG, index=False)

        resource_plan = staging_dir / LATENT_RESOURCE_PLAN
        atomic_write_json(
            resource_plan,
            {
                "schema_version": 1,
                "estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
                "resource_envelope_id": envelope_identity,
                "resource_envelope": envelope.as_dict(),
                "task_count": len(records),
                "units": [
                    {
                        **{
                            key: record[key]
                            for key in (
                                "reference",
                                "core_start",
                                "core_end",
                                "analysis_core_id",
                                "n_positions",
                                "resource_estimator_version",
                                "resource_envelope_id",
                                "requested_fit_reads",
                                "effective_fit_reads",
                                "requested_transform_chunk_reads",
                                "effective_transform_chunk_reads",
                                "requested_plot_reads",
                                "effective_plot_reads",
                                "predicted_peak_bytes",
                                "measured_peak_bytes",
                                "limiting_operation",
                                "cp_skip_reason",
                            )
                        },
                        "decisions": json.loads(str(record["resource_decisions"]))["decisions"],
                    }
                    for record in records
                ],
            },
        )

        generation_spine = staging_dir / LATENT_SPINE_FILENAME
        latent_spine = spine.copy()
        latent_spine.uns["latent_source_spine"] = relative_uns_path(spine_path, run_root)
        latent_spine.uns["latent_task_catalog"] = relative_uns_path(
            final_dir / LATENT_TASK_CATALOG, run_root
        )
        latent_spine.uns["latent_store"] = relative_uns_path(
            final_dir / LATENT_STORE_SUBDIR, run_root
        )
        latent_spine.uns["latent_resource_plan"] = relative_uns_path(
            final_dir / LATENT_RESOURCE_PLAN, run_root
        )
        latent_spine.uns["latent_filter_mask"] = filter_mask or ""
        latent_spine.uns["latent_schema_version"] = LATENT_TASK_CATALOG_SCHEMA_VERSION
        latent_spine.uns["latent_generation_id"] = generation_id
        latent_spine.uns["latent_coordinate_scope"] = "reference_core"
        safe_write_h5ad(latent_spine, generation_spine, backup=False, verbose=False)

        manifest = sidecar_manifest_path(staging_dir)
        register_sidecar(manifest, "latent_spine", generation_spine)
        register_sidecar(manifest, "latent_source_spine", spine_path)
        register_sidecar(manifest, "latent_task_catalog", staging_dir / LATENT_TASK_CATALOG)
        register_sidecar(manifest, "latent_store", staging_dir / LATENT_STORE_SUBDIR)
        register_sidecar(manifest, "latent_plot_catalog", layout.catalog)
        register_sidecar(manifest, "latent_resource_plan", resource_plan)

        generation_manifest = staging_dir / LATENT_GENERATION_MANIFEST
        atomic_write_json(
            generation_manifest,
            {
                "schema_version": LATENT_GENERATION_SCHEMA_VERSION,
                "generation_id": generation_id,
                "compute_config_hash": stage_config_hash(cfg, "latent"),
                "plot_config_hash": stage_plot_config_hash(cfg, "latent"),
                "source": _source_provenance(spine, spine_path, run_root),
                "analysis_planner_versions": sorted(
                    {int(record.get("analysis_planner_version", 1)) for record in records}
                ),
                "task_catalog_schema_version": LATENT_TASK_CATALOG_SCHEMA_VERSION,
                "task_count": len(records),
                "resource_estimator_version": LATENT_RESOURCE_ESTIMATOR_VERSION,
                "resource_envelope_id": envelope_identity,
                "resource_envelope": envelope.as_dict(),
                "resource_plan": LATENT_RESOURCE_PLAN,
                "resource_summary": {
                    "requested_fit_reads_ceiling": int(getattr(cfg, "latent_max_fit_reads", 5000)),
                    "requested_transform_chunk_reads_ceiling": int(
                        getattr(cfg, "latent_transform_chunk_reads", 2000)
                    ),
                    "requested_plot_reads_ceiling": int(
                        getattr(cfg, "latent_plot_max_reads", 10000)
                    ),
                    "predicted_peak_bytes": max(
                        int(record["predicted_peak_bytes"]) for record in records
                    ),
                    "measured_peak_bytes": max(
                        int(record["measured_peak_bytes"]) for record in records
                    ),
                    "limiting_operations": sorted(
                        {
                            str(record["limiting_operation"])
                            for record in records
                            if str(record["limiting_operation"])
                        }
                    ),
                    "cp_skip_reasons": sorted(
                        {
                            str(record["cp_skip_reason"])
                            for record in records
                            if str(record["cp_skip_reason"])
                        }
                    ),
                },
                "reused_compute_generation": (
                    Path(reuse_generation).name if reuse_generation is not None else None
                ),
            },
        )
        task_count = _validate_latent_generation(
            staging_dir, final_dir=final_dir, run_root=run_root
        )

        os.replace(staging_dir, final_dir)
        canonical_spine = output_dir / LATENT_SPINE_FILENAME
        _atomic_publish_spine(latent_spine, canonical_spine)
        current = output_dir / LATENT_CURRENT_FILENAME
        atomic_write_json(
            current,
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "generation_path": final_dir.relative_to(output_dir).as_posix(),
            },
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logger.info(
        "Published partitioned latent generation %s with %d unit(s)",
        generation_id,
        task_count,
    )
    return {
        "spine": canonical_spine,
        "generation_spine": final_dir / LATENT_SPINE_FILENAME,
        "task_catalog": final_dir / LATENT_TASK_CATALOG,
        "store": final_dir / LATENT_STORE_SUBDIR,
        "plots": final_dir / "plots",
        "plot_catalog": final_dir / "plots" / "catalog.parquet",
        "manifest": sidecar_manifest_path(final_dir),
        "generation_manifest": final_dir / LATENT_GENERATION_MANIFEST,
        "resource_plan": final_dir / LATENT_RESOURCE_PLAN,
        "generation": final_dir,
        "current": current,
        "generation_id": generation_id,
        "task_count": task_count,
    }
