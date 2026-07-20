"""Bounded latent-representation fitting over partitioned experiment spines.

Each output group owns an independent coordinate system for one reference locus or
genome core. Embeddings from different groups must not be compared as though their
component axes were shared.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from smftools.constants import LATENT_DIR, REFERENCE_STRAND, SEQUENCE_INTEGER_ENCODING
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
from ..informatics.experiment_spine import write_experiment_spine
from ..informatics.partition_read import load_spine, materialize, relative_uns_path
from ..informatics.sidecar_manifest import register_sidecar, sidecar_manifest_path
from ..optional_imports import require
from ..readwrite import safe_write_h5ad, safe_write_zarr

logger = get_logger(__name__)

LATENT_SPINE_FILENAME = "spine.h5ad"
LATENT_TASK_CATALOG = "task_catalog.parquet"
LATENT_STORE_SUBDIR = "store"


def _component(value: object) -> str:
    return quote(str(value), safe="._-")


def _task_path(output_dir: Path, reference: str, start: int, end: int) -> Path:
    return (
        output_dir
        / LATENT_STORE_SUBDIR
        / f"reference={_component(reference)}"
        / f"core={start:012d}-{end:012d}"
    )


def _analysis_units(spine, filter_mask: str | None) -> list[dict[str, object]]:
    """Plan one independent latent space per non-empty reference/core."""
    obs = spine.obs
    if filter_mask is not None:
        obs = obs.loc[obs[filter_mask].astype(bool)]
    units = []
    for reference, raw_plan in sorted(dict(spine.uns.get("reference_plans", {})).items()):
        plan = dict(raw_plan)
        length = int(plan["reference_length"])
        tile_size = length if plan.get("analysis_mode") == "locus" else int(plan["tile_size"])
        reference_obs = obs.loc[obs[REFERENCE_STRAND].astype(str) == str(reference)]
        for start in range(0, length, tile_size):
            end = min(start + tile_size, length)
            selected = reference_obs.loc[
                (reference_obs["reference_start"].astype("int64") < end)
                & (reference_obs["reference_end"].astype("int64") > start)
            ]
            if not selected.empty:
                units.append(
                    {
                        "reference": str(reference),
                        "analysis_mode": str(plan["analysis_mode"]),
                        "core_start": start,
                        "core_end": end,
                        "read_ids": list(map(str, selected.index)),
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

    layers = list(
        dict.fromkeys(
            [str(getattr(cfg, "layer_for_umap_plotting", "nan_half")), SEQUENCE_INTEGER_ENCODING]
        )
    )
    fit_positions = _fit_indices(
        len(read_ids),
        max(min_reads, int(getattr(cfg, "latent_max_fit_reads", 5000))),
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
    if len(read_ids) <= int(getattr(cfg, "latent_max_fit_reads", 5000)):
        _fit_cp_representations(
            fit_adata,
            mod_mask=mod_mask,
            non_mod_mask=non_mod_mask,
            valid_mask=valid_mask,
            cfg=cfg,
        )
    elif bool(getattr(cfg, "latent_run_cp", True)):
        logger.warning(
            "Skipping CP for %s:%d-%d: %d reads exceeds latent_max_fit_reads",
            reference,
            start,
            end,
            len(read_ids),
        )

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
    else:
        import anndata as ad

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
        chunk_size = max(1, int(getattr(cfg, "latent_transform_chunk_reads", 2000)))
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
    }
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
    return {
        "reference": reference,
        "analysis_mode": str(unit["analysis_mode"]),
        "core_start": start,
        "core_end": end,
        "n_reads": adata.n_obs,
        "fit_reads": int(len(fit_read_ids)),
        "group_path": output_path.relative_to(output_dir).as_posix(),
        "obsm_keys": list(adata.obsm.keys()),
        "varm_keys": list(adata.varm.keys()),
        "obs_columns": list(adata.obs.columns),
    }


def execute_partitioned_latent(spine_path, cfg, output_dir) -> dict[str, Path]:
    """Run bounded latent units and publish a linked thin spine."""
    spine_path = Path(spine_path)
    output_dir = Path(output_dir)
    if output_dir.name != LATENT_DIR:
        logger.debug("Using non-canonical latent output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spine = load_spine(spine_path)
    filter_mask = next(
        (column for column in ("passes_dedup", "passes_qc") if column in spine.obs),
        None,
    )
    units = _analysis_units(spine, filter_mask)
    if not units:
        raise RuntimeError("partitioned latent analysis has no non-empty units")

    records = []
    _record_memory_sample("executor_start")
    # Latent fits allocate model state in addition to their materialized matrix. Run
    # units sequentially so independently fitted spaces cannot multiply peak memory.
    for unit in units:
        record = execute_latent_unit(spine_path, unit, cfg, output_dir)
        if record is not None:
            records.append(record)
        _record_memory_sample(
            f"unit_complete:{unit['reference']}:{unit['core_start']}-{unit['core_end']}"
        )
    if not records:
        raise RuntimeError("partitioned latent analysis has no units meeting latent_min_reads")

    catalog_path = output_dir / LATENT_TASK_CATALOG
    pd.DataFrame(records).to_parquet(catalog_path, index=False)
    layout = prepare_analysis_plot_layout(output_dir, stage="latent", source_spine=spine_path)
    pd.DataFrame(columns=PLOT_CATALOG_COLUMNS).to_parquet(layout.catalog, index=False)
    from ..readwrite import safe_read_zarr

    for record in records:
        result, _ = safe_read_zarr(output_dir / str(record["group_path"]), verbose=False)
        _plot_task(result, record, cfg, layout)
        _record_memory_sample(
            f"plot_complete:{record['reference']}:{record['core_start']}-{record['core_end']}"
        )

    run_root = output_dir.parent
    output_spine = output_dir / LATENT_SPINE_FILENAME
    latent_spine = spine.copy()
    latent_spine.uns["latent_source_spine"] = relative_uns_path(spine_path, run_root)
    latent_spine.uns["latent_task_catalog"] = relative_uns_path(catalog_path, run_root)
    latent_spine.uns["latent_store"] = relative_uns_path(output_dir / LATENT_STORE_SUBDIR, run_root)
    latent_spine.uns["latent_filter_mask"] = filter_mask or ""
    latent_spine.uns["latent_schema_version"] = 1
    latent_spine.uns["latent_coordinate_scope"] = "reference_core"
    safe_write_h5ad(latent_spine, output_spine, backup=False, verbose=False)
    write_experiment_spine(run_root)

    manifest = sidecar_manifest_path(output_dir)
    register_sidecar(manifest, "latent_spine", output_spine)
    register_sidecar(manifest, "latent_source_spine", spine_path)
    register_sidecar(manifest, "latent_task_catalog", catalog_path)
    register_sidecar(manifest, "latent_store", output_dir / LATENT_STORE_SUBDIR)
    register_sidecar(manifest, "latent_plot_catalog", layout.catalog)
    logger.info("Wrote partitioned latent stage with %d unit(s)", len(records))
    return {
        "spine": output_spine,
        "task_catalog": catalog_path,
        "store": output_dir / LATENT_STORE_SUBDIR,
        "plots": layout.root,
        "plot_catalog": layout.catalog,
        "manifest": manifest,
    }
