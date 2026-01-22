from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

logger = get_logger(__name__)


def spatial_adata(
    config_path: str,
) -> Tuple[Optional[ad.AnnData], Optional[Path]]:
    """
    CLI-facing wrapper for spatial analyses.

    Called by: `smftools spatial <config_path>`

    Responsibilities:
    - Ensure a usable AnnData exists via `load_adata` + `preprocess_adata`.
    - Determine which AnnData stages exist (raw, pp, pp_dedup, spatial, hmm).
    - Respect cfg.force_redo_spatial_analyses.
    - Decide whether to skip (return existing) or run the spatial core.
    - Call `spatial_adata_core(...)` when actual work is needed.

    Returns
    -------
    spatial_adata : AnnData | None
        AnnData with spatial analyses, or None if we skipped because a later-stage
        AnnData already exists.
    spatial_adata_path : Path | None
        Path to the “current” spatial AnnData (or hmm AnnData if we skip to that).
    """
    from ..readwrite import add_or_update_column_in_csv, safe_read_h5ad
    from .helpers import get_adata_paths
    from .load_adata import load_adata
    from .preprocess_adata import preprocess_adata

    # 1) Ensure config + basic paths via load_adata
    loaded_adata, loaded_path, cfg = load_adata(config_path)
    paths = get_adata_paths(cfg)

    raw_path = paths.raw
    pp_path = paths.pp
    pp_dedup_path = paths.pp_dedup
    spatial_path = paths.spatial
    hmm_path = paths.hmm

    # Stage-skipping logic for spatial
    if not getattr(cfg, "force_redo_spatial_analyses", False):
        # If HMM exists, it's the most processed stage — reuse it.
        if hmm_path.exists():
            logger.info(f"HMM AnnData found: {hmm_path}\nSkipping smftools spatial")
            return None, hmm_path

        # If spatial exists, we consider spatial analyses already done.
        if spatial_path.exists():
            logger.info(f"Spatial AnnData found: {spatial_path}\nSkipping smftools spatial")
            return None, spatial_path

    # 2) Ensure preprocessing has been run
    #    This will create pp/pp_dedup as needed or return them if they already exist.
    pp_adata, pp_adata_path_ret, pp_dedup_adata, pp_dedup_adata_path_ret = preprocess_adata(
        config_path
    )

    # Helper to load from disk, reusing loaded_adata if it matches
    def _load(path: Path):
        if loaded_adata is not None and loaded_path == path:
            return loaded_adata
        adata, _ = safe_read_h5ad(path)
        return adata

    # 3) Decide which AnnData to use as the *starting point* for spatial analyses
    # Prefer in-memory pp_dedup_adata when preprocess_adata just ran.
    if pp_dedup_adata is not None:
        start_adata = pp_dedup_adata
        source_path = pp_dedup_adata_path_ret
    else:
        if pp_dedup_path.exists():
            start_adata = _load(pp_dedup_path)
            source_path = pp_dedup_path
        elif pp_path.exists():
            start_adata = _load(pp_path)
            source_path = pp_path
        elif raw_path.exists():
            start_adata = _load(raw_path)
            source_path = raw_path
        else:
            logger.warning("No suitable AnnData found for spatial analyses (need at least raw).")
            return None, None

    # 4) Run the spatial core
    adata_spatial, spatial_path = spatial_adata_core(
        adata=start_adata,
        cfg=cfg,
        spatial_adata_path=spatial_path,
        pp_adata_path=pp_path,
        pp_dup_rem_adata_path=pp_dedup_path,
        pp_adata_in_memory=pp_adata,
        source_adata_path=source_path,
        config_path=config_path,
    )

    # 5) Register spatial path in summary CSV
    add_or_update_column_in_csv(cfg.summary_file, "spatial_adata", spatial_path)

    return adata_spatial, spatial_path


def spatial_adata_core(
    adata: ad.AnnData,
    cfg,
    spatial_adata_path: Path,
    pp_adata_path: Path,
    pp_dup_rem_adata_path: Path,
    pp_adata_in_memory: Optional[ad.AnnData] = None,
    source_adata_path: Optional[Path] = None,
    config_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Path]:
    """
    Core spatial analysis pipeline.

    Assumes:
    - `adata` is (typically) the preprocessed, duplicate-removed AnnData.
    - `cfg` is the ExperimentConfig.
    - `spatial_adata_path`, `pp_adata_path`, `pp_dup_rem_adata_path` are canonical paths
      from `get_adata_paths`.
    - `pp_adata_in_memory` optionally holds the preprocessed (non-dedup) AnnData from
      the same run of `preprocess_adata`, to avoid re-reading from disk.

    Does:
    - Optional sample sheet load.
    - Optional inversion & reindexing.
    - Clustermaps on:
        * preprocessed (non-dedup) AnnData (for non-direct modalities), and
        * deduplicated preprocessed AnnData.
    - PCA/UMAP/Leiden.
    - Autocorrelation + rolling metrics + grids.
    - Positionwise correlation matrices (non-direct modalities).
    - Save spatial AnnData to `spatial_adata_path`.

    Returns
    -------
    adata : AnnData
        Spatially analyzed AnnData (same object, modified in-place).
    spatial_adata_path : Path
        Path where spatial AnnData was written.
    """
    import os
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sc = require("scanpy", extra="scanpy", purpose="spatial analyses")

    from ..metadata import record_smftools_metadata
    from ..plotting import (
        combined_raw_clustermap,
        plot_rolling_nn_and_layer,
        plot_rolling_grid,
        plot_spatial_autocorr_grid,
    )
    from ..preprocessing import (
        invert_adata,
        load_sample_sheet,
        reindex_references_adata,
    )
    from ..readwrite import make_dirs, safe_read_h5ad
    from ..tools import calculate_umap, rolling_window_nn_distance
    from ..tools.position_stats import (
        compute_positionwise_statistics,
        plot_positionwise_matrices,
    )
    from ..tools.spatial_autocorrelation import (
        analyze_autocorr_matrix,
        binary_autocorrelation_with_spacing,
        bootstrap_periodicity,
        rolling_autocorr_metrics,
    )
    from .helpers import write_gz_h5ad

    # -----------------------------
    # General setup
    # -----------------------------
    output_directory = Path(cfg.output_directory)
    make_dirs([output_directory])

    smf_modality = cfg.smf_modality
    if smf_modality == "conversion":
        deaminase = False
    else:
        deaminase = True

    first_pp_run = pp_adata_in_memory is not None and pp_dup_rem_adata_path.exists()

    # -----------------------------
    # Optional sample sheet metadata
    # -----------------------------
    if getattr(cfg, "sample_sheet_path", None):
        load_sample_sheet(
            adata,
            cfg.sample_sheet_path,
            mapping_key_column=cfg.sample_sheet_mapping_column,
            as_category=True,
            force_reload=cfg.force_reload_sample_sheet,
        )

    # -----------------------------
    # Optional inversion along positions axis
    # -----------------------------
    if getattr(cfg, "invert_adata", False):
        adata = invert_adata(adata)

    # -----------------------------
    # Optional reindexing by reference
    # -----------------------------
    reindex_references_adata(
        adata,
        reference_col=cfg.reference_column,
        offsets=cfg.reindexing_offsets,
        new_col=cfg.reindexed_var_suffix,
    )

    if adata.uns.get("reindex_references_adata_performed", False):
        reindex_suffix = cfg.reindexed_var_suffix
    else:
        reindex_suffix = None

    pp_dir = output_directory / "preprocessed"
    references = adata.obs[cfg.reference_column].cat.categories

    # ============================================================
    # 1) Clustermaps (non-direct modalities) on *preprocessed* data
    # ============================================================
    if smf_modality != "direct":
        preprocessed_version_available = pp_adata_path.exists()

        if preprocessed_version_available:
            pp_clustermap_dir = pp_dir / "06_clustermaps"

            if pp_clustermap_dir.is_dir() and not getattr(
                cfg, "force_redo_spatial_analyses", False
            ):
                logger.debug(
                    f"{pp_clustermap_dir} already exists. Skipping clustermap plotting for preprocessed AnnData."
                )
            else:
                make_dirs([pp_dir, pp_clustermap_dir])

                if first_pp_run and (pp_adata_in_memory is not None):
                    pp_adata = pp_adata_in_memory
                else:
                    pp_adata, _ = safe_read_h5ad(pp_adata_path)

                # -----------------------------
                # Optional sample sheet metadata
                # -----------------------------
                if getattr(cfg, "sample_sheet_path", None):
                    load_sample_sheet(
                        pp_adata,
                        cfg.sample_sheet_path,
                        mapping_key_column=cfg.sample_sheet_mapping_column,
                        as_category=True,
                        force_reload=cfg.force_reload_sample_sheet,
                    )

                # -----------------------------
                # Optional inversion along positions axis
                # -----------------------------
                if getattr(cfg, "invert_adata", False):
                    pp_adata = invert_adata(pp_adata)

                # -----------------------------
                # Optional reindexing by reference
                # -----------------------------
                reindex_references_adata(
                    pp_adata,
                    reference_col=cfg.reference_column,
                    offsets=cfg.reindexing_offsets,
                    new_col=cfg.reindexed_var_suffix,
                )

                combined_raw_clustermap(
                    pp_adata,
                    sample_col=cfg.sample_name_col_for_plotting,
                    reference_col=cfg.reference_column,
                    mod_target_bases=cfg.mod_target_bases,
                    layer_c=cfg.layer_for_clustermap_plotting,
                    layer_gpc=cfg.layer_for_clustermap_plotting,
                    layer_cpg=cfg.layer_for_clustermap_plotting,
                    layer_a=cfg.layer_for_clustermap_plotting,
                    cmap_c=cfg.clustermap_cmap_c,
                    cmap_gpc=cfg.clustermap_cmap_gpc,
                    cmap_cpg=cfg.clustermap_cmap_cpg,
                    cmap_a=cfg.clustermap_cmap_a,
                    min_quality=cfg.read_quality_filter_thresholds[0],
                    min_length=cfg.read_len_filter_thresholds[0],
                    min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[
                        0
                    ],
                    min_position_valid_fraction=cfg.min_valid_fraction_positions_in_read_vs_ref,
                    demux_types=cfg.clustermap_demux_types_to_plot,
                    bins=None,
                    sample_mapping=None,
                    save_path=pp_clustermap_dir,
                    sort_by=cfg.spatial_clustermap_sortby,
                    deaminase=deaminase,
                    index_col_suffix=reindex_suffix,
                )

    # ============================================================
    # 2) Clustermaps + UMAP on *deduplicated* preprocessed AnnData
    # ============================================================
    pp_dir_dedup = pp_dir / "deduplicated"
    pp_clustermap_dir_dedup = pp_dir_dedup / "06_clustermaps"
    pp_umap_dir = pp_dir_dedup / "07_umaps"

    # Clustermaps on deduplicated adata
    if pp_clustermap_dir_dedup.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(
            f"{pp_clustermap_dir_dedup} already exists. Skipping clustermap plotting for deduplicated AnnData."
        )
    else:
        make_dirs([pp_dir_dedup, pp_clustermap_dir_dedup])
        combined_raw_clustermap(
            adata,
            sample_col=cfg.sample_name_col_for_plotting,
            reference_col=cfg.reference_column,
            mod_target_bases=cfg.mod_target_bases,
            layer_c=cfg.layer_for_clustermap_plotting,
            layer_gpc=cfg.layer_for_clustermap_plotting,
            layer_cpg=cfg.layer_for_clustermap_plotting,
            layer_a=cfg.layer_for_clustermap_plotting,
            cmap_c=cfg.clustermap_cmap_c,
            cmap_gpc=cfg.clustermap_cmap_gpc,
            cmap_cpg=cfg.clustermap_cmap_cpg,
            cmap_a=cfg.clustermap_cmap_a,
            min_quality=cfg.read_quality_filter_thresholds[0],
            min_length=cfg.read_len_filter_thresholds[0],
            min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[
                0
            ],
            min_position_valid_fraction=1 - cfg.position_max_nan_threshold,
            demux_types=cfg.clustermap_demux_types_to_plot,
            bins=None,
            sample_mapping=None,
            save_path=pp_clustermap_dir_dedup,
            sort_by=cfg.spatial_clustermap_sortby,
            deaminase=deaminase,
            index_col_suffix=reindex_suffix,
        )

    # ============================================================
    # 2b) Rolling NN distances + layer clustermaps
    # ============================================================
    pp_rolling_nn_dir = pp_dir_dedup / "06b_rolling_nn_clustermaps"

    if pp_rolling_nn_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(
            f"{pp_rolling_nn_dir} already exists. Skipping rolling NN distance plots."
        )
    else:
        make_dirs([pp_rolling_nn_dir])
        samples = (
            adata.obs[cfg.sample_name_col_for_plotting]
            .astype("category")
            .cat.categories.tolist()
        )
        references = adata.obs[cfg.reference_column].astype("category").cat.categories.tolist()

        for reference in references:
            for sample in samples:
                mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & (
                    adata.obs[cfg.reference_column] == reference
                )
                if not mask.any():
                    continue

                subset = adata[mask].copy()
                try:
                    rolling_window_nn_distance(
                        subset,
                        layer=cfg.rolling_nn_layer,
                        window=cfg.rolling_nn_window,
                        step=cfg.rolling_nn_step,
                        min_overlap=cfg.rolling_nn_min_overlap,
                        return_fraction=cfg.rolling_nn_return_fraction,
                        store_obsm=cfg.rolling_nn_obsm_key,
                        site_types=cfg.rolling_nn_site_types,
                        reference=str(reference),
                    )
                except Exception as exc:
                    logger.warning(
                        "Rolling NN distance computation failed for sample=%s ref=%s: %s",
                        sample,
                        reference,
                        exc,
                    )
                    continue

                safe_sample = str(sample).replace(os.sep, "_")
                safe_ref = str(reference).replace(os.sep, "_")
                out_png = pp_rolling_nn_dir / f"{safe_sample}__{safe_ref}.png"
                title = f"{sample} {reference}"
                try:
                    plot_rolling_nn_and_layer(
                        subset,
                        rolling_obsm_key=cfg.rolling_nn_obsm_key,
                        layer=cfg.rolling_nn_plot_layer,
                        sample=sample,
                        reference=reference,
                        sample_col=cfg.sample_name_col_for_plotting,
                        reference_col=cfg.reference_column,
                        out_path=out_png,
                        title=title,
                        site_types=cfg.rolling_nn_site_types,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed rolling NN plot for sample=%s ref=%s: %s",
                        sample,
                        reference,
                        exc,
                    )

    # UMAP / Leiden
    if pp_umap_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{pp_umap_dir} already exists. Skipping UMAP plotting.")
    else:
        make_dirs([pp_umap_dir])

        var_filters = []
        if smf_modality == "direct":
            for ref in references:
                for base in cfg.mod_target_bases:
                    var_filters.append(f"{ref}_{base}_site")
        elif deaminase:
            for ref in references:
                var_filters.append(f"{ref}_C_site")
        else:
            for ref in references:
                for base in cfg.mod_target_bases:
                    var_filters.append(f"{ref}_{base}_site")

        adata = calculate_umap(
            adata,
            layer=cfg.layer_for_umap_plotting,
            var_filters=var_filters,
            n_pcs=10,
            knn_neighbors=15,
        )

        sc.tl.leiden(adata, resolution=0.1, flavor="igraph", n_iterations=2)

        sc.settings.figdir = pp_umap_dir
        umap_layers = ["leiden", cfg.sample_name_col_for_plotting, "Reference_strand"]
        umap_layers += cfg.umap_layers_to_plot
        sc.pl.umap(adata, color=umap_layers, show=False, save=True)

    # ============================================================
    # 3) Spatial autocorrelation + rolling metrics
    # ============================================================
    pp_autocorr_dir = pp_dir_dedup / "08_autocorrelations"

    if pp_autocorr_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{pp_autocorr_dir} already exists. Skipping autocorrelation plotting.")
    else:
        positions = adata.var_names.astype(int).values
        lags = np.arange(cfg.autocorr_max_lag + 1)

        try:
            from joblib import Parallel, delayed

            _have_joblib = True
        except Exception:
            _have_joblib = False

        samples = (
            adata.obs[cfg.sample_name_col_for_plotting].astype("category").cat.categories.tolist()
        )
        ref_col = getattr(cfg, "reference_strand_col", "Reference_strand")
        refs = adata.obs[ref_col].astype("category").cat.categories.tolist()

        for site_type in cfg.autocorr_site_types:
            layer_key = f"{site_type}_site_binary"
            if layer_key not in adata.layers:
                logger.debug(f"Layer {layer_key} not found in adata.layers — skipping {site_type}.")
                continue

            X = adata.layers[layer_key]
            if getattr(X, "shape", (0,))[0] == 0:
                logger.debug(f"Layer {layer_key} empty — skipping {site_type}.")
                continue

            rows = []
            counts = []

            if _have_joblib:

                def _worker(row):
                    try:
                        ac, cnts = binary_autocorrelation_with_spacing(
                            row,
                            positions,
                            max_lag=cfg.autocorr_max_lag,
                            return_counts=True,
                            normalize=cfg.autocorr_normalization_method,
                        )
                    except Exception:
                        ac = np.full(cfg.autocorr_max_lag + 1, np.nan, dtype=np.float32)
                        cnts = np.zeros(cfg.autocorr_max_lag + 1, dtype=np.int32)
                    return ac, cnts

                res = Parallel(n_jobs=getattr(cfg, "n_jobs", -1))(
                    delayed(_worker)(X[i]) for i in range(X.shape[0])
                )
                for ac, cnts in res:
                    rows.append(ac)
                    counts.append(cnts)
            else:
                for i in range(X.shape[0]):
                    ac, cnts = binary_autocorrelation_with_spacing(
                        X[i],
                        positions,
                        max_lag=cfg.autocorr_max_lag,
                        return_counts=True,
                        normalize=cfg.autocorr_normalization_method,
                    )
                    rows.append(ac)
                    counts.append(cnts)

            autocorr_matrix = np.asarray(rows, dtype=np.float32)
            counts_matrix = np.asarray(counts, dtype=np.int32)

            adata.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
            adata.obsm[f"{site_type}_spatial_autocorr_counts"] = counts_matrix
            adata.uns[f"{site_type}_spatial_autocorr_lags"] = lags

            try:
                results = analyze_autocorr_matrix(
                    autocorr_matrix,
                    counts_matrix,
                    lags,
                    nrl_search_bp=(120, 260),
                    pad_factor=4,
                    min_count=20,
                    max_harmonics=6,
                )
            except Exception as e:
                results = {"error": str(e)}

            global_metrics = {
                "nrl_bp": results.get("nrl_bp", np.nan),
                "xi": results.get("xi", np.nan),
                "snr": results.get("snr", np.nan),
                "fwhm_bp": results.get("fwhm_bp", np.nan),
                "envelope_sample_lags": results.get("envelope_sample_lags", np.array([])).tolist(),
                "envelope_heights": results.get("envelope_heights", np.array([])).tolist(),
                "analyzer_error": results.get("error", None),
            }
            adata.uns[f"{site_type}_spatial_periodicity_metrics"] = global_metrics

            n_boot = getattr(cfg, "autocorr_bootstrap_n", 200)
            try:
                bs = bootstrap_periodicity(
                    autocorr_matrix,
                    counts_matrix,
                    lags,
                    n_boot=n_boot,
                    nrl_search_bp=(120, 260),
                    pad_factor=4,
                    min_count=20,
                )
                adata.uns[f"{site_type}_spatial_periodicity_boot"] = {
                    "nrl_boot": np.asarray(bs["nrl_boot"]).tolist(),
                    "xi_boot": np.asarray(bs["xi_boot"]).tolist(),
                }
            except Exception as e:
                adata.uns[f"{site_type}_spatial_periodicity_boot_error"] = str(e)

            metrics_by_group = {}
            sample_col = cfg.sample_name_col_for_plotting

            for sample_name in samples:
                sample_mask = adata.obs[sample_col].values == sample_name

                # combined group
                mask = sample_mask
                ac_sel = autocorr_matrix[mask, :]
                cnt_sel = counts_matrix[mask, :] if counts_matrix is not None else None
                if ac_sel.size:
                    try:
                        r = analyze_autocorr_matrix(
                            ac_sel,
                            cnt_sel if cnt_sel is not None else np.zeros_like(ac_sel, dtype=int),
                            lags,
                            nrl_search_bp=(120, 260),
                            pad_factor=4,
                            min_count=10,
                            max_harmonics=6,
                        )
                    except Exception as e:
                        r = {"error": str(e)}
                else:
                    r = {"error": "no_data"}
                metrics_by_group[(sample_name, None)] = r

                for ref in refs:
                    mask_ref = sample_mask & (adata.obs[ref_col].values == ref)
                    ac_sel = autocorr_matrix[mask_ref, :]
                    cnt_sel = counts_matrix[mask_ref, :] if counts_matrix is not None else None
                    if ac_sel.size:
                        try:
                            r = analyze_autocorr_matrix(
                                ac_sel,
                                cnt_sel
                                if cnt_sel is not None
                                else np.zeros_like(ac_sel, dtype=int),
                                lags,
                                nrl_search_bp=(120, 260),
                                pad_factor=4,
                                min_count=10,
                                max_harmonics=6,
                            )
                        except Exception as e:
                            r = {"error": str(e)}
                    else:
                        r = {"error": "no_data"}
                    metrics_by_group[(sample_name, ref)] = r

            adata.uns[f"{site_type}_spatial_periodicity_metrics_by_group"] = metrics_by_group

            global_nrl = adata.uns.get(f"{site_type}_spatial_periodicity_metrics", {}).get(
                "nrl_bp", None
            )

            rolling_cfg = {
                "window_size": getattr(
                    cfg,
                    "rolling_window_size",
                    getattr(cfg, "autocorr_rolling_window_size", 600),
                ),
                "step": getattr(cfg, "rolling_step", 100),
                "max_lag": getattr(
                    cfg,
                    "rolling_max_lag",
                    getattr(cfg, "autocorr_max_lag", 500),
                ),
                "min_molecules_per_window": getattr(cfg, "rolling_min_molecules_per_window", 10),
                "nrl_search_bp": getattr(cfg, "rolling_nrl_search_bp", (120, 240)),
                "pad_factor": getattr(cfg, "rolling_pad_factor", 4),
                "min_count_for_mean": getattr(cfg, "rolling_min_count_for_mean", 10),
                "max_harmonics": getattr(cfg, "rolling_max_harmonics", 6),
                "n_jobs": getattr(cfg, "rolling_n_jobs", 4),
            }

            write_plots = getattr(cfg, "rolling_write_plots", True)
            write_csvs = getattr(cfg, "rolling_write_csvs", True)
            min_molecules_for_group = getattr(cfg, "rolling_min_molecules_for_group", 30)

            rolling_out_dir = os.path.join(pp_autocorr_dir, "rolling_metrics")
            os.makedirs(rolling_out_dir, exist_ok=True)
            site_out_dir = os.path.join(rolling_out_dir, site_type)
            os.makedirs(site_out_dir, exist_ok=True)

            combined_rows = []
            rolling_results_by_group = {}

            for sample_name in samples:
                sample_mask = adata.obs[sample_col].values == sample_name
                group_masks = [("all", sample_mask)]
                for ref in refs:
                    ref_mask = sample_mask & (adata.obs[ref_col].values == ref)
                    group_masks.append((ref, ref_mask))

                for ref_label, mask in group_masks:
                    n_group = int(mask.sum())
                    if n_group < min_molecules_for_group:
                        continue

                    X_group = X[mask, :]
                    try:
                        df_roll = rolling_autocorr_metrics(
                            X_group,
                            positions,
                            site_label=site_type,
                            window_size=rolling_cfg["window_size"],
                            step=rolling_cfg["step"],
                            max_lag=rolling_cfg["max_lag"],
                            min_molecules_per_window=rolling_cfg["min_molecules_per_window"],
                            nrl_search_bp=rolling_cfg["nrl_search_bp"],
                            pad_factor=rolling_cfg["pad_factor"],
                            min_count_for_mean=rolling_cfg["min_count_for_mean"],
                            max_harmonics=rolling_cfg["max_harmonics"],
                            n_jobs=rolling_cfg["n_jobs"],
                            verbose=False,
                            fixed_nrl_bp=global_nrl,
                        )
                    except Exception as e:
                        logger.warning(
                            f"rolling_autocorr_metrics failed for {site_type} "
                            f"{sample_name} {ref_label}: {e}"
                        )
                        continue

                    if "center" not in df_roll.columns:
                        logger.warning(
                            f"rolling_autocorr_metrics returned unexpected schema "
                            f"for {site_type} {sample_name} {ref_label}"
                        )
                        continue

                    compact_df = df_roll[
                        ["center", "n_molecules", "nrl_bp", "snr", "xi", "fwhm_bp"]
                    ].copy()
                    compact_df["site"] = site_type
                    compact_df["sample"] = sample_name
                    compact_df["reference"] = ref_label if ref_label != "all" else "all"

                    if write_csvs:
                        safe_sample = str(sample_name).replace(os.sep, "_")
                        safe_ref = str(ref_label if ref_label != "all" else "all").replace(
                            os.sep, "_"
                        )
                        out_csv = os.path.join(
                            site_out_dir,
                            f"{safe_sample}__{safe_ref}__rolling_metrics.csv",
                        )
                        try:
                            compact_df.to_csv(out_csv, index=False)
                        except Exception as e:
                            logger.warning(f"Failed to write rolling CSV {out_csv}: {e}")

                    if write_plots:
                        try:
                            from ..plotting import plot_rolling_metrics as _plot_roll
                        except Exception:
                            _plot_roll = None
                        if _plot_roll is not None:
                            plot_png = os.path.join(
                                site_out_dir,
                                f"{safe_sample}__{safe_ref}__rolling_metrics.png",
                            )
                            try:
                                _plot_roll(
                                    compact_df,
                                    out_png=plot_png,
                                    title=f"{site_type} {sample_name} {ref_label}",
                                    figsize=(10, 3.5),
                                    dpi=160,
                                    show=False,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to create rolling plot for {site_type} "
                                    f"{sample_name} {ref_label}: {e}"
                                )

                    combined_rows.append(
                        compact_df.assign(site=site_type, sample=sample_name, reference=ref_label)
                    )
                    rolling_results_by_group[
                        (sample_name, None if ref_label == "all" else ref_label)
                    ] = compact_df

            adata.uns[f"{site_type}_rolling_metrics_by_group"] = rolling_results_by_group

            if combined_rows:
                combined_df_site = pd.concat(combined_rows, ignore_index=True, sort=False)
                combined_out_csv = os.path.join(
                    rolling_out_dir, f"{site_type}__rolling_metrics_combined.csv"
                )
                try:
                    combined_df_site.to_csv(combined_out_csv, index=False)
                except Exception as e:
                    logger.warning(f"Failed to write combined rolling CSV for {site_type}: {e}")

            rolling_dict = adata.uns[f"{site_type}_rolling_metrics_by_group"]
            plot_out_dir = os.path.join(pp_autocorr_dir, "rolling_plots")
            os.makedirs(plot_out_dir, exist_ok=True)
            _ = plot_rolling_grid(
                rolling_dict,
                plot_out_dir,
                site_type,
                rows_per_page=cfg.rows_per_qc_autocorr_grid,
                cols_per_page=len(refs),
                dpi=160,
                metrics=("nrl_bp", "snr", "xi"),
                per_metric_ylim={"snr": (0, 25)},
            )

            make_dirs([pp_autocorr_dir])
            plot_spatial_autocorr_grid(
                adata,
                pp_autocorr_dir,
                site_types=cfg.autocorr_site_types,
                sample_col=cfg.sample_name_col_for_plotting,
                window=cfg.autocorr_rolling_window_size,
                rows_per_fig=cfg.rows_per_qc_autocorr_grid,
                normalization_method=cfg.autocorr_normalization_method,
            )

    # ============================================================
    # 4) Pearson / correlation matrices
    # ============================================================
    pp_corr_dir = pp_dir_dedup / "09_correlation_matrices"

    if pp_corr_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{pp_corr_dir} already exists. Skipping correlation matrix plotting.")
    else:
        compute_positionwise_statistics(
            adata,
            layer="nan0_0minus1",
            methods=cfg.correlation_matrix_types,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            output_key="positionwise_result",
            site_types=cfg.correlation_matrix_site_types,
            encoding="signed",
            max_threads=cfg.threads,
            min_count_for_pairwise=10,
        )

        plot_positionwise_matrices(
            adata,
            methods=cfg.correlation_matrix_types,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            figsize_per_cell=(4.0, 3.0),
            dpi=160,
            cmaps=cfg.correlation_matrix_cmaps,
            vmin=None,
            vmax=None,
            output_dir=pp_corr_dir,
            output_key="positionwise_result",
        )

    # ============================================================
    # 5) Save spatial AnnData
    # ============================================================
    if (not spatial_adata_path.exists()) or getattr(cfg, "force_redo_spatial_analyses", False):
        logger.info("Saving spatial analyzed AnnData (post preprocessing and duplicate removal).")
        record_smftools_metadata(
            adata,
            step_name="spatial",
            cfg=cfg,
            config_path=config_path,
            input_paths=[source_adata_path] if source_adata_path else None,
            output_path=spatial_adata_path,
        )
        write_gz_h5ad(adata, spatial_adata_path)

    return adata, spatial_adata_path
