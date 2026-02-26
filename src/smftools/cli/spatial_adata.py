from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from smftools.constants import LOGGING_DIR, SPATIAL_DIR
from smftools.logging_utils import get_logger, setup_logging

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
    from ..readwrite import safe_read_h5ad
    from .helpers import get_adata_paths, load_experiment_config, resolve_adata_stage

    # 1) Ensure config + basic paths via load_adata
    cfg = load_experiment_config(config_path)

    paths = get_adata_paths(cfg)

    spatial_path = paths.spatial

    # Stage-skipping logic for spatial
    if not getattr(cfg, "force_redo_spatial_analyses", False):
        # If spatial exists, we consider spatial analyses already done.
        if spatial_path.exists():
            logger.info(f"Spatial AnnData found: {spatial_path}\nSkipping smftools spatial")
            return None, spatial_path

    # Decide which AnnData to use as the *starting point* for spatial analyses
    source_path, stage = resolve_adata_stage(cfg, paths, min_stage="pp")
    if source_path is None:
        logger.warning(
            "No suitable AnnData found for spatial analyses (need at least preprocessed)."
        )
        return None, None

    start_adata, _ = safe_read_h5ad(source_path)

    # 4) Run the spatial core
    adata_spatial, spatial_path = spatial_adata_core(
        adata=start_adata,
        cfg=cfg,
        spatial_adata_path=spatial_path,
        pp_adata_path=paths.pp,
        source_adata_path=source_path,
        config_path=config_path,
    )

    return adata_spatial, spatial_path


def spatial_adata_core(
    adata: ad.AnnData,
    cfg,
    spatial_adata_path: Path,
    pp_adata_path: Path,
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
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from ..metadata import record_smftools_metadata
    from ..plotting import (
        combined_raw_clustermap,
        plot_rolling_grid,
        plot_spatial_autocorr_grid,
    )
    from ..preprocessing import (
        invert_adata,
        load_sample_sheet,
        reindex_references_adata,
    )
    from ..readwrite import make_dirs, safe_read_h5ad
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
    date_str = datetime.today().strftime("%y%m%d")
    now = datetime.now()
    time_str = now.strftime("%H%M%S")
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    output_directory = Path(cfg.output_directory)
    spatial_directory = output_directory / SPATIAL_DIR
    logging_directory = spatial_directory / LOGGING_DIR

    make_dirs([output_directory, spatial_directory])

    if cfg.emit_log_file:
        log_file = logging_directory / f"{date_str}_{time_str}_log.log"
        make_dirs([logging_directory])
    else:
        log_file = None

    setup_logging(level=log_level, log_file=log_file, reconfigure=log_file is not None)

    smf_modality = cfg.smf_modality
    if smf_modality == "conversion":
        deaminase = False
    else:
        deaminase = True

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

    references = adata.obs[cfg.reference_column].cat.categories

    # ============================================================
    # 1) Clustermaps (non-direct modalities) on preprocessed adata
    # ============================================================
    if smf_modality != "direct":
        preprocessed_version_available = pp_adata_path.exists()

        if preprocessed_version_available:
            pp_clustermap_dir = spatial_directory / "01_clustermaps"

            if pp_clustermap_dir.is_dir() and not getattr(
                cfg, "force_redo_spatial_analyses", False
            ):
                logger.debug(
                    f"{pp_clustermap_dir} already exists. Skipping clustermap plotting for preprocessed AnnData."
                )
            else:
                make_dirs([spatial_directory, pp_clustermap_dir])

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
                    min_length=cfg.mapped_len_filter_thresholds[0],
                    min_mapped_length_to_reference_length_ratio=cfg.mapped_len_to_ref_ratio_filter_thresholds[
                        0
                    ],
                    min_position_valid_fraction=1 - cfg.position_max_nan_threshold,
                    demux_types=cfg.clustermap_demux_types_to_plot,
                    bins=None,
                    sample_mapping=None,
                    save_path=pp_clustermap_dir,
                    sort_by=cfg.spatial_clustermap_sortby,
                    deaminase=deaminase,
                    index_col_suffix=reindex_suffix,
                    n_jobs=cfg.threads or 1,
                    omit_chimeric_reads=cfg.omit_chimeric_reads,
                )

    # ============================================================
    # 2) Clustermaps on deduplicated preprocessed AnnDatas
    # ============================================================
    spatial_dir_dedup = spatial_directory / "deduplicated"
    clustermap_dir_dedup = spatial_dir_dedup / "01_clustermaps"

    # Clustermaps on deduplicated adata
    if clustermap_dir_dedup.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(
            f"{clustermap_dir_dedup} already exists. Skipping clustermap plotting for deduplicated AnnData."
        )
    else:
        make_dirs([spatial_dir_dedup, clustermap_dir_dedup])
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
            min_length=cfg.mapped_len_filter_thresholds[0],
            min_mapped_length_to_reference_length_ratio=cfg.mapped_len_to_ref_ratio_filter_thresholds[
                0
            ],
            min_position_valid_fraction=1 - cfg.position_max_nan_threshold,
            demux_types=cfg.clustermap_demux_types_to_plot,
            bins=None,
            sample_mapping=None,
            save_path=clustermap_dir_dedup,
            sort_by=cfg.spatial_clustermap_sortby,
            deaminase=deaminase,
            index_col_suffix=reindex_suffix,
            n_jobs=cfg.threads or 1,
            omit_chimeric_reads=cfg.omit_chimeric_reads,
        )

    # ============================================================
    # 3) Spatial autocorrelation + rolling metrics
    # ============================================================
    pp_autocorr_dir = spatial_dir_dedup / "02_autocorrelations"

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

        # Build chimeric-read exclusion mask (True = keep)
        if getattr(cfg, "omit_chimeric_reads", False) and "chimeric_variant_sites" in adata.obs.columns:
            _keep = ~adata.obs["chimeric_variant_sites"].astype(bool).values
        else:
            _keep = np.ones(adata.n_obs, dtype=bool)

        for site_type in cfg.autocorr_site_types:
            layer_key = f"{site_type}_site_binary"
            if layer_key not in adata.layers:
                logger.debug(f"Layer {layer_key} not found in adata.layers — skipping {site_type}.")
                continue

            X = adata.layers[layer_key]
            if not _keep.all():
                X = X[_keep]
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

                from joblib import parallel_config

                with parallel_config(backend="loky", inner_max_num_threads=1):
                    res = Parallel(n_jobs=cfg.threads or 1)(
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

            if _keep.all():
                adata.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
                adata.obsm[f"{site_type}_spatial_autocorr_counts"] = counts_matrix
            else:
                full_ac = np.full((adata.n_obs, autocorr_matrix.shape[1]), np.nan, dtype=np.float32)
                full_ac[_keep] = autocorr_matrix
                adata.obsm[f"{site_type}_spatial_autocorr"] = full_ac
                full_cnt = np.zeros((adata.n_obs, counts_matrix.shape[1]), dtype=np.int32)
                full_cnt[_keep] = counts_matrix
                adata.obsm[f"{site_type}_spatial_autocorr_counts"] = full_cnt
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
                mask = sample_mask[_keep]
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
                    mask_ref = (sample_mask & (adata.obs[ref_col].values == ref))[_keep]
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
                "n_jobs": cfg.threads or 1,
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
                group_masks = [("all", sample_mask[_keep])]
                for ref in refs:
                    ref_mask = (sample_mask & (adata.obs[ref_col].values == ref))[_keep]
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
    corr_dir = spatial_dir_dedup / "03_correlation_matrices"

    if corr_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{corr_dir} already exists. Skipping correlation matrix plotting.")
    else:
        _corr_keep = ~adata.obs["chimeric_variant_sites"].astype(bool).values if (getattr(cfg, "omit_chimeric_reads", False) and "chimeric_variant_sites" in adata.obs.columns) else None
        _corr_adata = adata[_corr_keep, :] if _corr_keep is not None and not _corr_keep.all() else adata
        compute_positionwise_statistics(
            _corr_adata,
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
            _corr_adata,
            methods=cfg.correlation_matrix_types,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            figsize_per_cell=(4.0, 3.0),
            dpi=160,
            cmaps=cfg.correlation_matrix_cmaps,
            vmin=None,
            vmax=None,
            output_dir=corr_dir,
            output_key="positionwise_result",
        )

    # ============================================================
    # 4) Save spatial AnnData
    # ============================================================
    if (not spatial_adata_path.exists()) or getattr(cfg, "force_redo_spatial_analyses", False):
        logger.info("Saving spatial analyzed AnnData.")
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
