def spatial_adata(config_path):
    """
    High-level function to call for spatial analysis of an adata object. 
    Command line accesses this through smftools spatial <config_path>

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        (pp_dedup_spatial_adata, pp_dedup_spatial_adata_path)
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad, make_dirs
    from ..config import LoadExperimentConfig, ExperimentConfig
    from .load_adata import load_adata
    from .preprocess_adata import preprocess_adata

    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc

    import os
    from importlib import resources
    from pathlib import Path

    from datetime import datetime
    date_str = datetime.today().strftime("%y%m%d")

    ################################### 1) General params and input organization ###################################
    # Load experiment config parameters into global variables
    loader = LoadExperimentConfig(config_path)
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, report = ExperimentConfig.from_var_dict(loader.var_dict, date_str=date_str, defaults_dir=defaults_dir)

    # General config variable init - Necessary user passed inputs
    smf_modality = cfg.smf_modality # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    input_data_path = Path(cfg.input_data_path)  # Path to a directory of POD5s/FAST5s or to a BAM/FASTQ file. Necessary.
    output_directory = Path(cfg.output_directory)  # Path to the output directory to make for the analysis. Necessary.
    fasta = Path(cfg.fasta)  # Path to reference FASTA. Necessary.
    split_dir = Path(cfg.split_dir) # Relative path to directory for demultiplexing reads
    split_path = output_directory / split_dir # Absolute path to directory for demultiplexing reads

    # Naming of the demultiplexed output directory
    if cfg.barcode_both_ends:
        split_path = split_path.with_name(split_path.name + '_both_ends_barcoded')
    else:
        split_path = split_path.with_name(split_path.name + '_at_least_one_end_barcoded')

    # Make initial output directory
    make_dirs([output_directory])

    bam_suffix = cfg.bam_suffix
    strands = cfg.strands

    # General config variable init - Optional user passed inputs for enzyme base specificity
    mod_target_bases = cfg.mod_target_bases  # Nucleobases of interest that may be modified. ['GpC', 'CpG', 'C', 'A']

    # Conversion/deamination specific variable init
    conversion_types = cfg.conversion_types  # 5mC
    conversions = cfg.conversions

    # Common Anndata accession params
    reference_column = cfg.reference_column

    # If conversion_types is passed:
    if conversion_types:
        conversions += conversion_types
    ########################################################################################################################

    ############################################### smftools load start ###############################################
    initial_adata, initial_adata_path = load_adata(config_path)

    # Initial adata path info
    initial_backup_dir = initial_adata_path.parent / 'adata_accessory_data'
    ############################################### smftools load end ###############################################

    ############################################### smftools preprocess start ###############################################
    pp_adata, pp_adata_path, pp_dedup_adata, pp_dup_rem_adata_path = preprocess_adata(config_path)

    # Preprocessed adata path info
    pp_backup_dir = pp_adata_path.parent / 'pp_adata_accessory_data'

    # Preprocessed duplicate removed adata path info
    pp_dup_rem_backup_dir= pp_adata_path.parent / 'pp_dedup_adata_accessory_data'
    ############################################### smftools preprocess end ###############################################

    ############################################### smftools spatial start ###############################################
    # Preprocessed duplicate removed adata with basic analyses appended path info
    spatial_adata_basename = pp_dup_rem_adata_path.name.split(".")[0] + '_spatial.h5ad.gz'
    spatial_adata_path = pp_dup_rem_adata_path.parent / spatial_adata_basename
    spatial_backup_dir= pp_dup_rem_adata_path.parent /'pp_dedup_spatial_adata_accessory_data'

    if pp_adata and pp_dedup_adata:
        # This happens on first run of the pipeline
        adata = pp_adata
        adata_unique = pp_dedup_adata
    else:
        # If an anndata is saved, check which stages of the anndata are available
        initial_version_available = initial_adata_path.exists() and initial_backup_dir.is_dir()
        preprocessed_version_available = pp_adata_path.exists() and pp_backup_dir.is_dir()
        preprocessed_dup_removed_version_available = pp_dup_rem_adata_path.exists() and pp_dup_rem_backup_dir.is_dir()
        preprocessed_dedup_spatial_version_available = spatial_adata_path.exists() and spatial_backup_dir.is_dir()

        if cfg.force_redo_basic_analyses:
            print(f"Forcing redo of basic analysis workflow, starting from the preprocessed adata if available. Otherwise, will use the raw adata.")
            if preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path, backup_dir=pp_dup_rem_backup_dir)
            elif preprocessed_version_available:
                adata, load_report = safe_read_h5ad(pp_adata_path, backup_dir=pp_backup_dir)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path, backup_dir=initial_backup_dir)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
                return 
        elif preprocessed_dedup_spatial_version_available:
            return None, spatial_adata_path
        else:
            print(f"No adata available.")
            return

    if smf_modality != 'direct':
        if smf_modality == 'conversion':
            deaminase = False
        else:
            deaminase = True
        references = adata.obs[reference_column].cat.categories

        ######### Clustermaps #########

        pp_dir = split_path / "preprocessed"
        pp_clustermap_dir = pp_dir / "06_clustermaps"

        if pp_clustermap_dir.is_dir():
            print(f'{pp_clustermap_dir} already exists. Skipping clustermap plotting.')
        else:
            from ..plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(adata, 
                                                         sample_col=cfg.sample_name_col_for_plotting, 
                                                         reference_col=cfg.reference_column,
                                                         layer_any_c=cfg.layer_for_clustermap_plotting, 
                                                         layer_gpc=cfg.layer_for_clustermap_plotting, 
                                                         layer_cpg=cfg.layer_for_clustermap_plotting, 
                                                         cmap_any_c="coolwarm", 
                                                         cmap_gpc="coolwarm", 
                                                         cmap_cpg="viridis", 
                                                         min_quality=cfg.read_quality_filter_thresholds[0], 
                                                         min_length=cfg.read_len_filter_thresholds[0], 
                                                         min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[0],
                                                         min_position_valid_fraction=cfg.min_valid_fraction_positions_in_read_vs_ref,
                                                         bins=None,
                                                         sample_mapping=None, 
                                                         save_path=pp_clustermap_dir, 
                                                         sort_by='gpc', 
                                                         deaminase=deaminase)
        
        # Switch the main adata moving forward to be the one with duplicates removed.
        adata = adata_unique

        #### Repeat on duplicate scrubbed anndata ###

        pp_dir = pp_dir / "deduplicated"
        pp_clustermap_dir = pp_dir / "06_clustermaps"
        pp_umap_dir = pp_dir / "07_umaps"

        if pp_clustermap_dir.is_dir():
            print(f'{pp_clustermap_dir} already exists. Skipping clustermap plotting.')
        else:
            from ..plotting import combined_raw_clustermap
            make_dirs([pp_dir, pp_clustermap_dir])
            clustermap_results = combined_raw_clustermap(adata, 
                                                         sample_col=cfg.sample_name_col_for_plotting, 
                                                         reference_col=cfg.reference_column,
                                                         layer_any_c=cfg.layer_for_clustermap_plotting, 
                                                         layer_gpc=cfg.layer_for_clustermap_plotting, 
                                                         layer_cpg=cfg.layer_for_clustermap_plotting, 
                                                         cmap_any_c="coolwarm", 
                                                         cmap_gpc="coolwarm", 
                                                         cmap_cpg="viridis", 
                                                         min_quality=cfg.read_quality_filter_thresholds[0], 
                                                         min_length=cfg.read_len_filter_thresholds[0], 
                                                         min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[0],
                                                         min_position_valid_fraction=cfg.min_valid_fraction_positions_in_read_vs_ref,
                                                         bins=None,
                                                         sample_mapping=None, 
                                                         save_path=pp_clustermap_dir, 
                                                         sort_by='gpc', 
                                                         deaminase=deaminase)
        
        ######### PCA/UMAP/Leiden #########
        if pp_umap_dir.is_dir():
            print(f'{pp_umap_dir} already exists. Skipping UMAP plotting.')
        else:
            from ..tools import calculate_umap
            make_dirs([pp_umap_dir])
            var_filters = []
            for ref in references:
                var_filters += [f'{ref}_any_C_site']
            adata = calculate_umap(adata, 
                                   layer=cfg.layer_for_umap_plotting, 
                                   var_filters=var_filters, 
                                   n_pcs=10, 
                                   knn_neighbors=15)

            ## Clustering
            sc.tl.leiden(adata, resolution=0.1, flavor="igraph", n_iterations=2)

            # Plotting UMAP
            sc.settings.figdir = pp_umap_dir
            umap_layers = ['leiden', cfg.sample_name_col_for_plotting]
            umap_layers += cfg.umap_layers_to_plot
            sc.pl.umap(adata, color=umap_layers, show=False, save=True)

    ########## Spatial autocorrelation analyses ###########
    from ..tools.spatial_autocorrelation import binary_autocorrelation_with_spacing, analyze_autocorr_matrix, bootstrap_periodicity, rolling_autocorr_metrics
    from ..plotting import plot_rolling_grid
    import warnings

    pp_autocorr_dir = pp_dir / "08_autocorrelations"

    if pp_autocorr_dir.is_dir():
        print(f'{pp_autocorr_dir} already exists. Skipping autocorrelation plotting.')
    else:
        positions = adata.var_names.astype(int).values
        lags = np.arange(cfg.autocorr_max_lag + 1)

        # optional: try to parallelize autocorr per-row with joblib
        try:
            from joblib import Parallel, delayed
            _have_joblib = True
        except Exception:
            _have_joblib = False

        for site_type in cfg.autocorr_site_types:
            layer_key = f"{site_type}_site_binary"
            if layer_key not in adata.layers:
                print(f"Layer {layer_key} not found in adata.layers — skipping {site_type}.")
                continue

            X = adata.layers[layer_key]
            if getattr(X, "shape", (0,))[0] == 0:
                print(f"Layer {layer_key} empty — skipping {site_type}.")
                continue

            # compute per-molecule autocorrs (and counts)
            rows = []
            counts = []
            if _have_joblib:
                # parallel map
                def _worker(row):
                    try:
                        ac, cnts = binary_autocorrelation_with_spacing(
                            row, positions, max_lag=cfg.autocorr_max_lag, return_counts=True
                        )
                    except Exception as e:
                        # on error return NaN arrays
                        ac = np.full(cfg.autocorr_max_lag + 1, np.nan, dtype=np.float32)
                        cnts = np.zeros(cfg.autocorr_max_lag + 1, dtype=np.int32)
                    return ac, cnts

                res = Parallel(n_jobs=cfg.n_jobs if hasattr(cfg, "n_jobs") else -1)(
                    delayed(_worker)(X[i]) for i in range(X.shape[0])
                )
                for ac, cnts in res:
                    rows.append(ac)
                    counts.append(cnts)
            else:
                # sequential fallback
                for i in range(X.shape[0]):
                    ac, cnts = binary_autocorrelation_with_spacing(
                        X[i], positions, max_lag=cfg.autocorr_max_lag, return_counts=True
                    )
                    rows.append(ac)
                    counts.append(cnts)

            autocorr_matrix = np.asarray(rows, dtype=np.float32)
            counts_matrix = np.asarray(counts, dtype=np.int32)

            # store raw per-molecule arrays (keep memory format compact)
            adata.obsm[f"{site_type}_spatial_autocorr"] = autocorr_matrix
            adata.obsm[f"{site_type}_spatial_autocorr_counts"] = counts_matrix
            adata.uns[f"{site_type}_spatial_autocorr_lags"] = lags

            # compute global periodicity metrics across all molecules for this site_type
            try:
                results = analyze_autocorr_matrix(
                    autocorr_matrix, counts_matrix, lags,
                    nrl_search_bp=(120, 260), pad_factor=4, min_count=20, max_harmonics=6
                )
            except Exception as e:
                results = {"error": str(e)}

            # store global metrics (same keys you used)
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

            # bootstrap for CI (use a reasonable default; set low only for debugging)
            n_boot = getattr(cfg, "autocorr_bootstrap_n", 200)
            # if user intentionally set very low n_boot in cfg, we keep that; otherwise default 200
            try:
                bs = bootstrap_periodicity(
                    autocorr_matrix, counts_matrix, lags,
                    n_boot=n_boot, nrl_search_bp=(120, 260), pad_factor=4, min_count=20
                )
                adata.uns[f"{site_type}_spatial_periodicity_boot"] = {
                    "nrl_boot": np.asarray(bs["nrl_boot"]).tolist(),
                    "xi_boot": np.asarray(bs["xi_boot"]).tolist(),
                }
            except Exception as e:
                adata.uns[f"{site_type}_spatial_periodicity_boot_error"] = str(e)

            # ----------------------------
            # Compute group-level metrics for plotting (per sample × reference)
            # ----------------------------
            metrics_by_group = {}
            sample_col = cfg.sample_name_col_for_plotting
            ref_col = cfg.reference_strand_col if hasattr(cfg, "reference_strand_col") else "Reference_strand"
            samples = adata.obs[sample_col].astype("category").cat.categories.tolist()
            refs = adata.obs[ref_col].astype("category").cat.categories.tolist()

            # iterate groups and run analyzer on each group's subset; cache errors
            for sample_name in samples:
                sample_mask = (adata.obs[sample_col].values == sample_name)
                # combined group
                mask = sample_mask
                ac_sel = autocorr_matrix[mask, :]
                cnt_sel = counts_matrix[mask, :] if counts_matrix is not None else None
                if ac_sel.size:
                    try:
                        r = analyze_autocorr_matrix(ac_sel, cnt_sel if cnt_sel is not None else np.zeros_like(ac_sel, dtype=int),
                                                    lags, nrl_search_bp=(120,260), pad_factor=4, min_count=10, max_harmonics=6)
                    except Exception as e:
                        r = {"error": str(e)}
                else:
                    r = {"error": "no_data"}
                metrics_by_group[(sample_name, None)] = r

                # per-reference groups
                for ref in refs:
                    mask_ref = sample_mask & (adata.obs[ref_col].values == ref)
                    ac_sel = autocorr_matrix[mask_ref, :]
                    cnt_sel = counts_matrix[mask_ref, :] if counts_matrix is not None else None
                    if ac_sel.size:
                        try:
                            r = analyze_autocorr_matrix(ac_sel, cnt_sel if cnt_sel is not None else np.zeros_like(ac_sel, dtype=int),
                                                        lags, nrl_search_bp=(120,260), pad_factor=4, min_count=10, max_harmonics=6)
                        except Exception as e:
                            r = {"error": str(e)}
                    else:
                        r = {"error": "no_data"}
                    metrics_by_group[(sample_name, ref)] = r

            # persist group metrics
            adata.uns[f"{site_type}_spatial_periodicity_metrics_by_group"] = metrics_by_group

            global_nrl = adata.uns.get(f"{site_type}_spatial_periodicity_metrics", {}).get("nrl_bp", None)

            # configuration / sensible defaults (override in cfg if present)
            rolling_cfg = {
                "window_size": getattr(cfg, "rolling_window_size", getattr(cfg, "autocorr_rolling_window_size", 600)),
                "step": getattr(cfg, "rolling_step", 100),
                "max_lag": getattr(cfg, "rolling_max_lag", cfg.autocorr_max_lag if hasattr(cfg, "autocorr_max_lag") else 500),
                "min_molecules_per_window": getattr(cfg, "rolling_min_molecules_per_window", 10),
                "nrl_search_bp": getattr(cfg, "rolling_nrl_search_bp", (120, 240)),
                "pad_factor": getattr(cfg, "rolling_pad_factor", 4),
                "min_count_for_mean": getattr(cfg, "rolling_min_count_for_mean", 10),
                "max_harmonics": getattr(cfg, "rolling_max_harmonics", 6),
                "n_jobs": getattr(cfg, "rolling_n_jobs", 4),
            }

            write_plots = getattr(cfg, "rolling_write_plots", True)
            write_csvs = getattr(cfg, "rolling_write_csvs", True)
            min_molecules_for_group = getattr(cfg, "rolling_min_molecules_for_group", 30)  # only run rolling if group has >= this many molecules

            rolling_out_dir = os.path.join(pp_autocorr_dir, "rolling_metrics")
            os.makedirs(rolling_out_dir, exist_ok=True)
            # also a per-site subfolder
            site_out_dir = os.path.join(rolling_out_dir, site_type)
            os.makedirs(site_out_dir, exist_ok=True)

            combined_rows = []  # accumulate one row per window for combined CSV
            rolling_results_by_group = {}  # store DataFrame per group in memory (persist later to adata.uns)

            # iterate groups (samples × refs). `samples` and `refs` were computed above.
            for sample_name in samples:
                sample_mask = (adata.obs[sample_col].values == sample_name)
                # first the combined group ("all refs")
                group_masks = [("all", sample_mask)]
                # then per-reference groups
                for ref in refs:
                    ref_mask = sample_mask & (adata.obs[ref_col].values == ref)
                    group_masks.append((ref, ref_mask))

                for ref_label, mask in group_masks:
                    n_group = int(mask.sum())
                    if n_group < min_molecules_for_group:
                        # skip tiny groups
                        if cfg.get("verbosity", 0) if hasattr(cfg, "get") else False:
                            print(f"Skipping rolling for {site_type} {sample_name} {ref_label}: only {n_group} molecules (<{min_molecules_for_group})")
                        # still write an empty CSV row set if desired; here we skip
                        continue

                    # extract group matrix X_group (works with dense or sparse adata.layers)
                    X_group = X[mask, :]
                    # positions already set above
                    try:
                        # call your rolling function (this may be slow; it uses cfg.n_jobs)
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
                            fixed_nrl_bp=global_nrl
                        )
                    except Exception as e:
                        warnings.warn(f"rolling_autocorr_metrics failed for {site_type} {sample_name} {ref_label}: {e}")
                        continue

                    # normalize column names and keep only the compact set you want
                    # keep: center, n_molecules, nrl_bp, snr, xi, fwhm_bp
                    if "center" not in df_roll.columns:
                        # defensive: if the rolling function returned different schema, skip
                        warnings.warn(f"rolling_autocorr_metrics returned unexpected schema for {site_type} {sample_name} {ref_label}")
                        continue

                    compact_df = df_roll[["center", "n_molecules", "nrl_bp", "snr", "xi", "fwhm_bp"]].copy()
                    compact_df["site"] = site_type
                    compact_df["sample"] = sample_name
                    compact_df["reference"] = ref_label if ref_label != "all" else "all"

                    # save per-group CSV
                    if write_csvs:
                        safe_sample = str(sample_name).replace(os.sep, "_")
                        safe_ref = str(ref_label if ref_label != "all" else "all").replace(os.sep, "_")
                        out_csv = os.path.join(site_out_dir, f"{safe_sample}__{safe_ref}__rolling_metrics.csv")
                        try:
                            compact_df.to_csv(out_csv, index=False)
                        except Exception as e:
                            warnings.warn(f"Failed to write rolling CSV {out_csv}: {e}")

                    # save a plot per-group (NRL and SNR vs center)
                    if write_plots:
                        try:
                            # use your plot helper; if it's in a different module, import accordingly
                            from ..plotting import plot_rolling_metrics as _plot_roll
                        except Exception:
                            _plot_roll = globals().get("plot_rolling_metrics", None)
                        if _plot_roll is not None:
                            plot_png = os.path.join(site_out_dir, f"{safe_sample}__{safe_ref}__rolling_metrics.png")
                            try:
                                _plot_roll(compact_df, out_png=plot_png,
                                        title=f"{site_type} {sample_name} {ref_label}",
                                        figsize=(10,3.5), dpi=160, show=False)
                            except Exception as e:
                                warnings.warn(f"Failed to create rolling plot for {site_type} {sample_name} {ref_label}: {e}")

                    # store in combined_rows and in-memory dict
                    combined_rows.append(compact_df.assign(site=site_type, sample=sample_name, reference=ref_label))
                    rolling_results_by_group[(sample_name, None if ref_label == "all" else ref_label)] = compact_df

            # persist per-site rolling metrics into adata.uns as dict of DataFrames (or empty dict)
            adata.uns[f"{site_type}_rolling_metrics_by_group"] = rolling_results_by_group

            # write combined CSV for this site across all groups
            if len(combined_rows):
                combined_df_site = pd.concat(combined_rows, ignore_index=True, sort=False)
                combined_out_csv = os.path.join(rolling_out_dir, f"{site_type}__rolling_metrics_combined.csv")
                try:
                    combined_df_site.to_csv(combined_out_csv, index=False)
                except Exception as e:
                    warnings.warn(f"Failed to write combined rolling CSV for {site_type}: {e}")

            rolling_dict = adata.uns[f"{site_type}_rolling_metrics_by_group"]
            plot_out_dir = os.path.join(pp_autocorr_dir, "rolling_plots")
            os.makedirs(plot_out_dir, exist_ok=True)
            pages = plot_rolling_grid(rolling_dict, plot_out_dir, site_type,
                                    rows_per_page=cfg.rows_per_qc_autocorr_grid,
                                    cols_per_page=len(refs),
                                    dpi=160,
                                    metrics=("nrl_bp","snr", "xi"),
                                    per_metric_ylim={"snr": (0, 25)})

            from ..plotting import plot_spatial_autocorr_grid
            make_dirs([pp_autocorr_dir, pp_autocorr_dir])

            plot_spatial_autocorr_grid(adata, 
                                        pp_autocorr_dir, 
                                        site_types=cfg.autocorr_site_types, 
                                        sample_col=cfg.sample_name_col_for_plotting, 
                                        window=cfg.autocorr_rolling_window_size, 
                                        rows_per_fig=cfg.rows_per_qc_autocorr_grid)

    ############ Pearson analyses ###############
    if smf_modality != 'direct':
        from ..tools.position_stats import compute_positionwise_statistics, plot_positionwise_matrices

        pp_corr_dir = pp_dir / "09_correlation_matrices"

        if pp_corr_dir.is_dir():
            print(f'{pp_corr_dir} already exists. Skipping correlation matrix plotting.')
        else:
            compute_positionwise_statistics(
                adata,
                layer="nan0_0minus1",
                methods=cfg.correlation_matrix_types,
                sample_col=cfg.sample_name_col_for_plotting,
                ref_col=reference_column,
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
                ref_col=reference_column,
                figsize_per_cell=(4.0, 3.0),
                dpi=160,
                cmaps=cfg.correlation_matrix_cmaps,
                vmin=None,
                vmax=None,
                output_dir=pp_corr_dir,
                output_key= "positionwise_result"
            )

    ####### Save basic analysis adata - post preprocessing and duplicate removal ################
    from ..readwrite import safe_write_h5ad
    if not spatial_adata_path.exists() or cfg.force_redo_preprocessing:
        print('Saving spatial analyzed adata post preprocessing and duplicate removal')
        if ".gz" == spatial_adata_path.suffix:
            print(f"Spatial adata path: {spatial_adata_path}")
            safe_write_h5ad(adata, spatial_adata_path, compression='gzip', backup=True, backup_dir=spatial_backup_dir)
        else:
            spatial_adata_path = spatial_adata_path.with_name(spatial_adata_path.name + '.gz')
            print(f"Spatial adata path: {spatial_adata_path}")
            safe_write_h5ad(adata, spatial_adata_path, compression='gzip', backup=True, backup_dir=spatial_backup_dir)
    ############################################### smftools spatial end ###############################################

    return adata, spatial_adata_path