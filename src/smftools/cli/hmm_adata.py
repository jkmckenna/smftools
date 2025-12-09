def hmm_adata(config_path):
    """
    High-level function to call for hmm analysis of an adata object. 
    Command line accesses this through smftools hmm <config_path>

    Parameters:
        config_path (str): A string representing the file path to the experiment configuration csv file.

    Returns:
        (pp_dedup_spatial_hmm_adata, pp_dedup_spatial_hmm_adata_path)
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad, make_dirs, add_or_update_column_in_csv
    from .load_adata import load_adata
    from .preprocess_adata import preprocess_adata
    from .spatial_adata import spatial_adata

    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc

    import os
    from importlib import resources
    from pathlib import Path

    from datetime import datetime
    date_str = datetime.today().strftime("%y%m%d")

    ############################################### smftools load start ###############################################
    adata, adata_path, cfg = load_adata(config_path)
    # General config variable init - Necessary user passed inputs
    smf_modality = cfg.smf_modality # needed for specifying if the data is conversion SMF or direct methylation detection SMF. Or deaminase smf Necessary.
    output_directory = Path(cfg.output_directory)  # Path to the output directory to make for the analysis. Necessary.

    # Make initial output directory
    make_dirs([output_directory])
    ############################################### smftools load end ###############################################

    ############################################### smftools preprocess start ###############################################
    pp_adata, pp_adata_path, pp_dedup_adata, pp_dup_rem_adata_path = preprocess_adata(config_path)
    ############################################### smftools preprocess end ###############################################

    ############################################### smftools spatial start ###############################################
    spatial_ad, spatial_adata_path = spatial_adata(config_path)
    ############################################### smftools spatial end ###############################################

    ############################################### smftools hmm start ###############################################
    input_manager_df = pd.read_csv(cfg.summary_file)
    initial_adata_path = Path(input_manager_df['load_adata'][0])
    pp_adata_path = Path(input_manager_df['pp_adata'][0])
    pp_dup_rem_adata_path = Path(input_manager_df['pp_dedup_adata'][0])
    spatial_adata_path = Path(input_manager_df['spatial_adata'][0])
    hmm_adata_path = Path(input_manager_df['hmm_adata'][0])
    
    if spatial_ad:
        # This happens on first run of the pipeline
        adata = spatial_ad
    else:
        # If an anndata is saved, check which stages of the anndata are available
        initial_version_available = initial_adata_path.exists()
        preprocessed_version_available = pp_adata_path.exists()
        preprocessed_dup_removed_version_available = pp_dup_rem_adata_path.exists()
        preprocessed_dedup_spatial_version_available = spatial_adata_path.exists()
        preprocessed_dedup_spatial_hmm_version_available = hmm_adata_path.exists()

        if cfg.force_redo_hmm_fit or cfg.force_redo_hmm_apply:
            print(f"Forcing redo of hmm analysis workflow.")
            if preprocessed_dedup_spatial_hmm_version_available:
                adata, load_report = safe_read_h5ad(hmm_adata_path)
            elif preprocessed_dedup_spatial_version_available:
                adata, load_report = safe_read_h5ad(spatial_adata_path)
            elif preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
        elif preprocessed_dedup_spatial_hmm_version_available:
            adata, load_report = safe_read_h5ad(hmm_adata_path)
        else:
            if preprocessed_dedup_spatial_version_available:
                adata, load_report = safe_read_h5ad(spatial_adata_path)
            elif preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            else:            
                print(f"No adata available.")
                return
    references = adata.obs[cfg.reference_column].cat.categories  
    deaminase = smf_modality == 'deaminase'        
############################################### HMM based feature annotations ###############################################
    if not (cfg.bypass_hmm_fit and cfg.bypass_hmm_apply):
        from ..hmm.HMM import HMM
        from scipy.sparse import issparse, csr_matrix
        import warnings

        pp_dir = output_directory / "preprocessed"
        pp_dir = pp_dir / "deduplicated"
        hmm_dir = pp_dir / "10_hmm_models"

        if hmm_dir.is_dir():
            print(f'{hmm_dir} already exists.')
        else:
            make_dirs([pp_dir, hmm_dir])

        samples = adata.obs[cfg.sample_name_col_for_plotting].cat.categories
        references = adata.obs[cfg.reference_column].cat.categories
        uns_key = "hmm_appended_layers"

        # ensure uns key exists (avoid KeyError later)
        if adata.uns.get(uns_key) is None:
            adata.uns[uns_key] = []

        if adata.uns.get('hmm_annotated', False) and not cfg.force_redo_hmm_fit and not cfg.force_redo_hmm_apply:
            pass
        else:
            for sample in samples:
                for ref in references:
                    mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & (adata.obs[cfg.reference_column] == ref)
                    subset = adata[mask].copy()
                    if subset.shape[0] < 1:
                        continue

                    for mod_site in cfg.hmm_methbases:
                        mod_label = {'C': 'C'}.get(mod_site, mod_site)
                        hmm_path = hmm_dir / f"{sample}_{ref}_{mod_label}_hmm_model.pth"

                        # ensure the input obsm exists
                        obsm_key = f'{ref}_{mod_label}_site'
                        if obsm_key not in subset.obsm:
                            print(f"Skipping {sample} {ref} {mod_label}: missing obsm '{obsm_key}'")
                            continue

                        # Fit or load model
                        if hmm_path.exists() and not cfg.force_redo_hmm_fit:
                            hmm = HMM.load(hmm_path)
                            hmm.print_params()
                        else:
                            print(f"Fitting HMM for {sample} {ref} {mod_label}")
                            hmm = HMM.from_config(cfg)
                            # fit expects a list-of-seqs or 2D ndarray in the obsm
                            seqs = subset.obsm[obsm_key]
                            hmm.fit(seqs)
                            hmm.print_params()
                            hmm.save(hmm_path)

                        # Apply / annotate on the subset, then copy layers back to final_adata
                        if cfg.bypass_hmm_apply:
                            pass
                        else:
                            print(f"Applying HMM on subset for {sample} {ref} {mod_label}")
                            # Use the new uns_key argument so subset will record appended layer names
                            # (annotate_adata modifies subset.obs/layers in-place and should write subset.uns[uns_key])
                            if smf_modality == "direct":
                                hmm_layer = cfg.output_binary_layer_name
                            else:
                                hmm_layer = None

                            hmm.annotate_adata(subset,
                                            obs_column=cfg.reference_column,
                                            layer=hmm_layer,
                                            config=cfg,
                                            force_redo=cfg.force_redo_hmm_apply
                                            )
                            
                            if adata.uns.get('hmm_annotated', False) and not cfg.force_redo_hmm_apply:
                                pass
                            else:
                                to_merge = cfg.hmm_merge_layer_features
                                for layer_to_merge, merge_distance in to_merge:
                                    if layer_to_merge:
                                        hmm.merge_intervals_in_layer(subset,
                                                                    layer=layer_to_merge,
                                                                    distance_threshold=merge_distance,
                                                                    overwrite=True
                                                                    )
                                    else:
                                        pass

                                # collect appended layers from subset.uns
                                appended = list(subset.uns.get(uns_key, []))
                                print(appended)
                                if len(appended) == 0:
                                    # nothing appended for this subset; continue
                                    continue

                                # copy each appended layer into adata
                                subset_mask_bool = mask.values if hasattr(mask, "values") else np.asarray(mask)
                                for layer_name in appended:
                                    if layer_name not in subset.layers:
                                        # defensive: skip
                                        warnings.warn(f"Expected layer {layer_name} in subset but not found; skipping copy.")
                                        continue
                                    sub_layer = subset.layers[layer_name]
                                    # ensure final layer exists and assign rows
                                    try:
                                        hmm._ensure_final_layer_and_assign(adata, layer_name, subset_mask_bool, sub_layer)
                                    except Exception as e:
                                        warnings.warn(f"Failed to copy layer {layer_name} into adata: {e}", stacklevel=2)
                                        # fallback: if dense and small, try to coerce
                                        if issparse(sub_layer):
                                            arr = sub_layer.toarray()
                                        else:
                                            arr = np.asarray(sub_layer)
                                        adata.layers[layer_name] = adata.layers.get(layer_name, np.zeros((adata.shape[0], arr.shape[1]), dtype=arr.dtype))
                                        final_idx = np.nonzero(subset_mask_bool)[0]
                                        adata.layers[layer_name][final_idx, :] = arr

                                # merge appended layer names into adata.uns
                                existing = list(adata.uns.get(uns_key, []))
                                for ln in appended:
                                    if ln not in existing:
                                        existing.append(ln)
                                adata.uns[uns_key] = existing

    else:
        pass

    from ..hmm import call_hmm_peaks
    hmm_dir = pp_dir / "11_hmm_peak_calling"
    if hmm_dir.is_dir():
        pass
    else:
        make_dirs([pp_dir, hmm_dir])

        call_hmm_peaks(
                adata,
                feature_configs=cfg.hmm_peak_feature_configs,
                ref_column=cfg.reference_column,
                site_types=cfg.mod_target_bases,
                save_plot=True,
                output_dir=hmm_dir,
                index_col_suffix=cfg.reindexed_var_suffix)
    
    ## Save HMM annotated adata
    if not hmm_adata_path.exists():
        print('Saving hmm analyzed adata post preprocessing and duplicate removal')
        if ".gz" == hmm_adata_path.suffix:
            safe_write_h5ad(adata, hmm_adata_path, compression='gzip', backup=True)
        else:
            hmm_adata_path = hmm_adata_path.with_name(hmm_adata_path.name + '.gz')
            safe_write_h5ad(adata, hmm_adata_path, compression='gzip', backup=True)

    add_or_update_column_in_csv(cfg.summary_file, "hmm_adata", hmm_adata_path)

    ########################################################################################################################

############################################### HMM based feature plotting ###############################################
    from ..plotting import combined_hmm_raw_clustermap
    hmm_dir = pp_dir / "12_hmm_clustermaps"
    make_dirs([pp_dir, hmm_dir])

    layers: list[str] = []

    for base in cfg.hmm_methbases:
        layers.extend([f"{base}_{layer}" for layer in cfg.hmm_clustermap_feature_layers])

    if cfg.cpg:
        layers.extend(["CpG_cpg_patch"])

    if not layers:
        raise ValueError(
            f"No HMM feature layers matched mod_target_bases={cfg.mod_target_bases} "
            f"and smf_modality={smf_modality}"
        )

    for layer in layers:
        hmm_cluster_save_dir = hmm_dir / layer
        if hmm_cluster_save_dir.is_dir():
            pass
        else:
            make_dirs([hmm_cluster_save_dir])

            combined_hmm_raw_clustermap(
            adata,
            sample_col=cfg.sample_name_col_for_plotting,
            reference_col=cfg.reference_column,
            hmm_feature_layer=layer,
            layer_gpc=cfg.layer_for_clustermap_plotting,
            layer_cpg=cfg.layer_for_clustermap_plotting,
            layer_c=cfg.layer_for_clustermap_plotting,
            layer_a=cfg.layer_for_clustermap_plotting,
            cmap_hmm=cfg.clustermap_cmap_hmm,
            cmap_gpc=cfg.clustermap_cmap_gpc,
            cmap_cpg=cfg.clustermap_cmap_cpg,
            cmap_c=cfg.clustermap_cmap_c,
            cmap_a=cfg.clustermap_cmap_a,
            min_quality=cfg.read_quality_filter_thresholds[0],
            min_length=cfg.read_len_filter_thresholds[0],
            min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[0],
            min_position_valid_fraction=1-cfg.position_max_nan_threshold,
            save_path=hmm_cluster_save_dir,
            normalize_hmm=False,
            sort_by=cfg.hmm_clustermap_sortby,  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
            bins=None,
            deaminase=deaminase,
            min_signal=0,
            index_col_suffix=cfg.reindexed_var_suffix
            )

    hmm_dir = pp_dir / "13_hmm_bulk_traces"

    if hmm_dir.is_dir():
        print(f'{hmm_dir} already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_layers_rolling_by_sample_ref
        bulk_hmm_layers = [layer for layer in adata.uns['hmm_appended_layers'] if "_lengths" not in layer]
        saved = plot_hmm_layers_rolling_by_sample_ref(
            adata,
            layers=bulk_hmm_layers,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            window=101,
            rows_per_page=4,
            figsize_per_cell=(4,2.5),
            output_dir=hmm_dir,
            save=True,
            show_raw=False
        )

    hmm_dir = pp_dir / "14_hmm_fragment_distributions"

    if hmm_dir.is_dir():
        print(f'{hmm_dir} already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_size_contours

        if smf_modality == 'deaminase':
            fragments = [('C_all_accessible_features_lengths', 400), ('C_all_footprint_features_lengths', 250), ('C_all_accessible_features_merged_lengths', 800)]
        elif smf_modality == 'conversion':
            fragments = [('GpC_all_accessible_features_lengths', 400), ('GpC_all_footprint_features_lengths', 250), ('GpC_all_accessible_features_merged_lengths', 800)]
        elif smf_modality == "direct":
            fragments = [('A_all_accessible_features_lengths', 400), ('A_all_footprint_features_lengths', 200), ('A_all_accessible_features_merged_lengths', 800)]

        for layer, max in fragments:
            save_path = hmm_dir / layer
            make_dirs([save_path])

            figs = plot_hmm_size_contours(
                adata,
                length_layer=layer,
                sample_col=cfg.sample_name_col_for_plotting,
                ref_obs_col=cfg.reference_column,
                rows_per_page=6,
                max_length_cap=max,
                figsize_per_cell=(3.5, 2.2),
                save_path=save_path,
                save_pdf=False,
                save_each_page=True,
                dpi=200,
                smoothing_sigma=(10, 10),
                normalize_after_smoothing=True,
                cmap='Greens', 
                log_scale_z=True
            )
    ########################################################################################################################

    return (adata, hmm_adata_path)