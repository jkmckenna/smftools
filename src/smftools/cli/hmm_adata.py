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
    initial_adata, initial_adata_path, bam_files, cfg = load_adata(config_path)
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
    skip_spatial = True
    if skip_spatial:
        pp_dedup_spatial_adata = None
        spatial_adata_basename = pp_dup_rem_adata_path.name.split(".")[0] + '_spatial.h5ad.gz'
        pp_dedup_spatial_adata_path = pp_dup_rem_adata_path.parent / spatial_adata_basename
    else:
        pp_dedup_spatial_adata, pp_dedup_spatial_adata_path = spatial_adata(config_path)
    ############################################### smftools spatial end ###############################################

    ############################################### smftools hmm start ###############################################
    # hmm adata
    hmm_adata_basename = pp_dedup_spatial_adata_path.with_suffix("").name + '_hmm.h5ad.gz'
    hmm_adata_path = pp_dedup_spatial_adata_path.parent / hmm_adata_basename

    if pp_dedup_spatial_adata:
        # This happens on first run of the pipeline
        adata = pp_dedup_spatial_adata
    else:
        # If an anndata is saved, check which stages of the anndata are available
        initial_version_available = initial_adata_path.exists()
        preprocessed_version_available = pp_adata_path.exists()
        preprocessed_dup_removed_version_available = pp_dup_rem_adata_path.exists()
        preprocessed_dedup_spatial_version_available = pp_dedup_spatial_adata_path.exists()
        preprocessed_dedup_spatial_hmm_version_available = hmm_adata_path.exists()

        if cfg.force_redo_hmm_fit:
            print(f"Forcing redo of basic analysis workflow, starting from the preprocessed adata if available. Otherwise, will use the raw adata.")
            if preprocessed_dedup_spatial_version_available:
                adata, load_report = safe_read_h5ad(pp_dedup_spatial_adata_path)
            elif preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            else:
                print(f"Can not redo duplicate detection when there is no compatible adata available: either raw or preprocessed are required")
        elif preprocessed_dedup_spatial_hmm_version_available:
            return (None, hmm_adata_path)
        else:
            if preprocessed_dedup_spatial_version_available:
                adata, load_report = safe_read_h5ad(pp_dedup_spatial_adata_path)
            elif preprocessed_dup_removed_version_available:
                adata, load_report = safe_read_h5ad(pp_dup_rem_adata_path)
            elif initial_version_available:
                adata, load_report = safe_read_h5ad(initial_adata_path)
            else:            
                print(f"No adata available.")
                return
    references = adata.obs[cfg.reference_column].cat.categories          
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

        for sample in samples:
            for ref in references:
                mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & (adata.obs[cfg.reference_column] == ref)
                subset = adata[mask].copy()
                if subset.shape[0] < 1:
                    continue

                for mod_site in cfg.hmm_methbases:
                    mod_label = {'C': 'any_C'}.get(mod_site, mod_site)
                    hmm_path = os.path.join(hmm_dir, f"{sample}_{ref}_{mod_label}_hmm_model.pth")

                    # ensure the input obsm exists
                    obsm_key = f'{ref}_{mod_label}_site'
                    if obsm_key not in subset.obsm:
                        print(f"Skipping {sample} {ref} {mod_label}: missing obsm '{obsm_key}'")
                        continue

                    # Fit or load model
                    if os.path.exists(hmm_path) and not cfg.force_redo_hmm_fit:
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
                    if (not cfg.bypass_hmm_apply) or cfg.force_redo_hmm_apply:
                        print(f"Applying HMM on subset for {sample} {ref} {mod_label}")
                        # Use the new uns_key argument so subset will record appended layer names
                        # (annotate_adata modifies subset.obs/layers in-place and should write subset.uns[uns_key])
                        hmm.annotate_adata(subset,
                                        obs_column=cfg.reference_column,
                                        layer=cfg.layer_for_umap_plotting,
                                        config=cfg)
                        
                        #to_merge = [("C_all_accessible_features", 80)]
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
    
    hmm_dir = pp_dir / "11_hmm_clustermaps"

    if hmm_dir.is_dir():
        print(f'{hmm_dir} already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import combined_hmm_raw_clustermap

        for layer in ['C_all_accessible_features', 'C_small_bound_stretch', 'C_medium_bound_stretch', 'C_putative_nucleosome', 'C_all_accessible_features_merged']:
            save_path = hmm_dir / layer
            make_dirs([save_path])

            combined_hmm_raw_clustermap(
            adata,
            sample_col=cfg.sample_name_col_for_plotting,
            reference_col=cfg.reference_column,
            hmm_feature_layer=layer,
            layer_gpc="nan0_0minus1",
            layer_cpg="nan0_0minus1",
            layer_any_c="nan0_0minus1",
            cmap_hmm="coolwarm",
            cmap_gpc="coolwarm",
            cmap_cpg="viridis",
            cmap_any_c='coolwarm',
            min_quality=20,
            min_length=80,
            min_mapped_length_to_reference_length_ratio=0.2,
            min_position_valid_fraction=0.2,
            sample_mapping=None,
            save_path=save_path,
            normalize_hmm=False,
            sort_by="gpc",  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
            bins=None,
            deaminase=True,
            min_signal=0
            )

    hmm_dir = pp_dir / "12_hmm_bulk_traces"

    if hmm_dir.is_dir():
        print(f'{hmm_dir} already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_layers_rolling_by_sample_ref
        saved = plot_hmm_layers_rolling_by_sample_ref(
            adata,
            layers=adata.uns['hmm_appended_layers'],
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            window=101,
            rows_per_page=4,
            figsize_per_cell=(4,2.5),
            output_dir=hmm_dir,
            save=True,
            show_raw=False
        )

    hmm_dir = pp_dir / "13_hmm_fragment_distributions"

    if hmm_dir.is_dir():
        print(f'{hmm_dir} already exists.')
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_size_contours

        for layer, max in [('C_all_accessible_features_lengths', 400), ('C_all_footprint_features_lengths', 160), ('C_all_accessible_features_merged_lengths', 800)]:
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
                smoothing_sigma=None,
                normalize_after_smoothing=False,
                cmap='viridis', 
                log_scale_z=True
            )
    ########################################################################################################################

    return (adata, hmm_adata_path)