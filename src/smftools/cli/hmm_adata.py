from typing import Tuple
from pathlib import Path

def hmm_adata(config_path: str):
    """
    CLI-facing wrapper for HMM analysis.

    Command line entrypoint:
        smftools hmm <config_path>

    Responsibilities:
    - Build cfg via load_adata()
    - Ensure preprocess + spatial stages are run
    - Decide which AnnData to start from (hmm > spatial > pp_dedup > pp > raw)
    - Call hmm_adata_core(cfg, adata, paths)
    """
    from ..readwrite import safe_read_h5ad
    from .load_adata import load_adata
    from .preprocess_adata import preprocess_adata
    from .spatial_adata import spatial_adata
    from .helpers import get_adata_paths

    # 1) load cfg / stage paths
    _, _, cfg = load_adata(config_path)
    paths = get_adata_paths(cfg)

    # 2) make sure upstream stages are run (they have their own skipping logic)
    preprocess_adata(config_path)
    spatial_ad, _ = spatial_adata(config_path)

    # 3) choose starting AnnData
    # Prefer:
    #   - existing HMM h5ad if not forcing redo
    #   - in-memory spatial_ad from wrapper call
    #   - saved spatial / pp_dedup / pp / raw on disk
    if paths.hmm.exists() and not (cfg.force_redo_hmm_fit or cfg.force_redo_hmm_apply):
        adata, _ = safe_read_h5ad(paths.hmm)
        return adata, paths.hmm

    if spatial_ad is not None:
        adata = spatial_ad
    elif paths.spatial.exists():
        adata, _ = safe_read_h5ad(paths.spatial)
    elif paths.pp_dedup.exists():
        adata, _ = safe_read_h5ad(paths.pp_dedup)
    elif paths.pp.exists():
        adata, _ = safe_read_h5ad(paths.pp)
    elif paths.raw.exists():
        adata, _ = safe_read_h5ad(paths.raw)
    else:
        raise FileNotFoundError(
            "No AnnData available for HMM: expected at least raw or preprocessed h5ad."
        )

    # 4) delegate to core
    adata, hmm_adata_path = hmm_adata_core(cfg, adata, paths)
    return adata, hmm_adata_path

def hmm_adata_core(cfg, adata, paths) -> Tuple["anndata.AnnData", Path]:
    """
    Core HMM analysis pipeline.

    Assumes:
    - cfg is an ExperimentConfig
    - adata is the starting AnnData (typically spatial + dedup)
    - paths is an AdataPaths object (with .raw/.pp/.pp_dedup/.spatial/.hmm)

    Does NOT decide which h5ad to start from – that is the wrapper's job.
    """

    import os
    import warnings

    import numpy as np

    from scipy.sparse import issparse

    from ..readwrite import safe_write_h5ad, make_dirs, add_or_update_column_in_csv
    from .helpers import write_gz_h5ad
    from ..hmm.HMM import HMM
    from ..hmm import call_hmm_peaks
    from ..plotting import combined_hmm_raw_clustermap, plot_hmm_layers_rolling_by_sample_ref, plot_hmm_size_contours

    smf_modality = cfg.smf_modality
    deaminase = smf_modality == "deaminase"

    output_directory = Path(cfg.output_directory)
    make_dirs([output_directory])

    pp_dir = output_directory / "preprocessed" / "deduplicated"
############################################### HMM based feature annotations ###############################################
    if not (cfg.bypass_hmm_fit and cfg.bypass_hmm_apply):

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
    if not paths.hmm.exists():
        print("Saving spatial analyzed AnnData (post preprocessing and duplicate removal).")
        write_gz_h5ad(adata, paths.hmm)

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

    return (adata, paths.hmm)