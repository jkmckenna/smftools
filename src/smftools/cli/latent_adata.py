from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import anndata as ad

from smftools.constants import LATENT_DIR, LOGGING_DIR, SEQUENCE_INTEGER_ENCODING, REFERENCE_STRAND
from smftools.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _build_mod_sites_var_filter_mask(
    adata: ad.AnnData,
    references: Sequence[str],
    cfg,
    smf_modality: str,
    deaminase: bool,
) -> "np.ndarray":
    """Build a boolean var mask for mod sites across references."""
    import numpy as np

    ref_masks = []
    for ref in references:
        if smf_modality == "direct":
            mod_site_cols = [f"{ref}_{base}_site" for base in cfg.mod_target_bases]
        elif deaminase:
            mod_site_cols = [f"{ref}_C_site"]
        else:
            mod_site_cols = [f"{ref}_{base}_site" for base in cfg.mod_target_bases]

        position_col = f"position_in_{ref}"
        required_cols = mod_site_cols + [position_col]
        missing = [col for col in required_cols if col not in adata.var.columns]
        if missing:
            raise KeyError(f"var_filters not found in adata.var: {missing}")

        mod_masks = [np.asarray(adata.var[col].values, dtype=bool) for col in mod_site_cols]
        mod_mask = mod_masks[0] if len(mod_masks) == 1 else np.logical_or.reduce(mod_masks)
        position_mask = np.asarray(adata.var[position_col].values, dtype=bool)
        ref_masks.append(np.logical_and(mod_mask, position_mask))

    if not ref_masks:
        return np.ones(adata.n_vars, dtype=bool)

    return np.logical_and.reduce(ref_masks)


def _build_reference_position_mask(
    adata: ad.AnnData,
    references: Sequence[str],
) -> "np.ndarray":
    """Build a boolean var mask for positions valid across references."""
    import numpy as np

    ref_masks = []
    for ref in references:
        position_col = f"position_in_{ref}"
        if position_col not in adata.var.columns:
            raise KeyError(f"var_filters not found in adata.var: {position_col}")
        position_mask = np.asarray(adata.var[position_col].values, dtype=bool)
        ref_masks.append(position_mask)

    if not ref_masks:
        return np.ones(adata.n_vars, dtype=bool)

    return np.logical_and.reduce(ref_masks)


def latent_adata(
    config_path: str,
) -> Tuple[Optional[ad.AnnData], Optional[Path]]:
    """
    CLI-facing wrapper for representation learning.

    Called by: `smftools latent <config_path>`

    Responsibilities:
    - Determine which AnnData stages exist (pp, pp_dedup, spatial, hmm).
    - Call `latent_adata_core(...)` when actual work is needed.

    Returns
    -------
    latent_adata : AnnData | None
        AnnData with latent analyses, or None if we skipped because a later-stage
        AnnData already exists.
    latent_adata_path : Path | None
        Path to the “current” latent AnnData.
    """
    from ..readwrite import add_or_update_column_in_csv, safe_read_h5ad
    from .helpers import get_adata_paths, load_experiment_config

    # 1) Ensure config + basic paths via load_adata
    cfg = load_experiment_config(config_path)

    paths = get_adata_paths(cfg)

    pp_path = paths.pp
    pp_dedup_path = paths.pp_dedup
    spatial_path = paths.spatial
    hmm_path = paths.hmm
    latent_path = paths.latent

    # Stage-skipping logic for latent
    if not getattr(cfg, "force_redo_latent_analyses", False):
        # If latent exists, we consider latent analyses already done.
        if latent_path.exists():
            logger.info(f"Latent AnnData found: {latent_path}\nSkipping smftools latent")
            return None, latent_path

    # Helper to load from disk, reusing loaded_adata if it matches
    def _load(path: Path):
        adata, _ = safe_read_h5ad(path)
        return adata

    # 3) Decide which AnnData to use as the *starting point* for latent analyses
    if latent_path.exists():
        start_adata = _load(latent_path)
        source_path = latent_path
    elif hmm_path.exists():
        start_adata = _load(hmm_path)
        source_path = hmm_path
    elif spatial_path.exists():
        start_adata = _load(spatial_path)
        source_path = spatial_path
    elif pp_dedup_path.exists():
        start_adata = _load(pp_dedup_path)
        source_path = pp_dedup_path
    elif pp_path.exists():
        start_adata = _load(pp_path)
        source_path = pp_path
    else:
        logger.warning(
            "No suitable AnnData found for latent analyses (need at least preprocessed)."
        )
        return None, None

    # 4) Run the latent core
    adata_latent, latent_path = latent_adata_core(
        adata=start_adata,
        cfg=cfg,
        paths=paths,
        source_adata_path=source_path,
        config_path=config_path,
    )

    return adata_latent, latent_path


def latent_adata_core(
    adata: ad.AnnData,
    cfg,
    paths: AdataPaths,
    source_adata_path: Optional[Path] = None,
    config_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Path]:
    """
    Core spatial analysis pipeline.

    Assumes:
    - `adata` is (typically) the preprocessed, duplicate-removed AnnData.
    - `cfg` is the ExperimentConfig.

    Does:
    - Optional sample sheet load.
    - Optional inversion & reindexing.
    - PCA/KNN/UMAP/Leiden/NMP/PARAFAC 
    - Save latent AnnData to `latent_adata_path`.

    Returns
    -------
    adata : AnnData
        analyzed AnnData (same object, modified in-place).
    adata_path : Path
        Path where AnnData was written.
    """
    import os
    import warnings
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from ..metadata import record_smftools_metadata
    from ..plotting import (
        plot_cp_sequence_components,
        plot_embedding_grid,
        plot_nmf_components,
        plot_pca_components,
        plot_pca_explained_variance,
        plot_pca_grid,
        plot_umap_grid,
    )
    from ..preprocessing import (
        invert_adata,
        load_sample_sheet,
        reindex_references_adata,
    )
    from ..readwrite import make_dirs, safe_read_h5ad
    from ..tools import (
        calculate_leiden,
        calculate_nmf,
        calculate_sequence_cp_decomposition,
        calculate_umap,
        calculate_pca,
        calculate_knn,
    )
    from .helpers import write_gz_h5ad

    # -----------------------------
    # General setup
    # -----------------------------
    date_str = datetime.today().strftime("%y%m%d")
    now = datetime.now()
    time_str = now.strftime("%H%M%S")
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    latent_adata_path = paths.latent

    output_directory = Path(cfg.output_directory)
    latent_directory = output_directory / LATENT_DIR
    logging_directory = latent_directory / LOGGING_DIR

    make_dirs([output_directory, latent_directory])

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

    latent_dir_dedup = latent_directory / "deduplicated"
    
    # ============================================================
    # 2) PCA/UMAP/NMF at valid modified base site binary encodings shared across references
    # ============================================================
    SUBSET = "shared_valid_mod_sites_binary_mod_arrays"

    pca_dir = latent_dir_dedup / f"01_pca_{SUBSET}"
    umap_dir = latent_dir_dedup / f"01_umap_{SUBSET}"
    nmf_dir = latent_dir_dedup / f"01_nmf_{SUBSET}"

    mod_site_layers = []
    for mod_base in cfg.mod_target_bases:
        mod_site_layers += [f"Modified_{mod_base}_site_count", f"Fraction_{mod_base}_site_modified"]

    plotting_layers = [cfg.sample_name_col_for_plotting, REFERENCE_STRAND] + mod_site_layers
    plotting_layers += cfg.umap_layers_to_plot

    mod_sites_mask = _build_mod_sites_var_filter_mask(
        adata=adata,
        references=references,
        cfg=cfg,
        smf_modality=smf_modality,
        deaminase=deaminase,
    )

    # PCA calculation
    adata = calculate_pca(
        adata,
        layer=cfg.layer_for_umap_plotting,
        var_mask=mod_sites_mask,
        n_pcs=10,
        output_suffix=SUBSET,
    )

    # KNN calculation
    adata = calculate_knn(
        adata,
        obsm=f"X_pca_{SUBSET}",
        knn_neighbors=15,
    )

    # UMAP Calculation
    adata = calculate_umap(
        adata,
        obsm=f"X_pca_{SUBSET}",
        output_suffix=SUBSET,
    )

    # Leiden clustering
    calculate_leiden(adata, resolution=0.1, connectivities_key=f"connectivities_X_pca_{SUBSET}")

    # NMF Calculation
    adata = calculate_nmf(
        adata,
        layer=cfg.layer_for_umap_plotting,
        var_mask=mod_sites_mask,
        n_components=2,
        suffix=SUBSET,
    )

    # PCA
    if pca_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{pca_dir} already exists. Skipping PCA calculation and plotting.")
    else:
        make_dirs([pca_dir])
        plot_pca_grid(adata, subset=SUBSET, color=plotting_layers, output_dir=pca_dir)
        plot_pca_explained_variance(adata, subset=SUBSET, output_dir=pca_dir)
        plot_pca_components(adata, output_dir=pca_dir, suffix=SUBSET)

    # UMAP
    if umap_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{umap_dir} already exists. Skipping UMAP plotting.")
    else:
        make_dirs([umap_dir])
        plot_umap_grid(adata, subset=SUBSET, color=plotting_layers, output_dir=umap_dir)

    # NMF
    if nmf_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{nmf_dir} already exists. Skipping NMF plotting.")
    else:
        make_dirs([nmf_dir])

        plot_embedding_grid(adata, basis=f"nmf_{SUBSET}", color=plotting_layers, output_dir=nmf_dir)
        plot_nmf_components(adata, output_dir=nmf_dir)

    # ============================================================
    # 3) PCA/UMAP/NMF at valid base site integer encodings shared across references
    # ============================================================
    SUBSET = "shared_valid_ref_sites_integer_sequence_encodings"

    pca_dir = latent_dir_dedup / f"01_pca_{SUBSET}"
    umap_dir = latent_dir_dedup / f"01_umap_{SUBSET}"
    nmf_dir = latent_dir_dedup / f"01_nmf_{SUBSET}"

    valid_sites = _build_reference_position_mask(adata, references)

    # PCA calculation
    adata = calculate_pca(
        adata,
        layer=SEQUENCE_INTEGER_ENCODING,
        var_mask=valid_sites,
        n_pcs=10,
        output_suffix=SUBSET,
    )

    # KNN calculation
    adata = calculate_knn(
        adata,
        obsm=f"X_pca_{SUBSET}",
        knn_neighbors=15,
    )

    # UMAP Calculation
    adata = calculate_umap(
        adata,
        obsm=f"X_pca_{SUBSET}",
        output_suffix=SUBSET,
    )

    # Leiden clustering
    calculate_leiden(adata, resolution=0.1, connectivities_key=f"connectivities_X_pca_{SUBSET}")

    # NMF Calculation
    adata = calculate_nmf(
        adata,
        layer=SEQUENCE_INTEGER_ENCODING,
        var_mask=valid_sites,
        n_components=2,
        suffix=SUBSET,
    )

    # PCA
    if pca_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{pca_dir} already exists. Skipping PCA calculation and plotting.")
    else:
        make_dirs([pca_dir])
        plot_pca_grid(adata, subset=SUBSET, color=plotting_layers, output_dir=pca_dir)
        plot_pca_explained_variance(adata, subset=SUBSET, output_dir=pca_dir)
        plot_pca_components(adata, output_dir=pca_dir, suffix=SUBSET)

        # # Component plotting
        # title = "Component Loadings"
        # save_path = pca_dir / (title + ".png")
        # for i in range(10):
        #     pc = adata.varm["PCs"][:, i] 
        #     plt.scatter(adata.var_names, pc, label=f"PC{i+1}")
        # plt.savefig(save_path)

    # UMAP
    if umap_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{umap_dir} already exists. Skipping UMAP plotting.")
    else:
        make_dirs([umap_dir])
        plot_umap_grid(adata, subset=SUBSET, color=plotting_layers, output_dir=umap_dir)

    # NMF
    if nmf_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{nmf_dir} already exists. Skipping NMF plotting.")
    else:
        make_dirs([nmf_dir])

        plot_embedding_grid(adata, basis=f"nmf_{SUBSET}", color=plotting_layers, output_dir=nmf_dir)
        plot_nmf_components(adata, output_dir=nmf_dir)

    # ============================================================
    # 3) CP PARAFAC factorization of shared mod site OHE sequences with mask layer
    # ============================================================
    SUBSET = "shared_valid_mod_sites_ohe_sequence_N_masked"

    cp_sequence_dir = latent_dir_dedup / f"02_cp_{SUBSET}"

    # Calculate CP tensor factorization
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning(
            "Layer %s not found; skipping sequence integer encoding CP.",
            SEQUENCE_INTEGER_ENCODING,
        )
    else:
        adata = calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=mod_sites_mask,
            var_mask_name="shared_reference_and_mod_site_positions",
            rank=2,
            embedding_key=f"X_cp_{SUBSET}",
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
            non_negative=False,
        )

    # CP decomposition using sequence integer encoding (no var filters)
    if cp_sequence_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{cp_sequence_dir} already exists. Skipping sequence CP plotting.")
    else:
        make_dirs([cp_sequence_dir])
        plot_embedding_grid(
            adata,
            basis=f"cp_{SUBSET}",
            color=plotting_layers,
            output_dir=cp_sequence_dir,
        )
        plot_cp_sequence_components(
            adata,
            output_dir=cp_sequence_dir,
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
        )

    # ============================================================
    # 4) Non-negative CP PARAFAC factorization of shared mod site OHE sequences with mask layer
    # ============================================================
    SUBSET = "shared_valid_mod_sites_ohe_sequence_N_masked_non_negative"

    cp_sequence_dir = latent_dir_dedup / f"02_cp_{SUBSET}"

    # Calculate CP tensor factorization
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning(
            "Layer %s not found; skipping sequence integer encoding CP.",
            SEQUENCE_INTEGER_ENCODING,
        )
    else:
        adata = calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=mod_sites_mask,
            var_mask_name="shared_reference_mod_site_positions",
            rank=2,
            embedding_key=f"X_cp_{SUBSET}",
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
            non_negative=True,
        )

    # CP decomposition using sequence integer encoding (no var filters)
    if cp_sequence_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{cp_sequence_dir} already exists. Skipping sequence CP plotting.")
    else:
        make_dirs([cp_sequence_dir])
        plot_embedding_grid(
            adata,
            basis=f"cp_{SUBSET}",
            color=plotting_layers,
            output_dir=cp_sequence_dir,
        )
        plot_cp_sequence_components(
            adata,
            output_dir=cp_sequence_dir,
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
        )
    # ============================================================
    # 6) CP PARAFAC factorization of non mod-site OHE sequences with mask layer
    # ============================================================
    SUBSET = "non_mod_site_ohe_sequence_N_masked"

    cp_sequence_dir = latent_dir_dedup / f"03_cp_{SUBSET}"

    # Calculate CP tensor factorization
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning(
            "Layer %s not found; skipping sequence integer encoding CP.",
            SEQUENCE_INTEGER_ENCODING,
        )
    else:
        adata = calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=~mod_sites_mask,
            var_mask_name="non_mod_site_reference_positions",
            rank=2,
            embedding_key=f"X_cp_{SUBSET}",
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
            non_negative=False,
        )

    # CP decomposition using sequence integer encoding (no var filters)
    if cp_sequence_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{cp_sequence_dir} already exists. Skipping sequence CP plotting.")
    else:
        make_dirs([cp_sequence_dir])
        plot_embedding_grid(
            adata,
            basis=f"cp_{SUBSET}",
            color=plotting_layers,
            output_dir=cp_sequence_dir,
        )
        plot_cp_sequence_components(
            adata,
            output_dir=cp_sequence_dir,
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
        )

    # ============================================================
    # 7) Non-negative CP PARAFAC factorization of full OHE sequences with mask layer
    # ============================================================
    SUBSET = "non_mod_site_ohe_sequence_N_masked_non_negative"

    cp_sequence_dir = latent_dir_dedup / f"03_cp_{SUBSET}"

    # Calculate CP tensor factorization
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning(
            "Layer %s not found; skipping sequence integer encoding CP.",
            SEQUENCE_INTEGER_ENCODING,
        )
    else:
        adata = calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=~mod_sites_mask,
            var_mask_name="non_mod_site_reference_positions",
            rank=2,
            embedding_key=f"X_cp_{SUBSET}",
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
            non_negative=True,
        )

    # CP decomposition using sequence integer encoding (no var filters)
    if cp_sequence_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{cp_sequence_dir} already exists. Skipping sequence CP plotting.")
    else:
        make_dirs([cp_sequence_dir])
        plot_embedding_grid(
            adata,
            basis=f"cp_{SUBSET}",
            color=plotting_layers,
            output_dir=cp_sequence_dir,
        )
        plot_cp_sequence_components(
            adata,
            output_dir=cp_sequence_dir,
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
        )

    # ============================================================
    # 8) CP PARAFAC factorization of full OHE sequences with mask layer
    # ============================================================
    SUBSET = "full_ohe_sequence_N_masked"

    cp_sequence_dir = latent_dir_dedup / f"03_cp_{SUBSET}"

    # Calculate CP tensor factorization
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning(
            "Layer %s not found; skipping sequence integer encoding CP.",
            SEQUENCE_INTEGER_ENCODING,
        )
    else:
        adata = calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=_build_reference_position_mask(adata, references),
            var_mask_name="shared_reference_positions",
            rank=2,
            embedding_key=f"X_cp_{SUBSET}",
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
            non_negative=False,
        )

    # CP decomposition using sequence integer encoding (no var filters)
    if cp_sequence_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{cp_sequence_dir} already exists. Skipping sequence CP plotting.")
    else:
        make_dirs([cp_sequence_dir])
        plot_embedding_grid(
            adata,
            basis=f"cp_{SUBSET}",
            color=plotting_layers,
            output_dir=cp_sequence_dir,
        )
        plot_cp_sequence_components(
            adata,
            output_dir=cp_sequence_dir,
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
        )

    # ============================================================
    # 9) Non-negative CP PARAFAC factorization of full OHE sequences with mask layer
    # ============================================================
    SUBSET = "full_ohe_sequence_N_masked_non_negative"

    cp_sequence_dir = latent_dir_dedup / f"03_cp_{SUBSET}"

    # Calculate CP tensor factorization
    if SEQUENCE_INTEGER_ENCODING not in adata.layers:
        logger.warning(
            "Layer %s not found; skipping sequence integer encoding CP.",
            SEQUENCE_INTEGER_ENCODING,
        )
    else:
        adata = calculate_sequence_cp_decomposition(
            adata,
            layer=SEQUENCE_INTEGER_ENCODING,
            var_mask=_build_reference_position_mask(adata, references),
            var_mask_name="shared_reference_positions",
            rank=2,
            embedding_key=f"X_cp_{SUBSET}",
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
            non_negative=True,
        )

    # CP decomposition using sequence integer encoding (no var filters)
    if cp_sequence_dir.is_dir() and not getattr(cfg, "force_redo_spatial_analyses", False):
        logger.debug(f"{cp_sequence_dir} already exists. Skipping sequence CP plotting.")
    else:
        make_dirs([cp_sequence_dir])
        plot_embedding_grid(
            adata,
            basis=f"cp_{SUBSET}",
            color=plotting_layers,
            output_dir=cp_sequence_dir,
        )
        plot_cp_sequence_components(
            adata,
            output_dir=cp_sequence_dir,
            components_key=f"H_cp_{SUBSET}",
            uns_key=f"cp_{SUBSET}",
        )

    # ============================================================
    # 10) Save latent AnnData
    # ============================================================
    if (not latent_adata_path.exists()) or getattr(cfg, "force_redo_latent_analyses", False):
        logger.info("Saving latent analyzed AnnData (post preprocessing and duplicate removal).")
        record_smftools_metadata(
            adata,
            step_name="latent",
            cfg=cfg,
            config_path=config_path,
            input_paths=[source_adata_path] if source_adata_path else None,
            output_path=latent_adata_path,
        )
        write_gz_h5ad(adata, latent_adata_path)

    return adata, latent_adata_path
