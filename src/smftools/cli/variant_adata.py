from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from smftools.constants import LOGGING_DIR, VARIANT_DIR
from smftools.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def variant_adata(
    config_path: str,
) -> Tuple[Optional[ad.AnnData], Optional[Path]]:
    """
    CLI-facing wrapper for variant analyses.

    Called by: `smftools variant <config_path>`

    Responsibilities:
    - Ensure a usable AnnData exists.
    - Determine which AnnData stages exist.
    - Decide whether to skip (return existing) or run the core.
    - Call `variant_adata_core(...)` when actual work is needed.
    """
    from ..readwrite import safe_read_h5ad
    from .helpers import get_adata_paths, load_experiment_config

    # 1) Ensure config + basic paths via load_adata
    cfg = load_experiment_config(config_path)

    paths = get_adata_paths(cfg)

    pp_path = paths.pp
    pp_dedup_path = paths.pp_dedup
    spatial_path = paths.spatial
    chimeric_path = paths.chimeric
    variant_path = paths.variant
    hmm_path = paths.hmm
    latent_path = paths.latent

    # Stage-skipping logic
    if not getattr(cfg, "force_redo_variant_analyses", False):
        if variant_path.exists():
            logger.info(f"Variant AnnData found: {variant_path}\nSkipping smftools variant")
            return None, spatial_path

    # Helper to load from disk, reusing loaded_adata if it matches
    def _load(path: Path):
        adata, _ = safe_read_h5ad(path)
        return adata

    # 3) Decide which AnnData to use as the *starting point* for  analyses
    if hmm_path.exists():
        start_adata = _load(hmm_path)
        source_path = hmm_path
    elif latent_path.exists():
        start_adata = _load(latent_path)
        source_path = latent_path
    elif spatial_path.exists():
        start_adata = _load(spatial_path)
        source_path = spatial_path
    elif chimeric_path.exists():
        start_adata = _load(chimeric_path)
        source_path = chimeric_path
    elif variant_path.exists():
        start_adata = _load(variant_path)
        source_path = variant_path
    elif pp_dedup_path.exists():
        start_adata = _load(pp_dedup_path)
        source_path = pp_dedup_path
    elif pp_path.exists():
        start_adata = _load(pp_path)
        source_path = pp_path
    else:
        logger.warning(
            "No suitable AnnData found for variant analyses (need at least preprocessed)."
        )
        return None, None

    # 4) Run the core
    adata_variant, variant_path = variant_adata_core(
        adata=start_adata,
        cfg=cfg,
        paths=paths,
        source_adata_path=source_path,
        config_path=config_path,
    )

    return adata_variant, variant_path


def variant_adata_core(
    adata: ad.AnnData,
    cfg,
    paths: AdataPaths,
    source_adata_path: Optional[Path] = None,
    config_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Path]:
    """
    Core variant analysis pipeline.

    Assumes:
    - `cfg` is the ExperimentConfig.

    Does:
    -
    - Save AnnData
    """
    import os
    import warnings
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from ..metadata import record_smftools_metadata
    from ..plotting import (
        plot_mismatch_base_frequency_by_position,
        plot_sequence_integer_encoding_clustermaps,
    )
    from ..preprocessing import (
        append_mismatch_frequency_sites,
        append_sequence_mismatch_annotations,
        load_sample_sheet,
    )
    from ..readwrite import make_dirs
    from .helpers import write_gz_h5ad

    # -----------------------------
    # General setup
    # -----------------------------
    date_str = datetime.today().strftime("%y%m%d")
    now = datetime.now()
    time_str = now.strftime("%H%M%S")
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    output_directory = Path(cfg.output_directory)
    variant_directory = output_directory / VARIANT_DIR
    logging_directory = variant_directory / LOGGING_DIR

    make_dirs([output_directory, variant_directory])

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

    # ============================================================
    # 1) Reference variant position annotation
    # ============================================================
    seq1_col, seq2_col = getattr(cfg, "references_to_align_for_variant_annotation", [None, None])
    if seq1_col and seq2_col:
        append_sequence_mismatch_annotations(adata, seq1_col, seq2_col)

    ############################################### Append mismatch frequency per position ###############################################
    append_mismatch_frequency_sites(
        adata,
        ref_column=cfg.reference_column,
        mismatch_layer=cfg.mismatch_frequency_layer,
        read_span_layer=cfg.mismatch_frequency_read_span_layer,
        mismatch_frequency_range=cfg.mismatch_frequency_range,
        bypass=cfg.bypass_append_mismatch_frequency_sites,
        force_redo=cfg.force_redo_append_mismatch_frequency_sites,
    )

    ############################################### Plot mismatch base frequencies ###############################################
    if cfg.mismatch_frequency_layer not in adata.layers:
        logger.debug(
            "Mismatch layer '%s' not found; skipping mismatch base frequency plots.",
            cfg.mismatch_frequency_layer,
        )
    elif not adata.uns.get("mismatch_integer_encoding_map"):
        logger.debug("Mismatch encoding map not found; skipping mismatch base frequency plots.")
    else:
        mismatch_base_freq_dir = (
            variant_directory / "deduplicated" / "01_mismatch_base_frequency_plots"
        )
        if mismatch_base_freq_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{mismatch_base_freq_dir} already exists. Skipping mismatch base frequency plots."
            )
        else:
            make_dirs([mismatch_base_freq_dir])
            plot_mismatch_base_frequency_by_position(
                adata,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                mismatch_layer=cfg.mismatch_frequency_layer,
                read_span_layer=cfg.mismatch_frequency_read_span_layer,
                exclude_mod_sites=True,  # cfg.mismatch_base_frequency_exclude_mod_sites,
                mod_site_bases=cfg.mod_target_bases,
                save_path=mismatch_base_freq_dir,
                plot_zscores=True,
            )

    ############################################### Plot integer sequence encoding clustermaps ###############################################
    if "sequence_integer_encoding" not in adata.layers:
        logger.debug(
            "sequence_integer_encoding layer not found; skipping integer encoding clustermaps."
        )
    else:
        seq_clustermap_dir = (
            variant_directory / "deduplicated" / "02_sequence_integer_encoding_clustermaps"
        )
        if seq_clustermap_dir.is_dir() and not cfg.force_redo_preprocessing:
            logger.debug(
                f"{seq_clustermap_dir} already exists. Skipping sequence integer encoding clustermaps."
            )
        else:
            make_dirs([seq_clustermap_dir])
            plot_sequence_integer_encoding_clustermaps(
                adata,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                demux_types=cfg.clustermap_demux_types_to_plot,
                min_quality=None,
                min_length=None,
                min_mapped_length_to_reference_length_ratio=None,
                sort_by="none",
                max_unknown_fraction=0.5,
                save_path=seq_clustermap_dir,
                show_position_axis=True,
            )

        if "mismatch_integer_encoding" in adata.layers:
            mismatch_clustermap_dir = (
                variant_directory
                / "deduplicated"
                / "03_mismatch_integer_encoding_clustermaps_no_mod_sites"
            )
            if mismatch_clustermap_dir.is_dir():
                logger.debug(
                    f"{mismatch_clustermap_dir} already exists. "
                    "Skipping mismatch clustermaps without mod sites."
                )
            else:
                make_dirs([mismatch_clustermap_dir])
                plot_sequence_integer_encoding_clustermaps(
                    adata,
                    sample_col=cfg.sample_name_col_for_plotting,
                    reference_col=cfg.reference_column,
                    demux_types=cfg.clustermap_demux_types_to_plot,
                    min_quality=None,
                    min_length=None,
                    min_mapped_length_to_reference_length_ratio=None,
                    sort_by="none",
                    max_unknown_fraction=0.5,
                    save_path=mismatch_clustermap_dir,
                    show_position_axis=True,
                    exclude_mod_sites=True,
                    mod_site_bases=cfg.mod_target_bases,
                )

    # ============================================================
    # 5) Save AnnData
    # ============================================================
    if not paths.variant.exists():
        logger.info("Saving variant AnnData")
        record_smftools_metadata(
            adata,
            step_name="variant",
            cfg=cfg,
            config_path=config_path,
            input_paths=[source_adata_path] if source_adata_path else None,
            output_path=paths.variant,
        )
        write_gz_h5ad(adata, paths.variant)

    return adata, paths.variant
