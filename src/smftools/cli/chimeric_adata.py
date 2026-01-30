from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from smftools.constants import CHIMERIC_DIR, LOGGING_DIR
from smftools.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def chimeric_adata(
    config_path: str,
) -> Tuple[Optional[ad.AnnData], Optional[Path]]:
    """
    CLI-facing wrapper for chimeric analyses.

    Called by: `smftools chimeric <config_path>`

    Responsibilities:
    - Ensure a usable AnnData exists.
    - Determine which AnnData stages exist.
    - Call `chimeric_adata_core(...)` when actual work is needed.
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
    if not getattr(cfg, "force_redo_chimeric_analyses", False):
        if chimeric_path.exists():
            logger.info(f"Chimeric AnnData found: {chimeric_path}\nSkipping smftools chimeric")
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
            "No suitable AnnData found for chimeric analyses (need at least preprocessed)."
        )
        return None, None

    # 4) Run the core
    adata_chimeric, chimeric_path = chimeric_adata_core(
        adata=start_adata,
        cfg=cfg,
        paths=paths,
        source_adata_path=source_path,
        config_path=config_path,
    )

    return adata_chimeric, chimeric_path


def chimeric_adata_core(
    adata: ad.AnnData,
    cfg,
    paths: AdataPaths,
    source_adata_path: Optional[Path] = None,
    config_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Path]:
    """
    Core chimeric analysis pipeline.

    Assumes:
    - `cfg` is the ExperimentConfig.

    Does:
    -
    - Save AnnData.
    """
    import os
    import warnings
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from ..metadata import record_smftools_metadata
    from ..plotting import (
        plot_rolling_nn_and_layer,
        plot_zero_hamming_pair_counts,
        plot_zero_hamming_span_and_layer,
    )
    from ..preprocessing import (
        load_sample_sheet,
    )
    from ..readwrite import make_dirs
    from ..tools import annotate_zero_hamming_segments, rolling_window_nn_distance
    from ..tools.rolling_nn_distance import (
        assign_rolling_nn_results,
        zero_hamming_segments_to_dataframe,
        zero_pairs_to_dataframe,
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
    chimeric_directory = output_directory / CHIMERIC_DIR
    logging_directory = chimeric_directory / LOGGING_DIR

    make_dirs([output_directory, chimeric_directory])

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
    if smf_modality == "direct":
        rolling_nn_layer = "binarized_methylation"
    else:
        rolling_nn_layer = None

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

    references = adata.obs[cfg.reference_column].cat.categories

    # ============================================================
    # 1) Rolling NN distances + layer clustermaps
    # ============================================================
    rolling_nn_dir = chimeric_directory / "01_rolling_nn_clustermaps"

    if rolling_nn_dir.is_dir() and not getattr(cfg, "force_redo_chimeric_analyses", False):
        logger.debug(f"{rolling_nn_dir} already exists. Skipping rolling NN distance plots.")
    else:
        make_dirs([rolling_nn_dir])
        samples = (
            adata.obs[cfg.sample_name_col_for_plotting].astype("category").cat.categories.tolist()
        )
        references = adata.obs[cfg.reference_column].astype("category").cat.categories.tolist()

        for reference in references:
            for sample in samples:
                mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & (
                    adata.obs[cfg.reference_column] == reference
                )
                if not mask.any():
                    continue

                subset = adata[mask]
                position_col = f"position_in_{reference}"
                site_cols = [f"{reference}_{st}_site" for st in cfg.rolling_nn_site_types]
                missing_cols = [
                    col for col in [position_col, *site_cols] if col not in adata.var.columns
                ]
                if missing_cols:
                    raise KeyError(
                        f"Required site mask columns missing in adata.var: {missing_cols}"
                    )
                mod_site_mask = adata.var[site_cols].fillna(False).any(axis=1)
                site_mask = mod_site_mask & adata.var[position_col].fillna(False)
                subset = subset[:, site_mask].copy()
                try:
                    collect_zero_pairs = getattr(cfg, "rolling_nn_collect_zero_pairs", True)
                    zero_pairs_uns_key = getattr(cfg, "rolling_nn_zero_pairs_uns_key", None)
                    rolling_values, rolling_starts = rolling_window_nn_distance(
                        subset,
                        layer=rolling_nn_layer,
                        window=cfg.rolling_nn_window,
                        step=cfg.rolling_nn_step,
                        min_overlap=cfg.rolling_nn_min_overlap,
                        return_fraction=cfg.rolling_nn_return_fraction,
                        store_obsm=cfg.rolling_nn_obsm_key,
                        collect_zero_pairs=collect_zero_pairs,
                        zero_pairs_uns_key=zero_pairs_uns_key,
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
                map_key = f"{safe_sample}__{safe_ref}"
                parent_obsm_key = f"{cfg.rolling_nn_obsm_key}__{safe_ref}"
                try:
                    assign_rolling_nn_results(
                        adata,
                        subset,
                        rolling_values,
                        rolling_starts,
                        obsm_key=parent_obsm_key,
                        window=cfg.rolling_nn_window,
                        step=cfg.rolling_nn_step,
                        min_overlap=cfg.rolling_nn_min_overlap,
                        return_fraction=cfg.rolling_nn_return_fraction,
                        layer=rolling_nn_layer,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to merge rolling NN results for sample=%s ref=%s: %s",
                        sample,
                        reference,
                        exc,
                    )
                if collect_zero_pairs:
                    resolved_zero_pairs_key = (
                        zero_pairs_uns_key
                        if zero_pairs_uns_key is not None
                        else f"{cfg.rolling_nn_obsm_key}_zero_pairs"
                    )
                    parent_zero_pairs_key = f"{parent_obsm_key}__zero_pairs"
                    zero_pairs_data = subset.uns.get(resolved_zero_pairs_key)
                    rolling_zero_pairs_out_dir = rolling_nn_dir / "01_rolling_nn_zero_pairs"
                    write_zero_pairs_csvs = getattr(cfg, "rolling_nn_write_zero_pairs_csvs", True)
                    if zero_pairs_data is not None:
                        adata.uns[parent_zero_pairs_key] = zero_pairs_data
                        for suffix in (
                            "starts",
                            "window",
                            "step",
                            "min_overlap",
                            "return_fraction",
                            "layer",
                        ):
                            value = subset.uns.get(f"{resolved_zero_pairs_key}_{suffix}")
                            if value is not None:
                                adata.uns[f"{parent_zero_pairs_key}_{suffix}"] = value
                        adata.uns.setdefault(
                            f"{cfg.rolling_nn_obsm_key}_zero_pairs_map", {}
                        ).setdefault(map_key, {})["zero_pairs_key"] = parent_zero_pairs_key
                        if write_zero_pairs_csvs:
                            try:
                                make_dirs([rolling_zero_pairs_out_dir])
                                zero_pairs_df = zero_pairs_to_dataframe(
                                    subset, resolved_zero_pairs_key
                                )
                                zero_pairs_df.to_csv(
                                    rolling_zero_pairs_out_dir
                                    / f"{safe_sample}__{safe_ref}__zero_pairs_windows.csv",
                                    index=False,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Failed to write zero-pairs CSV for sample=%s ref=%s: %s",
                                    sample,
                                    reference,
                                    exc,
                                )
                    else:
                        logger.warning(
                            "Zero-pair data missing for sample=%s ref=%s (key=%s).",
                            sample,
                            reference,
                            resolved_zero_pairs_key,
                        )
                    segments_uns_key = getattr(
                        cfg,
                        "rolling_nn_zero_pairs_segments_key",
                        f"{parent_obsm_key}__zero_hamming_segments",
                    )
                    layer_key = getattr(
                        cfg,
                        "rolling_nn_zero_pairs_layer_key",
                        f"{parent_obsm_key}__zero_span",
                    )
                    try:
                        segment_records = annotate_zero_hamming_segments(
                            subset,
                            zero_pairs_uns_key=resolved_zero_pairs_key,
                            output_uns_key=segments_uns_key,
                            layer=rolling_nn_layer,
                            min_overlap=cfg.rolling_nn_min_overlap,
                            refine_segments=getattr(cfg, "rolling_nn_zero_pairs_refine", True),
                            binary_layer_key=layer_key,
                            parent_adata=adata,
                        )
                        adata.uns.setdefault(
                            f"{cfg.rolling_nn_obsm_key}_zero_pairs_map", {}
                        ).setdefault(map_key, {}).update(
                            {"segments_key": segments_uns_key, "layer_key": layer_key}
                        )
                        if write_zero_pairs_csvs:
                            try:
                                make_dirs([rolling_zero_pairs_out_dir])
                                segments_df = zero_hamming_segments_to_dataframe(
                                    segment_records, subset.var_names.to_numpy()
                                )
                                segments_df.to_csv(
                                    rolling_zero_pairs_out_dir
                                    / f"{safe_sample}__{safe_ref}__zero_pairs_segments.csv",
                                    index=False,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Failed to write zero-pair segments CSV for sample=%s ref=%s: %s",
                                    sample,
                                    reference,
                                    exc,
                                )
                    except Exception as exc:
                        logger.warning(
                            "Failed to annotate zero-pair segments for sample=%s ref=%s: %s",
                            sample,
                            reference,
                            exc,
                        )
                adata.uns.setdefault(f"{cfg.rolling_nn_obsm_key}_reference_map", {})[reference] = (
                    parent_obsm_key
                )
                out_png = rolling_nn_dir / f"{safe_sample}__{safe_ref}.png"
                title = f"{sample} {reference}"
                try:
                    plot_rolling_nn_and_layer(
                        subset,
                        obsm_key=cfg.rolling_nn_obsm_key,
                        layer_key=cfg.rolling_nn_plot_layer,
                        fill_nn_with_colmax=False,
                        drop_all_nan_windows=False,
                        max_nan_fraction=cfg.position_max_nan_threshold,
                        var_valid_fraction_col=f"{reference}_valid_fraction",
                        title=title,
                        save_name=out_png,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed rolling NN plot for sample=%s ref=%s: %s",
                        sample,
                        reference,
                        exc,
                    )

    # ============================================================
    # 2) Zero-Hamming span clustermaps
    # ============================================================
    zero_hamming_dir = chimeric_directory / "02_zero_hamming_span_clustermaps"

    if zero_hamming_dir.is_dir():
        logger.debug(f"{zero_hamming_dir} already exists. Skipping zero-Hamming plots.")
    else:
        zero_pairs_map = adata.uns.get(f"{cfg.rolling_nn_obsm_key}_zero_pairs_map", {})
        if zero_pairs_map:
            make_dirs([zero_hamming_dir])
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

                    safe_sample = str(sample).replace(os.sep, "_")
                    safe_ref = str(reference).replace(os.sep, "_")
                    map_key = f"{safe_sample}__{safe_ref}"
                    map_entry = zero_pairs_map.get(map_key)
                    if not map_entry:
                        continue

                    layer_key = map_entry.get("layer_key")
                    if not layer_key or layer_key not in adata.layers:
                        logger.warning(
                            "Zero-Hamming span layer %s missing for sample=%s ref=%s.",
                            layer_key,
                            sample,
                            reference,
                        )
                        continue

                    subset = adata[mask]
                    position_col = f"position_in_{reference}"
                    site_cols = [f"{reference}_{st}_site" for st in cfg.rolling_nn_site_types]
                    missing_cols = [
                        col for col in [position_col, *site_cols] if col not in adata.var.columns
                    ]
                    if missing_cols:
                        raise KeyError(
                            f"Required site mask columns missing in adata.var: {missing_cols}"
                        )
                    mod_site_mask = adata.var[site_cols].fillna(False).any(axis=1)
                    site_mask = mod_site_mask & adata.var[position_col].fillna(False)
                    subset = subset[:, site_mask].copy()
                    title = f"{sample} {reference}"
                    out_png = zero_hamming_dir / f"{safe_sample}__{safe_ref}.png"
                    try:
                        plot_zero_hamming_span_and_layer(
                            subset,
                            span_layer_key=layer_key,
                            layer_key=cfg.rolling_nn_plot_layer,
                            max_nan_fraction=cfg.position_max_nan_threshold,
                            var_valid_fraction_col=f"{reference}_valid_fraction",
                            title=title,
                            save_name=out_png,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed zero-Hamming span plot for sample=%s ref=%s: %s",
                            sample,
                            reference,
                            exc,
                        )
                    zero_pairs_key = map_entry.get("zero_pairs_key")
                    if zero_pairs_key and zero_pairs_key in adata.uns:
                        subset.uns[zero_pairs_key] = adata.uns[zero_pairs_key]
                        for suffix in (
                            "starts",
                            "window",
                            "step",
                            "min_overlap",
                            "return_fraction",
                            "layer",
                        ):
                            meta_key = f"{zero_pairs_key}_{suffix}"
                            if meta_key in adata.uns:
                                subset.uns[meta_key] = adata.uns[meta_key]
                        counts_png = zero_hamming_dir / f"{safe_sample}__{safe_ref}__zero_pairs.png"
                        try:
                            plot_zero_hamming_pair_counts(
                                subset,
                                zero_pairs_uns_key=zero_pairs_key,
                                title=title,
                                save_name=counts_png,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed zero-pair count plot for sample=%s ref=%s: %s",
                                sample,
                                reference,
                                exc,
                            )
        else:
            logger.debug("No zero-pair map found; skipping zero-Hamming span clustermaps.")

    # ============================================================
    # 5) Save AnnData
    # ============================================================
    if not paths.chimeric.exists():
        logger.info("Saving chimeric analyzed AnnData")
        record_smftools_metadata(
            adata,
            step_name="chimeric",
            cfg=cfg,
            config_path=config_path,
            input_paths=[source_adata_path] if source_adata_path else None,
            output_path=paths.chimeric,
        )
        write_gz_h5ad(adata, paths.chimeric)

    return adata, paths.chimeric
