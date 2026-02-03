from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad

from smftools.constants import CHIMERIC_DIR, LOGGING_DIR
from smftools.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

ZERO_HAMMING_DISTANCE_SPANS = "zero_hamming_distance_spans"


def _max_positive_span_length(delta_row: "np.ndarray") -> int:
    """Return the max contiguous run length where delta span values are > 0."""
    import numpy as np

    values = np.asarray(delta_row)
    if values.ndim != 1 or values.size == 0:
        return 0

    positive_mask = values > 0
    if not np.any(positive_mask):
        return 0

    transitions = np.diff(positive_mask.astype(np.int8))
    starts = np.flatnonzero(transitions == 1) + 1
    ends = np.flatnonzero(transitions == -1) + 1

    if positive_mask[0]:
        starts = np.r_[0, starts]
    if positive_mask[-1]:
        ends = np.r_[ends, positive_mask.size]

    return int(np.max(ends - starts))


def _compute_chimeric_by_mod_hamming_distance(
    delta_layer: "np.ndarray",
    span_threshold: int,
) -> "np.ndarray":
    """Flag reads with any delta-hamming span strictly larger than ``span_threshold``."""
    import numpy as np

    delta_values = np.asarray(delta_layer)
    if delta_values.ndim != 2:
        raise ValueError("delta_layer must be a 2D array with shape (n_obs, n_vars).")

    flags = np.zeros(delta_values.shape[0], dtype=bool)
    for obs_idx, row in enumerate(delta_values):
        flags[obs_idx] = _max_positive_span_length(row) > span_threshold
    return flags


def _build_top_segments_obs_tuples(
    read_df: "pd.DataFrame",
    obs_names: "pd.Index",
) -> list[tuple[int, int, str]]:
    """
    Build per-read top-segment tuples with integer spans and partner names.

    Args:
        read_df: DataFrame for a single read containing segment and partner fields.
        obs_names: AnnData obs names used to resolve partner indices.

    Returns:
        List of ``(segment_start, segment_end_exclusive, partner_name)`` tuples.
    """
    import pandas as pd

    tuples: list[tuple[int, int, str]] = []
    for row in read_df.itertuples(index=False):
        start_val = int(row.segment_start_label)
        end_val = int(row.segment_end_label)
        partner_name = row.partner_name
        if partner_name is None or pd.isna(partner_name):
            partner_id = int(row.partner_id)
            if 0 <= partner_id < len(obs_names):
                partner_name = str(obs_names[partner_id])
            else:
                partner_name = str(partner_id)
        tuples.append((start_val, end_val, str(partner_name)))
    return tuples


def _build_zero_hamming_span_layer_from_obs(
    adata: ad.AnnData,
    obs_key: str,
    layer_key: str,
) -> None:
    """
    Populate a count-based span layer from per-read obs tuples.

    Args:
        adata: AnnData to receive/update the layer.
        obs_key: obs column containing ``(start_label, end_label, partner)`` tuples.
        layer_key: Name of the layer to create/update.
    """
    import numpy as np
    import pandas as pd

    if obs_key not in adata.obs:
        return

    try:
        label_indexer = {int(label): idx for idx, label in enumerate(adata.var_names)}
    except (TypeError, ValueError):
        logger.warning(
            "Unable to build span layer %s: adata.var_names are not numeric labels.",
            layer_key,
        )
        return

    if layer_key in adata.layers:
        target_layer = np.asarray(adata.layers[layer_key])
        if target_layer.shape != (adata.n_obs, adata.n_vars):
            target_layer = np.zeros((adata.n_obs, adata.n_vars), dtype=np.uint16)
    else:
        target_layer = np.zeros((adata.n_obs, adata.n_vars), dtype=np.uint16)

    for obs_idx, spans in enumerate(adata.obs[obs_key].tolist()):
        if not isinstance(spans, list):
            continue
        for span in spans:
            if span is None or len(span) < 2:
                continue
            start_val, end_val = span[0], span[1]
            if start_val is None or end_val is None:
                continue
            if pd.isna(start_val) or pd.isna(end_val):
                continue
            try:
                start_label = int(start_val)
                end_label = int(end_val)
            except (TypeError, ValueError):
                continue
            if start_label not in label_indexer or end_label not in label_indexer:
                continue
            start_idx = label_indexer[start_label]
            end_idx = label_indexer[end_label]
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            target_layer[obs_idx, start_idx : end_idx + 1] = 1

    adata.layers[layer_key] = target_layer


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
        plot_delta_hamming_summary,
        plot_rolling_nn_and_layer,
        plot_rolling_nn_and_two_layers,
        plot_segment_length_histogram,
        plot_span_length_distributions,
        plot_zero_hamming_pair_counts,
        plot_zero_hamming_span_and_layer,
    )
    from ..preprocessing import (
        load_sample_sheet,
    )
    from ..readwrite import make_dirs
    from ..tools import (
        annotate_zero_hamming_segments,
        rolling_window_nn_distance,
        select_top_segments_per_read,
    )
    from ..tools.rolling_nn_distance import (
        assign_rolling_nn_results,
        zero_hamming_segments_to_dataframe,
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

    # Auto-detect variant call layer from a prior variant_adata run
    variant_call_layer_name = None
    variant_seq1_label = "seq1"
    variant_seq2_label = "seq2"
    seq1_col, seq2_col = getattr(cfg, "references_to_align_for_variant_annotation", [None, None])
    if seq1_col and seq2_col:
        candidate = f"{seq1_col}__{seq2_col}_variant_call"
        if candidate in adata.layers:
            variant_call_layer_name = candidate
            suffix = "_strand_FASTA_base"
            variant_seq1_label = seq1_col[: -len(suffix)] if seq1_col.endswith(suffix) else seq1_col
            variant_seq2_label = seq2_col[: -len(suffix)] if seq2_col.endswith(suffix) else seq2_col
            logger.info(
                "Detected variant call layer '%s'; will overlay on span clustermaps.",
                variant_call_layer_name,
            )

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
                    rolling_values, rolling_starts = rolling_window_nn_distance(
                        subset,
                        layer=rolling_nn_layer,
                        window=cfg.rolling_nn_window,
                        step=cfg.rolling_nn_step,
                        min_overlap=cfg.rolling_nn_min_overlap,
                        return_fraction=cfg.rolling_nn_return_fraction,
                        store_obsm=cfg.rolling_nn_obsm_key,
                        collect_zero_pairs=True,
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
                resolved_zero_pairs_key = f"{cfg.rolling_nn_obsm_key}_zero_pairs"
                parent_zero_pairs_key = f"{parent_obsm_key}__zero_pairs"
                zero_pairs_data = subset.uns.get(resolved_zero_pairs_key)
                rolling_zero_pairs_out_dir = rolling_nn_dir / "01_rolling_nn_zero_pairs"
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
                else:
                    logger.warning(
                        "Zero-pair data missing for sample=%s ref=%s (key=%s).",
                        sample,
                        reference,
                        resolved_zero_pairs_key,
                    )
                try:
                    segments_uns_key = f"{parent_obsm_key}__zero_hamming_segments"
                    segment_records = annotate_zero_hamming_segments(
                        subset,
                        zero_pairs_uns_key=resolved_zero_pairs_key,
                        output_uns_key=segments_uns_key,
                        layer=rolling_nn_layer,
                        min_overlap=cfg.rolling_nn_min_overlap,
                        refine_segments=getattr(cfg, "rolling_nn_zero_pairs_refine", True),
                        max_nan_run=getattr(cfg, "rolling_nn_zero_pairs_max_nan_run", None),
                        merge_gap=getattr(cfg, "rolling_nn_zero_pairs_merge_gap", 0),
                        max_segments_per_read=getattr(
                            cfg, "rolling_nn_zero_pairs_max_segments_per_read", None
                        ),
                        max_segment_overlap=getattr(cfg, "rolling_nn_zero_pairs_max_overlap", None),
                    )
                    adata.uns.setdefault(
                        f"{cfg.rolling_nn_obsm_key}_zero_pairs_map", {}
                    ).setdefault(map_key, {}).update({"segments_key": segments_uns_key})
                    if getattr(cfg, "rolling_nn_write_zero_pairs_csvs", True):
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
                    top_segments_per_read = getattr(
                        cfg, "rolling_nn_zero_pairs_top_segments_per_read", None
                    )
                    if top_segments_per_read is not None:
                        raw_df, filtered_df = select_top_segments_per_read(
                            segment_records,
                            subset.var_names.to_numpy(),
                            max_segments_per_read=top_segments_per_read,
                            max_segment_overlap=getattr(
                                cfg, "rolling_nn_zero_pairs_top_segments_max_overlap", None
                            ),
                            min_span=getattr(
                                cfg, "rolling_nn_zero_pairs_top_segments_min_span", None
                            ),
                        )
                        per_read_layer_key = ZERO_HAMMING_DISTANCE_SPANS
                        per_read_obs_key = f"{parent_obsm_key}__top_segments"
                        if per_read_obs_key in adata.obs:
                            per_read_obs_series = adata.obs[per_read_obs_key].copy()
                            per_read_obs_series = per_read_obs_series.apply(
                                lambda value: value if isinstance(value, list) else []
                            )
                        else:
                            per_read_obs_series = pd.Series(
                                [list() for _ in range(adata.n_obs)],
                                index=adata.obs_names,
                                dtype=object,
                            )
                        if not filtered_df.empty:
                            for read_id, read_df in filtered_df.groupby("read_id", sort=False):
                                read_index = int(read_id)
                                if read_index < 0 or read_index >= subset.n_obs:
                                    continue
                                tuples = _build_top_segments_obs_tuples(
                                    read_df,
                                    subset.obs_names,
                                )
                                per_read_obs_series.at[subset.obs_names[read_index]] = tuples
                        adata.obs[per_read_obs_key] = per_read_obs_series
                        _build_zero_hamming_span_layer_from_obs(
                            adata=adata,
                            obs_key=per_read_obs_key,
                            layer_key=per_read_layer_key,
                        )
                        adata.uns.setdefault(
                            f"{cfg.rolling_nn_obsm_key}_zero_pairs_map", {}
                        ).setdefault(map_key, {}).update(
                            {
                                "per_read_layer_key": per_read_layer_key,
                                "per_read_obs_key": per_read_obs_key,
                            }
                        )
                        if getattr(cfg, "rolling_nn_zero_pairs_top_segments_write_csvs", True):
                            try:
                                make_dirs([rolling_zero_pairs_out_dir])
                                filtered_df.to_csv(
                                    rolling_zero_pairs_out_dir
                                    / f"{safe_sample}__{safe_ref}__zero_pairs_top_segments_per_read.csv",
                                    index=False,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Failed to write top segments CSV for sample=%s ref=%s: %s",
                                    sample,
                                    reference,
                                    exc,
                                )
                        histogram_dir = rolling_zero_pairs_out_dir / "segment_histograms"
                        try:
                            make_dirs([histogram_dir])
                            raw_lengths = raw_df["segment_length_label"].to_numpy()
                            filtered_lengths = filtered_df["segment_length_label"].to_numpy()
                            hist_title = f"{sample} {reference} (n={subset.n_obs})"
                            plot_segment_length_histogram(
                                raw_lengths,
                                filtered_lengths,
                                bins=getattr(
                                    cfg,
                                    "rolling_nn_zero_pairs_segment_histogram_bins",
                                    30,
                                ),
                                title=hist_title,
                                density=True,
                                save_name=histogram_dir
                                / f"{safe_sample}__{safe_ref}__segment_lengths.png",
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to plot segment length histogram for sample=%s ref=%s: %s",
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
                title = f"{sample} {reference} (n={subset.n_obs}) | window={cfg.rolling_nn_window}"
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

                    layer_key = map_entry.get("per_read_layer_key")
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
                    # Build variant call DataFrame before column filtering
                    _variant_call_df = None
                    if variant_call_layer_name and variant_call_layer_name in adata.layers:
                        _vc = adata[mask].layers[variant_call_layer_name]
                        _vc = _vc.toarray() if hasattr(_vc, "toarray") else np.asarray(_vc)
                        _variant_call_df = pd.DataFrame(
                            _vc,
                            index=adata[mask].obs_names.astype(str),
                            columns=adata.var_names,
                        )

                    subset = subset[:, site_mask].copy()
                    title = f"{sample} {reference} (n={subset.n_obs})"
                    out_png = zero_hamming_dir / f"{safe_sample}__{safe_ref}.png"
                    try:
                        plot_zero_hamming_span_and_layer(
                            subset,
                            span_layer_key=layer_key,
                            layer_key=cfg.rolling_nn_plot_layer,
                            max_nan_fraction=cfg.position_max_nan_threshold,
                            var_valid_fraction_col=f"{reference}_valid_fraction",
                            variant_call_data=_variant_call_df,
                            seq1_label=variant_seq1_label,
                            seq2_label=variant_seq2_label,
                            ref1_marker_color=getattr(cfg, "variant_overlay_seq1_color", "white"),
                            ref2_marker_color=getattr(cfg, "variant_overlay_seq2_color", "black"),
                            variant_marker_size=getattr(cfg, "variant_overlay_marker_size", 4.0),
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
        else:
            logger.debug("No zero-pair map found; skipping zero-Hamming span clustermaps.")

    # ============================================================
    # 3) Rolling NN + two-layer clustermaps
    # ============================================================
    rolling_nn_layers_dir = chimeric_directory / "03_rolling_nn_two_layer_clustermaps"
    zero_pairs_map = adata.uns.get(f"{cfg.rolling_nn_obsm_key}_zero_pairs_map", {})

    if rolling_nn_layers_dir.is_dir() and not getattr(cfg, "force_redo_chimeric_analyses", False):
        logger.debug(
            "%s already exists. Skipping rolling NN two-layer clustermaps.",
            rolling_nn_layers_dir,
        )
    else:
        plot_layers = list(getattr(cfg, "rolling_nn_plot_layers", []) or [])
        if len(plot_layers) != 2:
            logger.warning(
                "rolling_nn_plot_layers should list exactly two layers; got %s. Skipping.",
                plot_layers,
            )
        else:
            make_dirs([rolling_nn_layers_dir])
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
                    parent_obsm_key = f"{cfg.rolling_nn_obsm_key}__{safe_ref}"
                    map_key = f"{safe_sample}__{safe_ref}"

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

                    if (
                        parent_obsm_key not in subset.obsm
                        and cfg.rolling_nn_obsm_key not in subset.obsm
                    ):
                        logger.warning(
                            "Rolling NN results missing for sample=%s ref=%s (key=%s).",
                            sample,
                            reference,
                            parent_obsm_key,
                        )
                        continue
                    plot_layers_resolved = list(plot_layers)
                    map_entry = zero_pairs_map.get(map_key, {})
                    zero_hamming_layer_key = map_entry.get("per_read_layer_key")
                    if zero_hamming_layer_key and len(plot_layers_resolved) == 2:
                        plot_layers_resolved[1] = zero_hamming_layer_key
                    elif (
                        ZERO_HAMMING_DISTANCE_SPANS in subset.layers
                        and len(plot_layers_resolved) == 2
                    ):
                        plot_layers_resolved[1] = ZERO_HAMMING_DISTANCE_SPANS

                    if (
                        cfg.rolling_nn_obsm_key not in subset.obsm
                        and parent_obsm_key in subset.obsm
                    ):
                        subset.obsm[cfg.rolling_nn_obsm_key] = subset.obsm[parent_obsm_key]
                        for suffix in (
                            "starts",
                            "centers",
                            "window",
                            "step",
                            "min_overlap",
                            "return_fraction",
                            "layer",
                        ):
                            parent_key = f"{parent_obsm_key}_{suffix}"
                            if parent_key in adata.uns:
                                subset.uns.setdefault(
                                    f"{cfg.rolling_nn_obsm_key}_{suffix}", adata.uns[parent_key]
                                )

                    missing_layers = [
                        layer_key
                        for layer_key in plot_layers_resolved
                        if layer_key not in subset.layers
                    ]
                    if missing_layers:
                        logger.warning(
                            "Layer(s) %s missing for sample=%s ref=%s.",
                            missing_layers,
                            sample,
                            reference,
                        )
                        continue

                    out_png = rolling_nn_layers_dir / f"{safe_sample}__{safe_ref}.png"
                    title = (
                        f"{sample} {reference} (n={subset.n_obs}) | window={cfg.rolling_nn_window}"
                    )
                    try:
                        plot_rolling_nn_and_two_layers(
                            subset,
                            obsm_key=cfg.rolling_nn_obsm_key,
                            layer_keys=plot_layers_resolved,
                            fill_nn_with_colmax=False,
                            drop_all_nan_windows=False,
                            max_nan_fraction=cfg.position_max_nan_threshold,
                            var_valid_fraction_col=f"{reference}_valid_fraction",
                            title=title,
                            save_name=out_png,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed rolling NN two-layer plot for sample=%s ref=%s: %s",
                            sample,
                            reference,
                            exc,
                        )

    # ============================================================
    # Cross-sample rolling NN analysis
    # ============================================================
    if getattr(cfg, "cross_sample_analysis", False):
        CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS = "cross_sample_zero_hamming_distance_spans"
        cross_nn_dir = chimeric_directory / "cross_sample_01_rolling_nn_clustermaps"
        cross_zh_dir = chimeric_directory / "cross_sample_02_zero_hamming_span_clustermaps"
        cross_two_dir = chimeric_directory / "cross_sample_03_rolling_nn_two_layer_clustermaps"

        if cross_nn_dir.is_dir() and not getattr(cfg, "force_redo_chimeric_analyses", False):
            logger.debug("Cross-sample dirs exist. Skipping cross-sample analysis.")
        else:
            make_dirs([cross_nn_dir, cross_zh_dir, cross_two_dir])
            samples = (
                adata.obs[cfg.sample_name_col_for_plotting]
                .astype("category")
                .cat.categories.tolist()
            )
            references = adata.obs[cfg.reference_column].astype("category").cat.categories.tolist()
            rng = np.random.RandomState(getattr(cfg, "cross_sample_random_seed", 42))

            for reference in references:
                ref_mask = adata.obs[cfg.reference_column] == reference
                position_col = f"position_in_{reference}"
                site_cols = [f"{reference}_{st}_site" for st in cfg.rolling_nn_site_types]
                missing_cols = [
                    col for col in [position_col, *site_cols] if col not in adata.var.columns
                ]
                if missing_cols:
                    logger.warning(
                        "Cross-sample: missing var columns %s for ref=%s, skipping.",
                        missing_cols,
                        reference,
                    )
                    continue
                mod_site_mask = adata.var[site_cols].fillna(False).any(axis=1)
                site_mask = mod_site_mask & adata.var[position_col].fillna(False)

                for sample in samples:
                    sample_mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & ref_mask
                    if not sample_mask.any():
                        continue

                    # Build cross-sample pool
                    grouping_col = getattr(cfg, "cross_sample_grouping_col", None)
                    if grouping_col and grouping_col in adata.obs.columns:
                        sample_group_val = adata.obs.loc[sample_mask, grouping_col].iloc[0]
                        pool_mask = ref_mask & (adata.obs[grouping_col] == sample_group_val)
                    else:
                        pool_mask = ref_mask

                    other_mask = pool_mask & ~sample_mask
                    if not other_mask.any():
                        logger.debug(
                            "Cross-sample: no other-sample reads for sample=%s ref=%s.",
                            sample,
                            reference,
                        )
                        continue

                    n_sample = int(sample_mask.sum())
                    n_other = int(other_mask.sum())
                    n_use = min(n_sample, n_other)

                    other_indices = np.where(other_mask.values)[0]
                    if n_other > n_use:
                        other_indices = rng.choice(other_indices, size=n_use, replace=False)

                    sample_indices = np.where(sample_mask.values)[0]
                    combined_indices = np.concatenate([sample_indices, other_indices])
                    cross_subset = adata[combined_indices][:, site_mask].copy()

                    # Build sample_labels: 0 = current sample, 1 = other
                    cross_labels = np.zeros(len(combined_indices), dtype=np.int32)
                    cross_labels[len(sample_indices) :] = 1

                    cross_obsm_key = "cross_sample_rolling_nn_dist"
                    try:
                        rolling_values, rolling_starts = rolling_window_nn_distance(
                            cross_subset,
                            layer=rolling_nn_layer,
                            window=cfg.rolling_nn_window,
                            step=cfg.rolling_nn_step,
                            min_overlap=cfg.rolling_nn_min_overlap,
                            return_fraction=cfg.rolling_nn_return_fraction,
                            store_obsm=cross_obsm_key,
                            collect_zero_pairs=True,
                            sample_labels=cross_labels,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Cross-sample rolling NN failed for sample=%s ref=%s: %s",
                            sample,
                            reference,
                            exc,
                        )
                        continue

                    safe_sample = str(sample).replace(os.sep, "_")
                    safe_ref = str(reference).replace(os.sep, "_")

                    # Assign results back to adata for sample reads only
                    parent_obsm_key = f"cross_sample_rolling_nn_dist__{safe_ref}"
                    sample_rolling = rolling_values[: len(sample_indices)]
                    try:
                        assign_rolling_nn_results(
                            adata,
                            cross_subset[: len(sample_indices)],
                            sample_rolling,
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
                            "Failed to merge cross-sample rolling NN for sample=%s ref=%s: %s",
                            sample,
                            reference,
                            exc,
                        )

                    # Zero-pair segments
                    resolved_zero_pairs_key = f"{cross_obsm_key}_zero_pairs"
                    zero_pairs_data = cross_subset.uns.get(resolved_zero_pairs_key)
                    if zero_pairs_data is not None:
                        try:
                            segments_uns_key = f"{parent_obsm_key}__zero_hamming_segments"
                            segment_records = annotate_zero_hamming_segments(
                                cross_subset,
                                zero_pairs_uns_key=resolved_zero_pairs_key,
                                output_uns_key=segments_uns_key,
                                layer=rolling_nn_layer,
                                min_overlap=cfg.rolling_nn_min_overlap,
                                refine_segments=getattr(cfg, "rolling_nn_zero_pairs_refine", True),
                                max_nan_run=getattr(cfg, "rolling_nn_zero_pairs_max_nan_run", None),
                                merge_gap=getattr(cfg, "rolling_nn_zero_pairs_merge_gap", 0),
                                max_segments_per_read=getattr(
                                    cfg, "rolling_nn_zero_pairs_max_segments_per_read", None
                                ),
                                max_segment_overlap=getattr(
                                    cfg, "rolling_nn_zero_pairs_max_overlap", None
                                ),
                            )

                            top_segments_per_read = getattr(
                                cfg, "rolling_nn_zero_pairs_top_segments_per_read", None
                            )
                            if top_segments_per_read is not None:
                                raw_df, filtered_df = select_top_segments_per_read(
                                    segment_records,
                                    cross_subset.var_names.to_numpy(),
                                    max_segments_per_read=top_segments_per_read,
                                    max_segment_overlap=getattr(
                                        cfg, "rolling_nn_zero_pairs_top_segments_max_overlap", None
                                    ),
                                    min_span=getattr(
                                        cfg, "rolling_nn_zero_pairs_top_segments_min_span", None
                                    ),
                                )
                                per_read_layer_key = CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS
                                per_read_obs_key = f"{parent_obsm_key}__top_segments"
                                if per_read_obs_key in adata.obs:
                                    per_read_obs_series = adata.obs[per_read_obs_key].copy()
                                    per_read_obs_series = per_read_obs_series.apply(
                                        lambda value: value if isinstance(value, list) else []
                                    )
                                else:
                                    per_read_obs_series = pd.Series(
                                        [list() for _ in range(adata.n_obs)],
                                        index=adata.obs_names,
                                        dtype=object,
                                    )
                                if not filtered_df.empty:
                                    for read_id, read_df in filtered_df.groupby(
                                        "read_id", sort=False
                                    ):
                                        read_index = int(read_id)
                                        if read_index < 0 or read_index >= cross_subset.n_obs:
                                            continue
                                        # Only assign for sample reads
                                        if read_index >= len(sample_indices):
                                            continue
                                        tuples = _build_top_segments_obs_tuples(
                                            read_df,
                                            cross_subset.obs_names,
                                        )
                                        per_read_obs_series.at[
                                            cross_subset.obs_names[read_index]
                                        ] = tuples
                                adata.obs[per_read_obs_key] = per_read_obs_series
                                _build_zero_hamming_span_layer_from_obs(
                                    adata=adata,
                                    obs_key=per_read_obs_key,
                                    layer_key=per_read_layer_key,
                                )
                        except Exception as exc:
                            logger.warning(
                                "Cross-sample zero-pair segments failed for sample=%s ref=%s: %s",
                                sample,
                                reference,
                                exc,
                            )

                    # Build variant call DataFrame before column filtering
                    _cross_variant_call_df = None
                    if variant_call_layer_name and variant_call_layer_name in adata.layers:
                        _vc = adata[sample_mask].layers[variant_call_layer_name]
                        _vc = _vc.toarray() if hasattr(_vc, "toarray") else np.asarray(_vc)
                        _cross_variant_call_df = pd.DataFrame(
                            _vc,
                            index=adata[sample_mask].obs_names.astype(str),
                            columns=adata.var_names,
                        )

                    # --- Plots ---
                    # Use the sample-only subset for plotting
                    plot_subset = adata[sample_mask][:, site_mask].copy()

                    # Copy cross-sample obsm into plot_subset
                    if parent_obsm_key in adata.obsm:
                        plot_subset.obsm[cfg.rolling_nn_obsm_key] = adata[sample_mask].obsm.get(
                            parent_obsm_key
                        )
                        for suffix in (
                            "starts",
                            "centers",
                            "window",
                            "step",
                            "min_overlap",
                            "return_fraction",
                            "layer",
                        ):
                            parent_key = f"{parent_obsm_key}_{suffix}"
                            if parent_key in adata.uns:
                                plot_subset.uns[f"{cfg.rolling_nn_obsm_key}_{suffix}"] = adata.uns[
                                    parent_key
                                ]

                    if grouping_col and grouping_col in adata.obs.columns:
                        cross_pool_desc = f"cross-sample ({grouping_col}={sample_group_val})"
                    else:
                        cross_pool_desc = "cross-sample (all samples)"
                    title = (
                        f"{sample} {reference} (n={n_sample})"
                        f" | {cross_pool_desc}"
                        f" | subsample={len(other_indices)}"
                        f" | window={cfg.rolling_nn_window}"
                    )

                    # Plot 1: rolling NN clustermap
                    try:
                        out_png = cross_nn_dir / f"{safe_sample}__{safe_ref}.png"
                        plot_rolling_nn_and_layer(
                            plot_subset,
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
                            "Cross-sample rolling NN plot failed for sample=%s ref=%s: %s",
                            sample,
                            reference,
                            exc,
                        )

                    # Plot 2: zero-hamming span clustermap
                    if CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS in adata.layers:
                        try:
                            out_png = cross_zh_dir / f"{safe_sample}__{safe_ref}.png"
                            plot_zero_hamming_span_and_layer(
                                plot_subset,
                                span_layer_key=CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS,
                                layer_key=cfg.rolling_nn_plot_layer,
                                max_nan_fraction=cfg.position_max_nan_threshold,
                                var_valid_fraction_col=f"{reference}_valid_fraction",
                                variant_call_data=_cross_variant_call_df,
                                seq1_label=variant_seq1_label,
                                seq2_label=variant_seq2_label,
                                ref1_marker_color=getattr(
                                    cfg, "variant_overlay_seq1_color", "white"
                                ),
                                ref2_marker_color=getattr(
                                    cfg, "variant_overlay_seq2_color", "black"
                                ),
                                variant_marker_size=getattr(
                                    cfg, "variant_overlay_marker_size", 4.0
                                ),
                                title=title,
                                save_name=out_png,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Cross-sample zero-Hamming span plot failed for sample=%s ref=%s: %s",
                                sample,
                                reference,
                                exc,
                            )

                    # Plot 3: two-layer clustermap
                    plot_layers = list(getattr(cfg, "rolling_nn_plot_layers", []) or [])
                    if len(plot_layers) == 2:
                        plot_layers_resolved = list(plot_layers)
                        if CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS in plot_subset.layers:
                            plot_layers_resolved[1] = CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS
                        missing_layers = [
                            lk for lk in plot_layers_resolved if lk not in plot_subset.layers
                        ]
                        if not missing_layers:
                            try:
                                out_png = cross_two_dir / f"{safe_sample}__{safe_ref}.png"
                                plot_rolling_nn_and_two_layers(
                                    plot_subset,
                                    obsm_key=cfg.rolling_nn_obsm_key,
                                    layer_keys=plot_layers_resolved,
                                    fill_nn_with_colmax=False,
                                    drop_all_nan_windows=False,
                                    max_nan_fraction=cfg.position_max_nan_threshold,
                                    var_valid_fraction_col=f"{reference}_valid_fraction",
                                    title=title,
                                    save_name=out_png,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Cross-sample two-layer plot failed for sample=%s ref=%s: %s",
                                    sample,
                                    reference,
                                    exc,
                                )

    # ============================================================
    # Delta: within-sample minus cross-sample hamming spans (clamped >= 0)
    # ============================================================
    if getattr(cfg, "cross_sample_analysis", False):
        DELTA_ZERO_HAMMING_DISTANCE_SPANS = "delta_zero_hamming_distance_spans"
        delta_summary_dir = chimeric_directory / "delta_hamming_summary"

        if delta_summary_dir.is_dir() and not getattr(cfg, "force_redo_chimeric_analyses", False):
            logger.debug("Delta summary dir exists. Skipping delta analysis.")
        else:
            make_dirs([delta_summary_dir])
            samples = (
                adata.obs[cfg.sample_name_col_for_plotting]
                .astype("category")
                .cat.categories.tolist()
            )
            references = adata.obs[cfg.reference_column].astype("category").cat.categories.tolist()

            # Build delta layer: within - cross, clamped at 0
            if (
                ZERO_HAMMING_DISTANCE_SPANS in adata.layers
                and CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS in adata.layers
            ):
                within_layer = np.asarray(
                    adata.layers[ZERO_HAMMING_DISTANCE_SPANS], dtype=np.float64
                )
                cross_layer = np.asarray(
                    adata.layers[CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS], dtype=np.float64
                )
                delta_layer = np.clip(within_layer - cross_layer, 0, None)
                adata.layers[DELTA_ZERO_HAMMING_DISTANCE_SPANS] = delta_layer
                threshold = getattr(cfg, "delta_hamming_chimeric_span_threshold", 200)
                try:
                    threshold = int(threshold)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid delta_hamming_chimeric_span_threshold=%s; using default 200.",
                        threshold,
                    )
                    threshold = 200
                if threshold < 0:
                    logger.warning(
                        "delta_hamming_chimeric_span_threshold=%s is negative; clamping to 0.",
                        threshold,
                    )
                    threshold = 0
                adata.obs["chimeric_by_mod_hamming_distance"] = (
                    _compute_chimeric_by_mod_hamming_distance(delta_layer, threshold)
                )
            else:
                logger.warning(
                    "Cannot compute delta: missing %s or %s layer.",
                    ZERO_HAMMING_DISTANCE_SPANS,
                    CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS,
                )
                adata.obs["chimeric_by_mod_hamming_distance"] = False

            if DELTA_ZERO_HAMMING_DISTANCE_SPANS in adata.layers:
                for reference in references:
                    ref_mask = adata.obs[cfg.reference_column] == reference
                    position_col = f"position_in_{reference}"
                    site_cols = [f"{reference}_{st}_site" for st in cfg.rolling_nn_site_types]
                    missing_cols = [
                        col for col in [position_col, *site_cols] if col not in adata.var.columns
                    ]
                    if missing_cols:
                        continue
                    mod_site_mask = adata.var[site_cols].fillna(False).any(axis=1)
                    site_mask = mod_site_mask & adata.var[position_col].fillna(False)

                    for sample in samples:
                        sample_mask = (
                            adata.obs[cfg.sample_name_col_for_plotting] == sample
                        ) & ref_mask
                        if not sample_mask.any():
                            continue

                        safe_sample = str(sample).replace(os.sep, "_")
                        safe_ref = str(reference).replace(os.sep, "_")
                        within_obsm_key = f"{cfg.rolling_nn_obsm_key}__{safe_ref}"
                        cross_obsm_key = f"cross_sample_rolling_nn_dist__{safe_ref}"

                        plot_subset = adata[sample_mask][:, site_mask].copy()

                        # Wire self NN obsm
                        self_nn_key = "self_rolling_nn_dist"
                        if within_obsm_key in plot_subset.obsm:
                            plot_subset.obsm[self_nn_key] = plot_subset.obsm[within_obsm_key]
                        elif cfg.rolling_nn_obsm_key in plot_subset.obsm:
                            plot_subset.obsm[self_nn_key] = plot_subset.obsm[
                                cfg.rolling_nn_obsm_key
                            ]
                        else:
                            logger.debug(
                                "Delta: missing self NN obsm for sample=%s ref=%s.",
                                sample,
                                reference,
                            )
                            continue

                        # Wire cross NN obsm
                        cross_nn_key = "cross_rolling_nn_dist"
                        if cross_obsm_key in plot_subset.obsm:
                            plot_subset.obsm[cross_nn_key] = plot_subset.obsm[cross_obsm_key]
                        else:
                            logger.debug(
                                "Delta: missing cross NN obsm for sample=%s ref=%s.",
                                sample,
                                reference,
                            )
                            continue

                        # Copy uns metadata for both NN keys
                        for src_obsm, dst_obsm in (
                            (within_obsm_key, self_nn_key),
                            (cross_obsm_key, cross_nn_key),
                        ):
                            for suffix in (
                                "starts",
                                "centers",
                                "window",
                                "step",
                                "min_overlap",
                                "return_fraction",
                                "layer",
                            ):
                                src_k = f"{src_obsm}_{suffix}"
                                if src_k in adata.uns:
                                    plot_subset.uns[f"{dst_obsm}_{suffix}"] = adata.uns[src_k]

                        # Check required span layers
                        required_layers = [
                            ZERO_HAMMING_DISTANCE_SPANS,
                            CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS,
                            DELTA_ZERO_HAMMING_DISTANCE_SPANS,
                        ]
                        missing_layers = [
                            lk for lk in required_layers if lk not in plot_subset.layers
                        ]
                        if missing_layers:
                            logger.debug(
                                "Delta: missing layers %s for sample=%s ref=%s.",
                                missing_layers,
                                sample,
                                reference,
                            )
                            continue

                        title = (
                            f"{sample} {reference}"
                            f" (n={int(sample_mask.sum())})"
                            f" | window={cfg.rolling_nn_window}"
                        )
                        out_png = delta_summary_dir / f"{safe_sample}__{safe_ref}.png"
                        try:
                            plot_delta_hamming_summary(
                                plot_subset,
                                self_obsm_key=self_nn_key,
                                cross_obsm_key=cross_nn_key,
                                layer_key=cfg.rolling_nn_plot_layer,
                                self_span_layer_key=ZERO_HAMMING_DISTANCE_SPANS,
                                cross_span_layer_key=CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS,
                                delta_span_layer_key=DELTA_ZERO_HAMMING_DISTANCE_SPANS,
                                fill_nn_with_colmax=False,
                                drop_all_nan_windows=False,
                                max_nan_fraction=cfg.position_max_nan_threshold,
                                var_valid_fraction_col=f"{reference}_valid_fraction",
                                title=title,
                                save_name=out_png,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Delta hamming summary plot failed for sample=%s ref=%s: %s",
                                sample,
                                reference,
                                exc,
                            )

    # ============================================================
    # Hamming span trio (self, cross, delta) — no column subsetting
    # ============================================================
    if getattr(cfg, "cross_sample_analysis", False):
        span_trio_dir = chimeric_directory / "hamming_span_trio"

        if span_trio_dir.is_dir() and not getattr(cfg, "force_redo_chimeric_analyses", False):
            logger.debug("Hamming span trio dir exists. Skipping.")
        else:
            _self_key = ZERO_HAMMING_DISTANCE_SPANS
            _cross_key = CROSS_SAMPLE_ZERO_HAMMING_DISTANCE_SPANS
            _delta_key = DELTA_ZERO_HAMMING_DISTANCE_SPANS
            has_layers = (
                _self_key in adata.layers
                and _cross_key in adata.layers
                and _delta_key in adata.layers
            )
            if has_layers:
                from smftools.plotting import plot_hamming_span_trio

                make_dirs([span_trio_dir])
                samples = (
                    adata.obs[cfg.sample_name_col_for_plotting]
                    .astype("category")
                    .cat.categories.tolist()
                )
                references = (
                    adata.obs[cfg.reference_column].astype("category").cat.categories.tolist()
                )

                for reference in references:
                    ref_mask = adata.obs[cfg.reference_column] == reference
                    position_col = f"position_in_{reference}"
                    if position_col not in adata.var.columns:
                        continue
                    pos_mask = adata.var[position_col].fillna(False).astype(bool)

                    for sample in samples:
                        sample_mask = (
                            adata.obs[cfg.sample_name_col_for_plotting] == sample
                        ) & ref_mask
                        if not sample_mask.any():
                            continue

                        # Build variant call DataFrame (full width, no subsetting)
                        _variant_call_df = None
                        if variant_call_layer_name and variant_call_layer_name in adata.layers:
                            _vc = adata[sample_mask].layers[variant_call_layer_name]
                            _vc = _vc.toarray() if hasattr(_vc, "toarray") else np.asarray(_vc)
                            _variant_call_df = pd.DataFrame(
                                _vc,
                                index=adata[sample_mask].obs_names.astype(str),
                                columns=adata.var_names,
                            )

                        plot_subset = adata[sample_mask][:, pos_mask].copy()

                        safe_sample = str(sample).replace(os.sep, "_")
                        safe_ref = str(reference).replace(os.sep, "_")
                        n_reads = int(sample_mask.sum())
                        trio_title = f"{sample} {reference} (n={n_reads})"
                        out_png = span_trio_dir / f"{safe_sample}__{safe_ref}.png"
                        try:
                            plot_hamming_span_trio(
                                plot_subset,
                                self_span_layer_key=_self_key,
                                cross_span_layer_key=_cross_key,
                                delta_span_layer_key=_delta_key,
                                variant_call_data=_variant_call_df,
                                seq1_label=variant_seq1_label,
                                seq2_label=variant_seq2_label,
                                ref1_marker_color=getattr(
                                    cfg, "variant_overlay_seq1_color", "white"
                                ),
                                ref2_marker_color=getattr(
                                    cfg, "variant_overlay_seq2_color", "black"
                                ),
                                variant_marker_size=getattr(
                                    cfg, "variant_overlay_marker_size", 4.0
                                ),
                                title=trio_title,
                                save_name=out_png,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Hamming span trio plot failed for sample=%s ref=%s: %s",
                                sample,
                                reference,
                                exc,
                            )

    # ============================================================
    # Span length distribution histograms
    # ============================================================
    if getattr(cfg, "cross_sample_analysis", False):
        span_hist_dir = chimeric_directory / "span_length_distributions"
        if span_hist_dir.is_dir() and not getattr(cfg, "force_redo_chimeric_analyses", False):
            logger.debug("Span length distribution dir exists. Skipping.")
        else:
            _self_key = ZERO_HAMMING_DISTANCE_SPANS
            _cross_key = "cross_sample_zero_hamming_distance_spans"
            _delta_key = "delta_zero_hamming_distance_spans"
            has_layers = (
                _self_key in adata.layers
                and _cross_key in adata.layers
                and _delta_key in adata.layers
            )
            if has_layers:
                make_dirs([span_hist_dir])
                samples = (
                    adata.obs[cfg.sample_name_col_for_plotting]
                    .astype("category")
                    .cat.categories.tolist()
                )
                references = (
                    adata.obs[cfg.reference_column].astype("category").cat.categories.tolist()
                )
                for reference in references:
                    ref_mask = adata.obs[cfg.reference_column] == reference
                    position_col = f"position_in_{reference}"
                    site_cols = [f"{reference}_{st}_site" for st in cfg.rolling_nn_site_types]
                    missing_cols = [
                        col for col in [position_col, *site_cols] if col not in adata.var.columns
                    ]
                    if missing_cols:
                        continue
                    mod_site_mask = adata.var[site_cols].fillna(False).any(axis=1)
                    site_mask = mod_site_mask & adata.var[position_col].fillna(False)

                    for sample in samples:
                        sample_mask = (
                            adata.obs[cfg.sample_name_col_for_plotting] == sample
                        ) & ref_mask
                        if not sample_mask.any():
                            continue

                        safe_sample = str(sample).replace(os.sep, "_")
                        safe_ref = str(reference).replace(os.sep, "_")
                        plot_subset = adata[sample_mask][:, site_mask].copy()

                        title = f"{sample} {reference} (n={int(sample_mask.sum())})"
                        out_png = span_hist_dir / f"{safe_sample}__{safe_ref}.png"
                        try:
                            plot_span_length_distributions(
                                plot_subset,
                                self_span_layer_key=_self_key,
                                cross_span_layer_key=_cross_key,
                                delta_span_layer_key=_delta_key,
                                bins=getattr(
                                    cfg,
                                    "rolling_nn_zero_pairs_segment_histogram_bins",
                                    30,
                                ),
                                title=title,
                                save_name=out_png,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Span length distribution plot failed for sample=%s ref=%s: %s",
                                sample,
                                reference,
                                exc,
                            )
            else:
                logger.debug("Span length distribution: missing required layers, skipping.")

    # ============================================================
    # 4) Save AnnData
    # ============================================================
    zero_pairs_map_key = f"{cfg.rolling_nn_obsm_key}_zero_pairs_map"
    zero_pairs_map = adata.uns.get(zero_pairs_map_key, {})
    if not getattr(cfg, "rolling_nn_zero_pairs_keep_uns", True):
        for entry in zero_pairs_map.values():
            zero_pairs_key = entry.get("zero_pairs_key")
            if zero_pairs_key and zero_pairs_key in adata.uns:
                del adata.uns[zero_pairs_key]
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
                        del adata.uns[meta_key]
        if zero_pairs_map_key in adata.uns:
            del adata.uns[zero_pairs_map_key]
    if not getattr(cfg, "rolling_nn_zero_pairs_segments_keep_uns", True):
        for entry in zero_pairs_map.values():
            segments_key = entry.get("segments_key")
            if segments_key and segments_key in adata.uns:
                del adata.uns[segments_key]

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
