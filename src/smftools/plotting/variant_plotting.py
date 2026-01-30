from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, List

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

plt = require("matplotlib.pyplot", extra="plotting", purpose="plot rendering")
patches = require("matplotlib.patches", extra="plotting", purpose="plot rendering")
colors = require("matplotlib.colors", extra="plotting", purpose="plot rendering")
grid_spec = require("matplotlib.gridspec", extra="plotting", purpose="heatmap plotting")
sns = require("seaborn", extra="plotting", purpose="plot styling")

logger = get_logger(__name__)

DNA_5COLOR_PALETTE = {
    "A": "#00A000",  # green
    "C": "#0000FF",  # blue
    "G": "#FF7F00",  # orange
    "T": "#FF0000",  # red
    "OTHER": "#808080",  # gray (N, PAD, unknown)
}

def plot_mismatch_base_frequency_by_position(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    mismatch_layer: str = "mismatch_integer_encoding",
    read_span_layer: str = "read_span_mask",
    quality_layer: str = "base_quality_scores",
    plot_zscores: bool = False,
    exclude_mod_sites: bool = False,
    mod_site_bases: Sequence[str] | None = None,
    min_quality: float | None = None,
    min_length: int | None = None,
    min_mapped_length_to_reference_length_ratio: float | None = None,
    demux_types: Sequence[str] = ("single", "double", "already"),
    save_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Plot mismatch base frequencies by position per sample/reference.

    Args:
        adata: AnnData with mismatch integer encoding layer.
        sample_col: Column in ``adata.obs`` that identifies samples.
        reference_col: Column in ``adata.obs`` that identifies references.
        mismatch_layer: Layer name containing mismatch integer encodings.
        read_span_layer: Layer name containing read-span masks.
        quality_layer: Layer name containing base-quality scores used for z-scores.
        plot_zscores: Whether to plot quality-normalized z-scores in a separate panel.
        exclude_mod_sites: Whether to exclude annotated modification sites.
        mod_site_bases: Base-context labels used to build mod-site masks (e.g., ``["GpC", "CpG"]``).
        min_quality: Optional minimum read quality filter.
        min_length: Optional minimum mapped length filter.
        min_mapped_length_to_reference_length_ratio: Optional min length ratio filter.
        demux_types: Allowed ``demux_type`` values, if present in ``adata.obs``.
        save_path: Optional output directory for saving plots.

    Returns:
        List of dictionaries with per-plot metadata and output paths. Includes
        a pooled-samples entry per reference.
    """
    logger.info("Plotting mismatch base frequency by position.")

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=s.index)

    def _build_mod_site_mask(var_frame, ref_name: str) -> np.ndarray | None:
        if not exclude_mod_sites or not mod_site_bases:
            return None

        mod_site_cols = [f"{ref_name}_{base}_site" for base in mod_site_bases]
        missing_required = [col for col in mod_site_cols if col not in var_frame.columns]
        if missing_required:
            return None

        extra_cols = []
        if any(base in {"GpC", "CpG"} for base in mod_site_bases):
            ambiguous_col = f"{ref_name}_ambiguous_GpC_CpG_site"
            if ambiguous_col in var_frame.columns:
                extra_cols.append(ambiguous_col)

        mod_site_cols.extend(extra_cols)
        mod_site_cols = list(dict.fromkeys(mod_site_cols))

        mod_masks = [np.asarray(var_frame[col].values, dtype=bool) for col in mod_site_cols]
        mod_mask = mod_masks[0] if len(mod_masks) == 1 else np.logical_or.reduce(mod_masks)

        position_col = f"position_in_{ref_name}"
        if position_col in var_frame.columns:
            position_mask = np.asarray(var_frame[position_col].values, dtype=bool)
            mod_mask = np.logical_and(mod_mask, position_mask)

        return mod_mask

    def _get_reference_base_series(subset, ref_name: str) -> pd.Series | None:
        if f"{ref_name}_strand_FASTA_base" in subset.var.columns:
            return subset.var[f"{ref_name}_strand_FASTA_base"].astype("string")
        return None

    if mismatch_layer not in adata.layers:
        raise KeyError(f"Layer '{mismatch_layer}' not found in adata.layers")
    if plot_zscores and quality_layer not in adata.layers:
        raise KeyError(f"Layer '{quality_layer}' not found in adata.layers")

    mismatch_map = adata.uns.get("mismatch_integer_encoding_map", {}) or {}
    if not mismatch_map:
        raise KeyError("Mismatch encoding map not found in adata.uns")

    base_int_to_label = {
        int(value): str(base)
        for base, value in mismatch_map.items()
        if base not in {"N", "PAD"} and isinstance(value, (int, np.integer))
    }
    if not base_int_to_label:
        raise ValueError("Mismatch encoding map missing base labels.")

    base_label_to_int = {label: int_val for int_val, label in base_int_to_label.items()}

    def _pooled_label(sample_categories: Sequence[str]) -> str:
        base_label = "pooled_samples"
        if base_label not in sample_categories:
            return base_label
        suffix = 1
        while f"{base_label}_{suffix}" in sample_categories:
            suffix += 1
        return f"{base_label}_{suffix}"

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    sample_categories = [str(sample) for sample in adata.obs[sample_col].cat.categories]
    pooled_sample = _pooled_label(sample_categories)

    for ref in adata.obs[reference_col].cat.categories:
        ref_name = str(ref)
        base_mask = np.ones(adata.n_vars, dtype=bool)
        position_col = f"position_in_{ref_name}"
        if position_col in adata.var.columns:
            base_mask = np.asarray(adata.var[position_col].values, dtype=bool)

        base_mod_mask = _build_mod_site_mask(adata.var, ref_name)
        if base_mod_mask is not None:
            base_mask = base_mask & ~base_mod_mask

        summary_data = {
            "var_names_position": np.asarray(adata.var_names)[base_mask],
        }
        if "Original_var_names" in adata.var.columns:
            summary_data["original_var_names_position"] = np.asarray(
                adata.var["Original_var_names"]
            )[base_mask]
        reindexed_col = f"{ref_name}_reindexed"
        if reindexed_col in adata.var.columns:
            summary_data["reindexed_var_names_position"] = np.asarray(adata.var[reindexed_col])[
                base_mask
            ]
        ref_sequence_col = f"{ref_name}_strand_FASTA_base"
        if ref_sequence_col in adata.var.columns:
            summary_data["reference_sequence_base"] = np.asarray(adata.var[ref_sequence_col])[
                base_mask
            ]

        summary_df = pd.DataFrame(summary_data)
        mean_positional_error = adata.var[f"{ref}_mean_error_rate"].values
        std_positional_error = adata.var[f"{ref}_std_error_rate"].values
        for sample in [*sample_categories, pooled_sample]:
            qmask = _mask_or_true(
                "read_quality",
                (lambda s: s >= float(min_quality))
                if (min_quality is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lm_mask = _mask_or_true(
                "mapped_length",
                (lambda s: s >= float(min_length))
                if (min_length is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lrr_mask = _mask_or_true(
                "mapped_length_to_reference_length_ratio",
                (lambda s: s >= float(min_mapped_length_to_reference_length_ratio))
                if (min_mapped_length_to_reference_length_ratio is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            demux_mask = _mask_or_true(
                "demux_type",
                (lambda s: s.astype("string").isin(list(demux_types)))
                if (demux_types is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            row_mask = (adata.obs[reference_col] == ref) & qmask & lm_mask & lrr_mask & demux_mask
            if sample != pooled_sample:
                row_mask = row_mask & (adata.obs[sample_col] == sample)
            if not bool(row_mask.any()):
                continue

            subset = adata[row_mask, :].copy()
            mismatch_matrix = np.asarray(subset.layers[mismatch_layer])

            if read_span_layer in subset.layers:
                span_matrix = np.asarray(subset.layers[read_span_layer])
                coverage_mask = span_matrix > 0
            else:
                coverage_mask = np.ones_like(mismatch_matrix, dtype=bool)

            ref_bases = _get_reference_base_series(subset, str(ref))
            logger.debug(f"Unconverted reference sequence for {ref}: {ref_bases.values}")
            ref_lower = str(ref).lower()
            if ref_bases is not None:
                target_base = None
                if "top" in ref_lower:
                    target_base = "C"
                elif "bottom" in ref_lower:
                    target_base = "G"
                else:
                    logger.debug(f"Could not find strand in {ref_lower}")
                if target_base in base_label_to_int:
                    target_int = base_label_to_int[target_base]
                    logger.debug(f"Ignoring {target_base} site mismatches in {ref} that yield a mismatch ambiguous to conversion")
                    ref_base_mask = np.asarray(ref_bases.values == target_base, dtype=bool)
                    ignore_mask = (mismatch_matrix == target_int) & ref_base_mask[None, :]
                    coverage_mask = coverage_mask & ~ignore_mask

            coverage_counts = coverage_mask.sum(axis=0).astype(float)

            ref_position_mask = subset.var.get(f"position_in_{ref}")
            if ref_position_mask is None:
                position_mask = np.ones(mismatch_matrix.shape[1], dtype=bool)
            else:
                position_mask = np.asarray(ref_position_mask.values, dtype=bool)

            mod_site_mask = _build_mod_site_mask(subset.var, str(ref))
            if mod_site_mask is not None:
                position_mask = position_mask & ~mod_site_mask

            position_mask = position_mask & (coverage_counts > 0)
            if not np.any(position_mask):
                continue

            positions = np.arange(mismatch_matrix.shape[1])[position_mask]
            mean_errors = mean_positional_error[position_mask]
            normalized_mean_errors = mean_errors / 3 # This is a conservative normalization against variant specific error rate
            std_errors = std_positional_error[position_mask]
            base_freqs: Dict[str, np.ndarray] = {}
            for base_int, base_label in base_int_to_label.items():
                base_counts = ((mismatch_matrix == base_int) & coverage_mask).sum(axis=0)
                freq = np.divide(
                    base_counts,
                    coverage_counts,
                    out=np.full(mismatch_matrix.shape[1], np.nan, dtype=float),
                    where=coverage_counts > 0,
                )
                freq = np.where(freq > 0, freq, np.nan)
                freq = freq[position_mask]
                if np.all(np.isnan(freq)):
                    continue
                base_freqs[base_label] = freq

            if not base_freqs:
                continue

            zscore_freqs: Dict[str, np.ndarray] = {}
            if plot_zscores:
                quality_matrix = np.asarray(subset.layers[quality_layer]).astype(float)
                quality_matrix[quality_matrix < 0] = np.nan
                valid_quality = coverage_mask & ~np.isnan(quality_matrix)
                error_probs = np.power(10.0, -quality_matrix / 10.0)
                error_probs = np.where(valid_quality, error_probs, 0.0)
                variant_probs = error_probs / 3.0
                variance = (variant_probs * (1.0 - variant_probs)).sum(axis=0)
                variance = variance[position_mask]
                variance = np.where(variance > 0, variance, np.nan)

                for base_int, base_label in base_int_to_label.items():
                    base_counts = ((mismatch_matrix == base_int) & coverage_mask).sum(axis=0).astype(float)
                    expected_counts = variant_probs.sum(axis=0)
                    expected_counts = expected_counts[position_mask]
                    observed_counts = base_counts[position_mask]
                    zscores = np.divide(
                        observed_counts - expected_counts,
                        np.sqrt(variance),
                        out=np.full_like(expected_counts, np.nan, dtype=float),
                        where=~np.isnan(variance),
                    )
                    if np.all(np.isnan(zscores)):
                        continue
                    zscore_freqs[base_label] = zscores
            if plot_zscores and save_path is not None:
                full_max = np.full(adata.n_vars, np.nan, dtype=float)
                full_base = np.full(adata.n_vars, None, dtype=object)
                if zscore_freqs:
                    base_labels = sorted(zscore_freqs.keys())
                    zscore_stack = []
                    for base_label in base_labels:
                        full_z = np.full(adata.n_vars, np.nan, dtype=float)
                        full_z[position_mask] = zscore_freqs[base_label]
                        zscore_stack.append(full_z)
                    zscore_stack = np.vstack(zscore_stack)
                    all_nan = np.all(np.isnan(zscore_stack), axis=0)
                    safe_stack = np.where(np.isnan(zscore_stack), -np.inf, zscore_stack)
                    with np.errstate(invalid="ignore"):
                        full_max = np.nanmax(zscore_stack, axis=0)
                    max_idx = np.argmax(safe_stack, axis=0)
                    full_base = np.array([base_labels[idx] for idx in max_idx], dtype=object)
                    full_max[all_nan] = np.nan
                    full_base[all_nan] = None
                summary_df[f"{sample}_max_zscore"] = full_max[base_mask]
                summary_df[f"{sample}_max_zscore_base"] = full_base[base_mask]

            if plot_zscores:
                fig, axes = plt.subplots(nrows=2, figsize=(12, 7), sharex=True)
                ax = axes[0]
                zscore_ax = axes[1]
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                zscore_ax = None

            for base_label in sorted(base_freqs.keys()):
                normalized_base = base_label if base_label in {"A", "C", "G", "T"} else "OTHER"
                color = DNA_5COLOR_PALETTE.get(normalized_base, DNA_5COLOR_PALETTE["OTHER"])
                ax.scatter(positions, base_freqs[base_label], label=base_label, color=color, linewidth=1)

            ax.plot(positions, normalized_mean_errors, label="Mean error rate", color="black", linestyle="--")
            # ax.fill_between(
            #     positions,
            #     np.full_like(positions, lower, dtype=float),
            #     np.full_like(positions, upper, dtype=float),
            #     color="black",
            #     alpha=0.12,
            #     label="±1 std error",
            # )

            ax.set_yscale("log")
            ax.set_xlabel("Position")
            ax.set_ylabel("Mismatch frequency")
            ax.set_title(f"{sample} - {ref} mismatch base frequencies")
            ax.legend(title="Mismatch base", ncol=4, fontsize=9)

            if plot_zscores and zscore_ax is not None and zscore_freqs:
                for base_label in sorted(zscore_freqs.keys()):
                    normalized_base = base_label if base_label in {"A", "C", "G", "T"} else "OTHER"
                    color = DNA_5COLOR_PALETTE.get(normalized_base, DNA_5COLOR_PALETTE["OTHER"])
                    zscore_ax.scatter(positions, zscore_freqs[base_label], label=base_label, color=color)
                zscore_ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
                zscore_ax.set_xlabel("Position")
                zscore_ax.set_ylabel("Z-score")
                zscore_ax.set_title(f"{sample} - {ref} quality-normalized mismatch z-scores")
                zscore_ax.legend(title="Mismatch base", ncol=4, fontsize=9)
            fig.tight_layout()

            out_file = None
            if save_path is not None:
                safe_name = f"{ref}__{sample}__mismatch_base_frequency".replace("=", "").replace(
                    ",", "_"
                )
                out_file = save_path / f"{safe_name}.png"
                fig.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved mismatch base frequency plot to %s.", out_file)
            else:
                plt.show()

            results.append(
                {
                    "reference": str(ref),
                    "sample": str(sample),
                    "n_positions": int(positions.size),
                    "quality_layer": quality_layer if plot_zscores else None,
                    "output_path": str(out_file) if out_file is not None else None,
                }
            )

        if save_path is not None and not summary_df.empty:
            safe_ref = f"{ref_name}__mismatch_base_frequency_summary".replace("=", "").replace(
                ",", "_"
            )
            summary_file = save_path / f"{safe_ref}.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info("Saved mismatch base frequency summary to %s.", summary_file)

    return results


def plot_sequence_integer_encoding_clustermaps(
    adata,
    sample_col: str = "Sample_Names",
    reference_col: str = "Reference_strand",
    layer: str = "sequence_integer_encoding",
    mismatch_layer: str = "mismatch_integer_encoding",
    exclude_mod_sites: bool = False,
    mod_site_bases: Sequence[str] | None = None,
    min_quality: float | None = 20,
    min_length: int | None = 200,
    min_mapped_length_to_reference_length_ratio: float | None = 0,
    demux_types: Sequence[str] = ("single", "double", "already"),
    sort_by: str = "none",  # "none", "hierarchical", "obs:<col>"
    cmap: str = "viridis",
    max_unknown_fraction: float | None = None,
    unknown_values: Sequence[int] = (4, 5),
    xtick_step: int | None = None,
    xtick_rotation: int = 90,
    xtick_fontsize: int = 9,
    max_reads: int | None = None,
    save_path: str | Path | None = None,
    use_dna_5color_palette: bool = True,
    show_numeric_colorbar: bool = False,
    show_position_axis: bool = False,
    position_axis_tick_target: int = 25,
):
    """Plot integer-encoded sequence clustermaps per sample/reference.

    Args:
        adata: AnnData with a ``sequence_integer_encoding`` layer.
        sample_col: Column in ``adata.obs`` that identifies samples.
        reference_col: Column in ``adata.obs`` that identifies references.
        layer: Layer name containing integer-encoded sequences.
        mismatch_layer: Optional layer name containing mismatch integer encodings.
        exclude_mod_sites: Whether to exclude annotated modification sites.
        mod_site_bases: Base-context labels used to build mod-site masks (e.g., ``["GpC", "CpG"]``).
        min_quality: Optional minimum read quality filter.
        min_length: Optional minimum mapped length filter.
        min_mapped_length_to_reference_length_ratio: Optional min length ratio filter.
        demux_types: Allowed ``demux_type`` values, if present in ``adata.obs``.
        sort_by: Row sorting strategy: ``none``, ``hierarchical``, or ``obs:<col>``.
        cmap: Matplotlib colormap for the heatmap when ``use_dna_5color_palette`` is False.
        max_unknown_fraction: Optional maximum fraction of ``unknown_values`` allowed per
            position; positions above this threshold are excluded.
        unknown_values: Integer values to treat as unknown/padding.
        xtick_step: Spacing between x-axis tick labels (None = no labels).
        xtick_rotation: Rotation for x-axis tick labels.
        xtick_fontsize: Font size for x-axis tick labels.
        max_reads: Optional maximum number of reads to plot per sample/reference.
        save_path: Optional output directory for saving plots.
        use_dna_5color_palette: Whether to use a fixed A/C/G/T/Other palette.
        show_numeric_colorbar: If False, use a legend instead of a numeric colorbar.
        show_position_axis: Whether to draw a position axis with tick labels.
        position_axis_tick_target: Approximate number of ticks to show when auto-sizing.

    Returns:
        List of dictionaries with per-plot metadata and output paths.
    """
    logger.info("Plotting sequence integer encoding clustermaps.")

    def _mask_or_true(series_name: str, predicate):
        if series_name not in adata.obs:
            return pd.Series(True, index=adata.obs.index)
        s = adata.obs[series_name]
        try:
            return predicate(s)
        except Exception:
            return pd.Series(True, index=adata.obs.index)

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    if max_unknown_fraction is not None and not (0 <= max_unknown_fraction <= 1):
        raise ValueError("max_unknown_fraction must be between 0 and 1.")

    if position_axis_tick_target < 1:
        raise ValueError("position_axis_tick_target must be at least 1.")

    results: List[Dict[str, Any]] = []
    save_path = Path(save_path) if save_path is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for col in (sample_col, reference_col):
        if col not in adata.obs:
            raise KeyError(f"{col} not in adata.obs")
        if not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    int_to_base = adata.uns.get("sequence_integer_decoding_map", {}) or {}
    if not int_to_base:
        encoding_map = adata.uns.get("sequence_integer_encoding_map", {}) or {}
        int_to_base = {int(v): str(k) for k, v in encoding_map.items()} if encoding_map else {}

    coerced_int_to_base = {}
    for key, value in int_to_base.items():
        try:
            coerced_key = int(key)
        except Exception:
            continue
        coerced_int_to_base[coerced_key] = str(value)
    int_to_base = coerced_int_to_base

    def normalize_base(base: str) -> str:
        return base if base in {"A", "C", "G", "T"} else "OTHER"

    mismatch_int_to_base = {}
    if mismatch_layer in adata.layers:
        mismatch_encoding_map = adata.uns.get("mismatch_integer_encoding_map", {}) or {}
        mismatch_int_to_base = {
            int(v): str(k)
            for k, v in mismatch_encoding_map.items()
            if isinstance(v, (int, np.integer))
        }

    def _resolve_xtick_step(n_positions: int) -> int | None:
        if xtick_step is not None:
            return xtick_step
        if not show_position_axis:
            return None
        return max(1, int(np.ceil(n_positions / position_axis_tick_target)))

    def _build_mod_site_mask(var_frame, ref_name: str) -> np.ndarray | None:
        if not exclude_mod_sites or not mod_site_bases:
            return None
        
        if hasattr(var_frame, "var"):
            var_frame = var_frame.var

        mod_site_cols = [f"{ref_name}_{base}_site" for base in mod_site_bases]
        missing_required = [col for col in mod_site_cols if col not in var_frame.columns]
        if missing_required:
            return None

        extra_cols = []
        if any(base in {"GpC", "CpG"} for base in mod_site_bases):
            ambiguous_col = f"{ref_name}_ambiguous_GpC_CpG_site"
            if ambiguous_col in var_frame.columns:
                extra_cols.append(ambiguous_col)

        mod_site_cols.extend(extra_cols)
        mod_site_cols = list(dict.fromkeys(mod_site_cols))

        mod_masks = [np.asarray(var_frame[col].values, dtype=bool) for col in mod_site_cols]
        mod_mask = mod_masks[0] if len(mod_masks) == 1 else np.logical_or.reduce(mod_masks)

        position_col = f"position_in_{ref_name}"
        if position_col in var_frame.columns:
            position_mask = np.asarray(var_frame[position_col].values, dtype=bool)
            mod_mask = np.logical_and(mod_mask, position_mask)

        return mod_mask

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            qmask = _mask_or_true(
                "read_quality",
                (lambda s: s >= float(min_quality))
                if (min_quality is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lm_mask = _mask_or_true(
                "mapped_length",
                (lambda s: s >= float(min_length))
                if (min_length is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            lrr_mask = _mask_or_true(
                "mapped_length_to_reference_length_ratio",
                (lambda s: s >= float(min_mapped_length_to_reference_length_ratio))
                if (min_mapped_length_to_reference_length_ratio is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )
            demux_mask = _mask_or_true(
                "demux_type",
                (lambda s: s.astype("string").isin(list(demux_types)))
                if (demux_types is not None)
                else (lambda s: pd.Series(True, index=s.index)),
            )

            row_mask = (
                (adata.obs[reference_col] == ref)
                & (adata.obs[sample_col] == sample)
                & qmask
                & lm_mask
                & lrr_mask
                & demux_mask
            )
            if not bool(row_mask.any()):
                continue

            subset = adata[row_mask, :].copy()
            matrix = np.asarray(subset.layers[layer])
            mismatch_matrix = None
            if mismatch_layer in subset.layers:
                mismatch_matrix = np.asarray(subset.layers[mismatch_layer])

            mod_site_mask = _build_mod_site_mask(subset, str(ref))
            if mod_site_mask is not None:
                keep_columns = ~mod_site_mask
                if not np.any(keep_columns):
                    continue
                matrix = matrix[:, keep_columns]
                subset = subset[:, keep_columns].copy()
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[:, keep_columns]

            if max_unknown_fraction is not None:
                unknown_mask = np.isin(matrix, np.asarray(unknown_values))
                unknown_fraction = unknown_mask.mean(axis=0)
                keep_columns = unknown_fraction <= max_unknown_fraction
                if not np.any(keep_columns):
                    continue
                matrix = matrix[:, keep_columns]
                subset = subset[:, keep_columns].copy()
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[:, keep_columns]

            if max_reads is not None and matrix.shape[0] > max_reads:
                matrix = matrix[:max_reads]
                subset = subset[:max_reads, :].copy()
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[:max_reads]

            if matrix.size == 0:
                continue

            if use_dna_5color_palette and not int_to_base:
                uniq_vals = np.unique(matrix[~pd.isna(matrix)])
                guess = {}
                for val in uniq_vals:
                    try:
                        int_val = int(val)
                    except Exception:
                        continue
                    guess[int_val] = {0: "A", 1: "C", 2: "G", 3: "T"}.get(int_val, "OTHER")
                int_to_base_local = guess
            else:
                int_to_base_local = int_to_base

            order = None
            if sort_by.startswith("obs:"):
                colname = sort_by.split("obs:")[1]
                order = np.argsort(subset.obs[colname].values)
            elif sort_by == "hierarchical":
                linkage = sch.linkage(np.nan_to_num(matrix), method="ward")
                order = sch.leaves_list(linkage)
            elif sort_by != "none":
                raise ValueError("sort_by must be 'none', 'hierarchical', or 'obs:<col>'")

            if order is not None:
                matrix = matrix[order]
                if mismatch_matrix is not None:
                    mismatch_matrix = mismatch_matrix[order]

            has_mismatch = mismatch_matrix is not None
            fig, axes = plt.subplots(
                ncols=2 if has_mismatch else 1,
                figsize=(18, 6) if has_mismatch else (12, 6),
                sharey=has_mismatch,
            )
            if not isinstance(axes, np.ndarray):
                axes = np.asarray([axes])
            ax = axes[0]

            if use_dna_5color_palette and int_to_base_local:
                int_to_color = {
                    int(int_val): DNA_5COLOR_PALETTE[normalize_base(str(base))]
                    for int_val, base in int_to_base_local.items()
                }
                uniq_matrix = np.unique(matrix[~pd.isna(matrix)])
                for val in uniq_matrix:
                    try:
                        int_val = int(val)
                    except Exception:
                        continue
                    if int_val not in int_to_color:
                        int_to_color[int_val] = DNA_5COLOR_PALETTE["OTHER"]

                ordered = sorted(int_to_color.items(), key=lambda x: x[0])
                colors_list = [color for _, color in ordered]
                bounds = [int_val - 0.5 for int_val, _ in ordered]
                bounds.append(ordered[-1][0] + 0.5)

                cmap_obj = colors.ListedColormap(colors_list)
                norm = colors.BoundaryNorm(bounds, cmap_obj.N)

                sns.heatmap(
                    matrix,
                    cmap=cmap_obj,
                    norm=norm,
                    ax=ax,
                    yticklabels=False,
                    cbar=show_numeric_colorbar,
                )

                legend_handles = [
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["A"], label="A"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["C"], label="C"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["G"], label="G"),
                    patches.Patch(facecolor=DNA_5COLOR_PALETTE["T"], label="T"),
                    patches.Patch(
                        facecolor=DNA_5COLOR_PALETTE["OTHER"],
                        label="Other (N / PAD / unknown)",
                    ),
                ]
                ax.legend(
                    handles=legend_handles,
                    title="Base",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                )
            else:
                sns.heatmap(matrix, cmap=cmap, ax=ax, yticklabels=False, cbar=True)

            ax.set_title(layer)

            resolved_step = _resolve_xtick_step(matrix.shape[1])
            if resolved_step is not None and resolved_step > 0:
                sites = np.arange(0, matrix.shape[1], resolved_step)
                ax.set_xticks(sites)
                ax.set_xticklabels(
                    subset.var_names[sites].astype(str),
                    rotation=xtick_rotation,
                    fontsize=xtick_fontsize,
                )
            else:
                ax.set_xticks([])
            if show_position_axis or xtick_step is not None:
                ax.set_xlabel("Position")

            if has_mismatch:
                mismatch_ax = axes[1]
                mismatch_int_to_base_local = mismatch_int_to_base or int_to_base_local
                if use_dna_5color_palette and mismatch_int_to_base_local:
                    mismatch_int_to_color = {}
                    for int_val, base in mismatch_int_to_base_local.items():
                        base_upper = str(base).upper()
                        if base_upper == "PAD":
                            mismatch_int_to_color[int(int_val)] = "#D3D3D3"
                        elif base_upper == "N":
                            mismatch_int_to_color[int(int_val)] = "#808080"
                        else:
                            mismatch_int_to_color[int(int_val)] = DNA_5COLOR_PALETTE[
                                normalize_base(base_upper)
                            ]

                    uniq_mismatch = np.unique(mismatch_matrix[~pd.isna(mismatch_matrix)])
                    for val in uniq_mismatch:
                        try:
                            int_val = int(val)
                        except Exception:
                            continue
                        if int_val not in mismatch_int_to_color:
                            mismatch_int_to_color[int_val] = DNA_5COLOR_PALETTE["OTHER"]

                    ordered_mismatch = sorted(mismatch_int_to_color.items(), key=lambda x: x[0])
                    mismatch_colors = [color for _, color in ordered_mismatch]
                    mismatch_bounds = [int_val - 0.5 for int_val, _ in ordered_mismatch]
                    mismatch_bounds.append(ordered_mismatch[-1][0] + 0.5)

                    mismatch_cmap = colors.ListedColormap(mismatch_colors)
                    mismatch_norm = colors.BoundaryNorm(mismatch_bounds, mismatch_cmap.N)

                    sns.heatmap(
                        mismatch_matrix,
                        cmap=mismatch_cmap,
                        norm=mismatch_norm,
                        ax=mismatch_ax,
                        yticklabels=False,
                        cbar=show_numeric_colorbar,
                    )

                    mismatch_legend_handles = [
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["A"], label="A"),
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["C"], label="C"),
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["G"], label="G"),
                        patches.Patch(facecolor=DNA_5COLOR_PALETTE["T"], label="T"),
                        patches.Patch(facecolor="#808080", label="Match/N"),
                        patches.Patch(facecolor="#D3D3D3", label="PAD"),
                    ]
                    mismatch_ax.legend(
                        handles=mismatch_legend_handles,
                        title="Mismatch base",
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        frameon=False,
                    )
                else:
                    sns.heatmap(
                        mismatch_matrix,
                        cmap=cmap,
                        ax=mismatch_ax,
                        yticklabels=False,
                        cbar=True,
                    )

                mismatch_ax.set_title(mismatch_layer)
                if resolved_step is not None and resolved_step > 0:
                    sites = np.arange(0, mismatch_matrix.shape[1], resolved_step)
                    mismatch_ax.set_xticks(sites)
                    mismatch_ax.set_xticklabels(
                        subset.var_names[sites].astype(str),
                        rotation=xtick_rotation,
                        fontsize=xtick_fontsize,
                    )
                else:
                    mismatch_ax.set_xticks([])
                if show_position_axis or xtick_step is not None:
                    mismatch_ax.set_xlabel("Position")

            n_reads = matrix.shape[0]

            fig.suptitle(f"{sample} - {ref} - {n_reads} reads")
            fig.tight_layout(rect=(0, 0, 1, 0.95))

            out_file = None
            if save_path is not None:
                safe_name = f"{ref}__{sample}__{layer}".replace("=", "").replace(",", "_")
                out_file = save_path / f"{safe_name}.png"
                fig.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved sequence encoding clustermap to %s.", out_file)
            else:
                plt.show()

            results.append(
                {
                    "reference": str(ref),
                    "sample": str(sample),
                    "layer": layer,
                    "n_positions": int(matrix.shape[1]),
                    "mismatch_layer": mismatch_layer if has_mismatch else None,
                    "mismatch_layer_present": bool(has_mismatch),
                    "output_path": str(out_file) if out_file is not None else None,
                }
            )

    return results

