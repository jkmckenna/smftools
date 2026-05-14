from __future__ import annotations

# FILE: smftools/hmm/call_hmm_peaks.py
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

logger = get_logger(__name__)


def call_hmm_peaks(
    adata,
    feature_configs: Dict[str, Dict[str, Any]],
    ref_column: str = "Reference_strand",
    site_types: Sequence[str] = ("GpC", "CpG"),
    save_plot: bool = False,
    output_dir: Optional[Union[str, "Path"]] = None,
    date_tag: Optional[str] = None,
    inplace: bool = True,
    index_col_suffix: Optional[str] = None,
    alternate_labels: bool = False,
):
    """
    Peak calling over HMM (or other) layers, per reference group and per layer.
    Writes:
      - adata.uns["{layer}_{ref}_peak_centers"] = list of centers
      - adata.var["{layer}_{ref}_peak_{center}"] boolean window masks
      - adata.var["is_in_any_{layer}_peak_{ref}"]
      - adata.var["is_in_any_peak"] (global)
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks
    from scipy.sparse import issparse

    plt = require("matplotlib.pyplot", extra="plotting", purpose="HMM peak plots")

    if not inplace:
        adata = adata.copy()

    if ref_column not in adata.obs:
        raise KeyError(f"obs column '{ref_column}' not found")

    # Ensure categorical for predictable ref iteration
    if not isinstance(adata.obs[ref_column].dtype, pd.CategoricalDtype):
        adata.obs[ref_column] = adata.obs[ref_column].astype("category")

    # Optional: drop duplicate obs columns once to avoid Pandas/AnnData view quirks
    if getattr(adata.obs.columns, "duplicated", None) is not None:
        if adata.obs.columns.duplicated().any():
            adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated(keep="first")].copy()

    # Fallback coordinates from var_names
    try:
        base_coordinates = adata.var_names.astype(int).values
    except Exception:
        base_coordinates = np.arange(adata.n_vars, dtype=int)

    # Output dir
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build search pool = union of declared HMM layers and actual layers; exclude helper suffixes
    raw = adata.uns.get("hmm_appended_layers", [])
    declared = list(raw) if len(raw) > 0 else []
    search_pool = [
        layer
        for layer in declared
        if not any(s in layer for s in ("_lengths", "_states", "_posterior"))
    ]

    all_peak_var_cols = []

    # Iterate per reference
    for ref in adata.obs[ref_column].cat.categories:
        ref_mask = (adata.obs[ref_column] == ref).values
        if not ref_mask.any():
            continue

        # Per-ref coordinate system
        if index_col_suffix is not None:
            coord_col = f"{ref}_{index_col_suffix}"
            if coord_col not in adata.var:
                raise KeyError(
                    f"index_col_suffix='{index_col_suffix}' requested, missing var column '{coord_col}' for ref '{ref}'."
                )
            coord_vals = adata.var[coord_col].values
            try:
                coordinates = coord_vals.astype(int)
            except Exception:
                coordinates = np.asarray(coord_vals, dtype=float)
        else:
            coordinates = base_coordinates

        if coordinates.shape[0] != adata.n_vars:
            raise ValueError(f"Coordinate length {coordinates.shape[0]} != n_vars {adata.n_vars}")

        # Feature keys to consider
        for feature_key, config in feature_configs.items():
            # Resolve candidate layers: exact → suffix → direct present
            candidates = [ln for ln in search_pool if ln == feature_key]
            if not candidates:
                candidates = [ln for ln in search_pool if str(ln).endswith(feature_key)]
            if not candidates and feature_key in adata.layers:
                candidates = [feature_key]

            if not candidates:
                logger.warning(
                    "[call_hmm_peaks] No layers found matching '%s' in ref '%s'. Skipping.",
                    feature_key,
                    ref,
                )
                continue

            # Hyperparams (sanitized)
            min_distance = max(1, int(config.get("min_distance", 200)))
            peak_width = max(1, int(config.get("peak_width", 200)))
            peak_prom = float(config.get("peak_prominence", 0.2))
            rolling_window = max(1, int(config.get("rolling_window", 1)))

            for layer_name in candidates:
                if layer_name not in adata.layers:
                    logger.warning(
                        "[call_hmm_peaks] Layer '%s' not in adata.layers; skipping.",
                        layer_name,
                    )
                    continue

                # Dense layer data
                L = adata.layers[layer_name]
                L = L.toarray() if issparse(L) else np.asarray(L)
                if L.shape != (adata.n_obs, adata.n_vars):
                    logger.warning(
                        "[call_hmm_peaks] Layer '%s' has shape %s, expected (%s, %s); skipping.",
                        layer_name,
                        L.shape,
                        adata.n_obs,
                        adata.n_vars,
                    )
                    continue

                # Ref subset
                matrix = L[ref_mask, :]
                if matrix.size == 0 or matrix.shape[0] == 0:
                    continue

                means = np.nanmean(matrix, axis=0)
                means = np.nan_to_num(means, nan=0.0)

                if rolling_window > 1:
                    kernel = np.ones(rolling_window, dtype=float) / float(rolling_window)
                    peak_metric = np.convolve(means, kernel, mode="same")
                else:
                    peak_metric = means

                # Peak detection
                peak_indices, _ = find_peaks(
                    peak_metric, prominence=peak_prom, distance=min_distance
                )
                if peak_indices.size == 0:
                    logger.info(
                        "[call_hmm_peaks] No peaks for layer '%s' in ref '%s'.",
                        layer_name,
                        ref,
                    )
                    continue

                peak_centers = coordinates[peak_indices]
                adata.uns[f"{layer_name}_{ref}_peak_centers"] = peak_centers.tolist()

                # Plot once per layer/ref
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(coordinates, peak_metric, linewidth=1)
                ax.set_title(f"{layer_name} peaks in {ref}")
                ax.set_xlabel("Coordinate")
                ax.set_ylabel(f"Rolling Mean (win={rolling_window})")
                for i, center in enumerate(peak_centers):
                    start = center - peak_width // 2
                    end = center + peak_width // 2
                    height = peak_metric[peak_indices[i]]
                    ax.axvspan(start, end, alpha=0.2)
                    ax.axvline(center, linestyle="--", linewidth=0.8)
                    x_text, ha = (
                        (start, "right") if (not alternate_labels or i % 2 == 0) else (end, "left")
                    )
                    ax.text(
                        x_text, height * 0.8, f"Peak {i}\n{center}", ha=ha, va="bottom", fontsize=8
                    )

                if save_plot and output_dir is not None:
                    tag = date_tag or "output"
                    safe_ref = str(ref).replace("/", "_")
                    safe_layer = str(layer_name).replace("/", "_")
                    fname = output_dir / f"{tag}_{safe_layer}_{safe_ref}_peaks.png"
                    fig.savefig(fname, bbox_inches="tight", dpi=200)
                    logger.info("[call_hmm_peaks] Saved plot to %s", fname)
                    plt.close(fig)
                else:
                    fig.tight_layout()
                    plt.show()

                feature_peak_cols = []

                for center in np.asarray(peak_centers).tolist():
                    start = center - peak_width // 2
                    end = center + peak_width // 2

                    colname = f"{layer_name}_{ref}_peak_{center}"
                    feature_peak_cols.append(colname)
                    all_peak_var_cols.append(colname)
                    peak_mask = (coordinates >= start) & (coordinates <= end)
                    adata.var[colname] = peak_mask

                # per (layer, ref) any-peak
                any_col = f"is_in_any_{layer_name}_peak_{ref}"
                if feature_peak_cols:
                    adata.var[any_col] = adata.var[feature_peak_cols].any(axis=1)
                else:
                    adata.var[any_col] = False

                logger.info(
                    "[call_hmm_peaks] Annotated %s peaks for layer '%s' in ref '%s'.",
                    len(peak_centers),
                    layer_name,
                    ref,
                )

    # global any-peak across all layers/refs
    if all_peak_var_cols:
        adata.var["is_in_any_peak"] = adata.var[all_peak_var_cols].any(axis=1)

    return None if inplace else adata
