# FILE: smftools/hmm/call_hmm_peaks.py

from typing import Dict, Optional, Any, Union, Sequence
from pathlib import Path

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
      - adata.obs per-read summaries for each peak window:
            mean_{layer}_{ref}_around_{center}
            sum_{layer}_{ref}_around_{center}
            {layer}_{ref}_present_at_{center} (bool)
        and per site-type:
            sum_{layer}_{site}_{ref}_around_{center}
            mean_{layer}_{site}_{ref}_around_{center}
      - adata.var["is_in_any_{layer}_peak_{ref}"]
      - adata.var["is_in_any_peak"] (global)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.sparse import issparse

    if not inplace:
        adata = adata.copy()

    if ref_column not in adata.obs:
        raise KeyError(f"obs column '{ref_column}' not found")

    # Ensure categorical for predictable ref iteration
    if not pd.api.types.is_categorical_dtype(adata.obs[ref_column]):
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
    declared = list(adata.uns.get("hmm_appended_layers", []) or [])
    search_pool = [l for l in declared if not any(s in l for s in ("_lengths", "_states", "_posterior"))]

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
                raise KeyError(f"index_col_suffix='{index_col_suffix}' requested, missing var column '{coord_col}' for ref '{ref}'.")
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
                print(f"[call_hmm_peaks] WARNING: no layers found matching '{feature_key}' in ref '{ref}'. Skipping.")
                continue

            # Hyperparams (sanitized)
            min_distance   = max(1, int(config.get("min_distance",   200)))
            peak_width     = max(1, int(config.get("peak_width",     200)))
            peak_prom      = float(config.get("peak_prominence", 0.2))
            peak_threshold = float(config.get("peak_threshold",  0.8))
            rolling_window = max(1, int(config.get("rolling_window", 1)))

            for layer_name in candidates:
                if layer_name not in adata.layers:
                    print(f"[call_hmm_peaks] WARNING: layer '{layer_name}' not in adata.layers; skipping.")
                    continue

                # Dense layer data
                L = adata.layers[layer_name]
                L = L.toarray() if issparse(L) else np.asarray(L)
                if L.shape != (adata.n_obs, adata.n_vars):
                    print(f"[call_hmm_peaks] WARNING: layer '{layer_name}' has shape {L.shape}, expected ({adata.n_obs}, {adata.n_vars}); skipping.")
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
                peak_indices, _ = find_peaks(peak_metric, prominence=peak_prom, distance=min_distance)
                if peak_indices.size == 0:
                    print(f"[call_hmm_peaks] No peaks for layer '{layer_name}' in ref '{ref}'.")
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
                    end   = center + peak_width // 2
                    height = peak_metric[peak_indices[i]]
                    ax.axvspan(start, end, alpha=0.2)
                    ax.axvline(center, linestyle="--", linewidth=0.8)
                    x_text, ha = ((start, "right") if (not alternate_labels or i % 2 == 0) else (end, "left"))
                    ax.text(x_text, height * 0.8, f"Peak {i}\n{center}", ha=ha, va="bottom", fontsize=8)

                if save_plot and output_dir is not None:
                    tag = date_tag or "output"
                    safe_ref = str(ref).replace("/", "_")
                    safe_layer = str(layer_name).replace("/", "_")
                    fname = output_dir / f"{tag}_{safe_layer}_{safe_ref}_peaks.png"
                    fig.savefig(fname, bbox_inches="tight", dpi=200)
                    print(f"[call_hmm_peaks] Saved plot to {fname}")
                    plt.close(fig)
                else:
                    fig.tight_layout()
                    plt.show()

                # Collect new obs columns; assign once per layer/ref
                new_obs_cols: Dict[str, np.ndarray] = {}
                feature_peak_cols = []

                for center in np.asarray(peak_centers).tolist():
                    start = center - peak_width // 2
                    end   = center + peak_width // 2

                    # var window mask
                    colname = f"{layer_name}_{ref}_peak_{center}"
                    feature_peak_cols.append(colname)
                    all_peak_var_cols.append(colname)
                    peak_mask = (coordinates >= start) & (coordinates <= end)
                    adata.var[colname] = peak_mask

                    # feature-layer summaries for reads in this ref
                    region = matrix[:, peak_mask]  # (n_ref, n_window)

                    mean_col    = f"mean_{layer_name}_{ref}_around_{center}"
                    sum_col     = f"sum_{layer_name}_{ref}_around_{center}"
                    present_col = f"{layer_name}_{ref}_present_at_{center}"

                    for nm, default, dt in (
                        (mean_col, np.nan, float),
                        (sum_col, 0.0, float),
                        (present_col, False, bool),
                    ):
                        if nm not in new_obs_cols:
                            new_obs_cols[nm] = np.full(adata.n_obs, default, dtype=dt)

                    if region.shape[1] > 0:
                        means_per_read = np.nanmean(region, axis=1)
                        sums_per_read  = np.nansum(region, axis=1)
                    else:
                        means_per_read = np.full(matrix.shape[0], np.nan, dtype=float)
                        sums_per_read  = np.zeros(matrix.shape[0], dtype=float)

                    new_obs_cols[mean_col][ref_mask]    = means_per_read
                    new_obs_cols[sum_col][ref_mask]     = sums_per_read
                    new_obs_cols[present_col][ref_mask] = np.nan_to_num(means_per_read, nan=0.0) > peak_threshold

                    # site-type summaries from adata.X, not an AnnData view
                    Xmat = adata.X
                    for site_type in site_types:
                        mask_key = f"{ref}_{site_type}_site"
                        if mask_key not in adata.var:
                            continue

                        site_mask = adata.var[mask_key].values.astype(bool)
                        if not site_mask.any():
                            continue

                        site_coords = coordinates[site_mask]
                        site_region_mask = (site_coords >= start) & (site_coords <= end)
                        sum_site_col  = f"sum_{layer_name}_{site_type}_{ref}_around_{center}"
                        mean_site_col = f"mean_{layer_name}_{site_type}_{ref}_around_{center}"

                        if sum_site_col not in new_obs_cols:
                            new_obs_cols[sum_site_col] = np.zeros(adata.n_obs, dtype=float)
                        if mean_site_col not in new_obs_cols:
                            new_obs_cols[mean_site_col] = np.full(adata.n_obs, np.nan, dtype=float)

                        if not site_region_mask.any():
                            continue

                        full_mask = np.zeros_like(site_mask, dtype=bool)
                        full_mask[site_mask] = site_region_mask

                        if issparse(Xmat):
                            site_region = Xmat[ref_mask][:, full_mask]
                            site_region = site_region.toarray()
                        else:
                            Xnp = np.asarray(Xmat)
                            site_region = Xnp[np.asarray(ref_mask), :][:, np.asarray(full_mask)]

                        if site_region.shape[1] > 0:
                            new_obs_cols[sum_site_col][ref_mask]  = np.nansum(site_region, axis=1)
                            new_obs_cols[mean_site_col][ref_mask] = np.nanmean(site_region, axis=1)

                # one-shot assignment to avoid fragmentation
                if new_obs_cols:
                    adata.obs = adata.obs.assign(**{k: pd.Series(v, index=adata.obs.index) for k, v in new_obs_cols.items()})

                # per (layer, ref) any-peak
                any_col = f"is_in_any_{layer_name}_peak_{ref}"
                if feature_peak_cols:
                    adata.var[any_col] = adata.var[feature_peak_cols].any(axis=1)
                else:
                    adata.var[any_col] = False

                print(f"[call_hmm_peaks] Annotated {len(peak_centers)} peaks for layer '{layer_name}' in ref '{ref}'.")

    # global any-peak across all layers/refs
    if all_peak_var_cols:
        adata.var["is_in_any_peak"] = adata.var[all_peak_var_cols].any(axis=1)

    return None if inplace else adata
