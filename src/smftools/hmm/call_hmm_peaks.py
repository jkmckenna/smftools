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
    Call peaks on one or more HMM-derived (or other) layers and annotate adata.var / adata.obs,
    doing peak calling *within each reference subset*.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with layers already containing feature tracks (e.g. HMM-derived masks).
    feature_configs : dict
        Mapping: feature_type_or_layer_suffix -> {
            "min_distance": int (default 200),
            "peak_width":   int (default 200),
            "peak_prominence": float (default 0.2),
            "peak_threshold":  float (default 0.8),
        }

        Keys are usually *feature types* like "all_accessible_features" or
        "small_bound_stretch". These are matched against existing HMM layers
        (e.g. "GpC_all_accessible_features", "Combined_small_bound_stretch")
        using a suffix match. You can also pass full layer names if you wish.
    ref_column : str
        Column in adata.obs defining reference groups (e.g. "Reference_strand").
    site_types : sequence of str
        Site types (without "_site"); expects var columns like f"{ref}_{site_type}_site".
        e.g. ("GpC", "CpG") -> "6B6_top_GpC_site", etc.
    save_plot : bool
        If True, save peak diagnostic plots instead of just showing them.
    output_dir : path-like or None
        Directory for saved plots (created if needed).
    date_tag : str or None
        Optional tag to prefix plot filenames.
    inplace : bool
        If False, operate on a copy and return it. If True, modify adata and return None.
    index_col_suffix : str or None
        If None, coordinates come from adata.var_names (cast to int when possible).
        If set, for each ref we use adata.var[f"{ref}_{index_col_suffix}"] as the
        coordinate system (e.g. a reindexed coordinate).

    Returns
    -------
    None or AnnData
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.sparse import issparse

    if not inplace:
        adata = adata.copy()

    # Ensure ref_column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[ref_column]):
        adata.obs[ref_column] = adata.obs[ref_column].astype("category")

    # Base coordinates (fallback)
    try:
        base_coordinates = adata.var_names.astype(int).values
    except Exception:
        base_coordinates = np.arange(adata.n_vars, dtype=int)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # HMM layers known to the object (if present)
    hmm_layers = list(adata.uns.get("hmm_appended_layers", [])) or []
    # keep only the binary masks, not *_lengths
    hmm_layers = [layer for layer in hmm_layers if not layer.endswith("_lengths")]

    # Fallback: use all layer names if hmm_appended_layers is empty/missing
    all_layer_names = list(adata.layers.keys())

    all_peak_var_cols = []

    # Iterate over each reference separately
    for ref in adata.obs[ref_column].cat.categories:
        ref_mask = (adata.obs[ref_column] == ref).values
        if not ref_mask.any():
            continue

        # Per-ref coordinates: either from a reindexed column or global fallback
        if index_col_suffix is not None:
            coord_col = f"{ref}_{index_col_suffix}"
            if coord_col not in adata.var:
                raise KeyError(
                    f"index_col_suffix='{index_col_suffix}' requested, "
                    f"but var column '{coord_col}' is missing for ref '{ref}'."
                )
            coord_vals = adata.var[coord_col].values
            # Try to coerce to numeric
            try:
                coordinates = coord_vals.astype(int)
            except Exception:
                coordinates = np.asarray(coord_vals, dtype=float)
        else:
            coordinates = base_coordinates

        # Resolve each feature_config key to one or more actual layer names
        for feature_key, config in feature_configs.items():
            # Candidate search space: HMM layers if present, else all layers
            search_layers = hmm_layers if hmm_layers else all_layer_names

            candidate_layers = []

            # First: exact match
            for lname in search_layers:
                if lname == feature_key:
                    candidate_layers.append(lname)

            # Second: suffix match (e.g. "all_accessible_features" ->
            # "GpC_all_accessible_features", "Combined_all_accessible_features", etc.)
            if not candidate_layers:
                for lname in search_layers:
                    if lname.endswith(feature_key):
                        candidate_layers.append(lname)

            # Third: if user passed a full layer name that wasn't in hmm_layers,
            # but does exist in adata.layers, allow it.
            if not candidate_layers and feature_key in adata.layers:
                candidate_layers.append(feature_key)

            if not candidate_layers:
                print(
                    f"[call_hmm_peaks] WARNING: no layers found matching feature key "
                    f"'{feature_key}' in ref '{ref}'. Skipping."
                )
                continue

            # Run peak calling on each resolved layer for this ref
            for layer_name in candidate_layers:
                if layer_name not in adata.layers:
                    print(
                        f"[call_hmm_peaks] WARNING: resolved layer '{layer_name}' "
                        f"not found in adata.layers; skipping."
                    )
                    continue

                min_distance = int(config.get("min_distance", 200))
                peak_width = int(config.get("peak_width", 200))
                peak_prominence = float(config.get("peak_prominence", 0.2))
                peak_threshold = float(config.get("peak_threshold", 0.8))

                layer_data = adata.layers[layer_name]
                if issparse(layer_data):
                    layer_data = layer_data.toarray()
                else:
                    layer_data = np.asarray(layer_data)

                # Subset rows for this ref
                matrix = layer_data[ref_mask, :]  # (n_ref_reads, n_vars)
                if matrix.shape[0] == 0:
                    continue

                # Mean signal along positions (within this ref only)
                means = np.nanmean(matrix, axis=0)

                # Optional rolling-mean smoothing before peak detection
                rolling_window = int(config.get("rolling_window", 1))
                if rolling_window > 1:
                    # Simple centered rolling mean via convolution
                    kernel = np.ones(rolling_window, dtype=float) / float(rolling_window)
                    smoothed = np.convolve(means, kernel, mode="same")
                    peak_metric = smoothed
                else:
                    peak_metric = means

                # Peak detection
                peak_indices, _ = find_peaks(
                    peak_metric, prominence=peak_prominence, distance=min_distance
                )
                if peak_indices.size == 0:
                    print(
                        f"[call_hmm_peaks] No peaks found for layer '{layer_name}' "
                        f"in ref '{ref}'."
                    )
                    continue

                peak_centers = coordinates[peak_indices]
                # Store per-ref peak centers
                adata.uns[f"{layer_name}_{ref}_peak_centers"] = peak_centers.tolist()

                # ---- Plot ----
                plt.figure(figsize=(6, 3))
                plt.plot(coordinates, peak_metric, linewidth=1)
                plt.title(f"{layer_name} peaks in {ref}")
                plt.xlabel("Coordinate")
                plt.ylabel(f"Rolling Mean - roll size {rolling_window}")

                for i, center in enumerate(peak_centers):
                    start = center - peak_width // 2
                    end = center + peak_width // 2
                    height = peak_metric[peak_indices[i]]
                    plt.axvspan(start, end, color="purple", alpha=0.2)
                    plt.axvline(center, color="red", linestyle="--", linewidth=0.8)

                    # alternate label placement a bit left/right
                    if alternate_labels:
                        if i % 2 == 0:
                            x_text, ha = start, "right"
                        else:
                            x_text, ha = end, "left"
                    else:
                        x_text, ha = start, "right"
                        
                    plt.text(
                        x_text,
                        height * 0.8,
                        f"Peak {i}\n{center}",
                        color="red",
                        ha=ha,
                        va="bottom",
                        fontsize=8,
                    )

                if save_plot and output_dir is not None:
                    tag = date_tag or "output"
                    # include ref in filename
                    safe_ref = str(ref).replace("/", "_")
                    safe_layer = str(layer_name).replace("/", "_")
                    fname = output_dir / f"{tag}_{safe_layer}_{safe_ref}_peaks.png"
                    plt.savefig(fname, bbox_inches="tight", dpi=200)
                    print(f"[call_hmm_peaks] Saved plot to {fname}")
                    plt.close()
                else:
                    plt.tight_layout()
                    plt.show()

                feature_peak_cols = []

                # ---- Per-peak annotations (within this ref) ----
                for center in peak_centers:
                    start = center - peak_width // 2
                    end = center + peak_width // 2

                    # Make column names ref- and layer-specific so they don't collide
                    colname = f"{layer_name}_{ref}_peak_{center}"
                    feature_peak_cols.append(colname)
                    all_peak_var_cols.append(colname)

                    # Var-level mask: is this position in the window?
                    peak_mask = (coordinates >= start) & (coordinates <= end)
                    adata.var[colname] = peak_mask

                    # Extract signal in that window from the *ref subset* matrix
                    region = matrix[:, peak_mask]  # (n_ref_reads, n_positions_in_window)

                    # Per-read summary in this window for the feature layer itself
                    mean_col = f"mean_{layer_name}_{ref}_around_{center}"
                    sum_col = f"sum_{layer_name}_{ref}_around_{center}"
                    present_col = f"{layer_name}_{ref}_present_at_{center}"

                    # Create columns if missing, then fill only the ref rows
                    if mean_col not in adata.obs:
                        adata.obs[mean_col] = np.nan
                    if sum_col not in adata.obs:
                        adata.obs[sum_col] = 0.0
                    if present_col not in adata.obs:
                        adata.obs[present_col] = False

                    adata.obs.loc[ref_mask, mean_col] = np.nanmean(region, axis=1)
                    adata.obs.loc[ref_mask, sum_col] = np.nansum(region, axis=1)
                    adata.obs.loc[ref_mask, present_col] = (
                        adata.obs.loc[ref_mask, mean_col].values > peak_threshold
                    )

                    # Initialize site-type summaries (global columns; filled per ref)
                    for site_type in site_types:
                        sum_site_col = f"{site_type}_{ref}_sum_around_{center}"
                        mean_site_col = f"{site_type}_{ref}_mean_around_{center}"
                        if sum_site_col not in adata.obs:
                            adata.obs[sum_site_col] = 0.0
                        if mean_site_col not in adata.obs:
                            adata.obs[mean_site_col] = np.nan

                    # Per-site-type summaries for this ref
                    for site_type in site_types:
                        mask_key = f"{ref}_{site_type}_site"
                        if mask_key not in adata.var:
                            continue

                        site_mask = adata.var[mask_key].values.astype(bool)
                        if not site_mask.any():
                            continue

                        site_coords = coordinates[site_mask]
                        region_mask = (site_coords >= start) & (site_coords <= end)
                        if not region_mask.any():
                            continue

                        full_mask = np.zeros_like(site_mask, dtype=bool)
                        full_mask[site_mask] = region_mask

                        site_region = adata[ref_mask, full_mask].X
                        if hasattr(site_region, "A"):
                            site_region = site_region.A  # sparse -> dense

                        if site_region.shape[1] == 0:
                            continue

                        sum_site_col = f"{site_type}_{ref}_sum_around_{center}"
                        mean_site_col = f"{site_type}_{ref}_mean_around_{center}"

                        adata.obs.loc[ref_mask, sum_site_col] = np.nansum(site_region, axis=1)
                        adata.obs.loc[ref_mask, mean_site_col] = np.nanmean(site_region, axis=1)

                # Mark "any peak" for this (layer, ref)
                any_col = f"is_in_any_{layer_name}_peak_{ref}"
                adata.var[any_col] = adata.var[feature_peak_cols].any(axis=1)
                print(
                    f"[call_hmm_peaks] Annotated {len(peak_centers)} peaks "
                    f"for layer '{layer_name}' in ref '{ref}'."
                )

    # Global any-peak flag across all feature layers and references
    if all_peak_var_cols:
        adata.var["is_in_any_peak"] = adata.var[all_peak_var_cols].any(axis=1)

    return None if inplace else adata
