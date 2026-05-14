"""
hmm_histogram.py — Per-feature-type binning, rolling mean, and peak-calling
config for HMM feature histograms.

Structure
---------
HISTOGRAM_CONFIGS[feature_layer_name][hist_type] → dict or None

hist_type keys
--------------
  "count"             — features-per-read histogram
  "size"              — feature size histogram (bp)
  "neighbor_distance" — neighbor center-to-center distance histogram (bp)

Per-entry fields
----------------
  bin_size_bp        : int | None   bin width in bp; None → sqrt(n) bins
  rolling_window_bp  : int | None   smoothing window for centered rolling mean
  peak_kwargs        : dict | None  kwargs for scipy.signal.find_peaks on [0,1]-normalised curve
  rolling_color      : str          matplotlib color for rolling mean line
  peak_color         : str          matplotlib color for peak annotations
  gaussian_fit       : dict | None  if present, fit a Gaussian to a bp-range subset
                                    fields: fit_range_bp (tuple), fit_color (str)
"""

_NONE = None

HISTOGRAM_CONFIGS: dict[str, dict[str, dict | None]] = {

    "C_all_accessible_features_merged": {
        "count": _NONE,
        "size": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
    },

    "C_all_footprint_features": {
        "count": _NONE,
        "size": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
    },

    "C_nucleosome_depleted_region_merged": {
        "count": _NONE,
        "size": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
    },

    "C_putative_nucleosome": {
        "count": _NONE,
        "size": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp":       5,
            "rolling_window_bp": 20,
            "peak_kwargs":  {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color":    "#d62728",
            "gaussian_fit":  {"fit_range_bp": (0, 300), "fit_color": "#1f77b4"},
        },
    },
}
