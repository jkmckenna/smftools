"""
Per-feature-type binning, rolling-mean, and peak-calling config for HMM feature histograms.

``HISTOGRAM_CONFIGS[feature_layer_name][hist_type]`` returns a dict or ``None``.

``hist_type`` is one of ``"count"``, ``"size"``, or ``"neighbor_distance"``.
Each dict may contain ``bin_size_bp``, ``rolling_window_bp``, ``peak_kwargs``,
``rolling_color``, ``peak_color``, and optionally ``gaussian_fit``.
"""

_NONE = None

HISTOGRAM_CONFIGS: dict[str, dict[str, dict | None]] = {
    "C_all_accessible_features_merged": {
        "count": _NONE,
        "size": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
    },
    "C_all_footprint_features": {
        "count": _NONE,
        "size": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
    },
    "C_nucleosome_depleted_region_merged": {
        "count": _NONE,
        "size": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 4, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
    },
    "C_putative_nucleosome": {
        "count": _NONE,
        "size": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
        },
        "neighbor_distance": {
            "bin_size_bp": 5,
            "rolling_window_bp": 20,
            "peak_kwargs": {"height": 0.15, "distance": 10, "prominence": 0.10},
            "rolling_color": "#333333",
            "peak_color": "#d62728",
            "gaussian_fit": {"fit_range_bp": (0, 300), "fit_color": "#1f77b4"},
        },
    },
}
