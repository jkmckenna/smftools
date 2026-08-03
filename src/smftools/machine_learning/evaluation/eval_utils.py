from __future__ import annotations

import pandas as pd

from ..compatibility._warnings import deprecated_ml_alias


@deprecated_ml_alias(
    "smftools.machine_learning.evaluation.flatten_sliding_window_results",
    "smftools.analysis.compute.ml_results fixed-schema tables",
)
def flatten_sliding_window_results(results_dict):
    """
    Flatten nested sliding window results into pandas DataFrame.

    Expects structure:
        results[model_name][window_size][window_center]['metrics'][metric_name]
    """
    records = []

    for model_name, model_results in results_dict.items():
        for window_size, window_results in model_results.items():
            for center_var, result in window_results.items():
                metrics = result["metrics"]
                record = {"model": model_name, "window_size": window_size, "center_var": center_var}
                # Add all metrics
                record.update(metrics)
                records.append(record)

    df = pd.DataFrame.from_records(records)

    # Convert center_var to numeric if possible (optional but helpful for plotting)
    df["center_var"] = pd.to_numeric(df["center_var"], errors="coerce")
    df = df.sort_values(["model", "window_size", "center_var"])

    return df
