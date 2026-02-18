"""Plotting utilities for UMI bipartite graph analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

sns = require("seaborn", extra="plotting", purpose="UMI bipartite clustermap")
plt = require("matplotlib.pyplot", extra="plotting", purpose="UMI bipartite clustermap")

logger = get_logger(__name__)


_MAX_CLUSTERMAP_DIM = 200


def plot_umi_bipartite_clustermap(
    count_matrix: pd.DataFrame,
    sample_name: str,
    reference_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot a clustermap of the U1×U2 count matrix.

    Parameters
    ----------
    count_matrix : pd.DataFrame
        Rows are U1 cluster consensus sequences, columns are U2 cluster consensus
        sequences, values are read counts.
    sample_name : str
        Sample identifier (used in title and filename).
    reference_name : str
        Reference identifier (used in title and filename).
    save_path : Path or None
        When set, saves the figure as PNG at 300 dpi. Otherwise displays interactively.
    """
    if count_matrix.empty:
        logger.warning(
            "Empty count matrix for sample=%s, reference=%s; skipping clustermap.",
            sample_name,
            reference_name,
        )
        return

    n_rows, n_cols = count_matrix.shape

    if max(n_rows, n_cols) > _MAX_CLUSTERMAP_DIM:
        logger.warning(
            "Count matrix too large for clustermap (%d × %d > %d); "
            "skipping plot for sample=%s, reference=%s. "
            "Consider increasing min_edge_count_for_plot to reduce matrix size.",
            n_rows,
            n_cols,
            _MAX_CLUSTERMAP_DIM,
            sample_name,
            reference_name,
        )
        return

    # Log1p scale for readability
    log_matrix = np.log1p(count_matrix.astype(float))

    # Determine whether to annotate cells with counts and show tick labels
    annotate = n_rows <= 20 and n_cols <= 20
    show_tick_labels = max(n_rows, n_cols) <= 80

    figsize = (max(6, n_cols * 0.6 + 2), max(5, n_rows * 0.5 + 2))

    try:
        g = sns.clustermap(
            log_matrix,
            cmap="YlOrRd",
            annot=count_matrix.values if annotate else False,
            fmt="d" if annotate else "",
            figsize=figsize,
            linewidths=0.5 if annotate else 0,
            xticklabels=show_tick_labels,
            yticklabels=show_tick_labels,
            cbar_kws={"label": "log1p(read count)"},
        )
        g.fig.suptitle(
            f"UMI Bipartite: {sample_name} / {reference_name}\n"
            f"({n_rows} U1 × {n_cols} U2 clusters)",
            y=1.02,
            fontsize=12,
        )
        g.ax_heatmap.set_xlabel("U2 cluster")
        g.ax_heatmap.set_ylabel("U1 cluster")

        if save_path is not None:
            save_path = Path(save_path)
            g.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved UMI bipartite clustermap: %s", save_path)
            plt.close(g.fig)
        else:
            plt.show()
    except Exception:
        logger.exception(
            "Failed to plot UMI bipartite clustermap for sample=%s, reference=%s",
            sample_name,
            reference_name,
        )
