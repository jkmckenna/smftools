"""Plotting utilities for UMI bipartite graph analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

plt = require("matplotlib.pyplot", extra="plotting", purpose="UMI bipartite summary")

logger = get_logger(__name__)


def plot_umi_bipartite_summary(
    count_matrix: pd.DataFrame,
    sample_name: str,
    reference_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot a fidelity scatter of U1-U2 pairing quality.

    Each point is a unique (U1_cluster, U2_cluster) pair.  Position shows the
    fraction of that cluster's reads accounted for by this pairing (U1 fidelity
    on x, U2 fidelity on y).  Point size and colour encode read count.
    Dominant 1:1 pairs cluster at (1, 1); chimeric noise sits at lower values.

    Parameters
    ----------
    count_matrix : pd.DataFrame
        Rows = U1 cluster consensus sequences, columns = U2 cluster consensus
        sequences, values = read counts.
    sample_name : str
        Sample identifier (used in title).
    reference_name : str
        Reference identifier (used in title).
    save_path : Path or None
        When set, saves figure as PNG at 300 dpi.  Otherwise displays.
    """
    if count_matrix.empty:
        logger.warning(
            "Empty count matrix for sample=%s, reference=%s; skipping summary plot.",
            sample_name,
            reference_name,
        )
        return

    mat = count_matrix.values
    n_u1, n_u2 = mat.shape

    # Flatten to non-zero (u1, u2, count) triples
    u1_idx, u2_idx = np.nonzero(mat)
    counts = mat[u1_idx, u2_idx]

    # Marginals
    u1_totals = mat.sum(axis=1)
    u2_totals = mat.sum(axis=0)

    # Per-pair fidelity
    u1_fidelity = counts / u1_totals[u1_idx]
    u2_fidelity = counts / u2_totals[u2_idx]

    # ---- Fidelity scatter ----
    fig, ax = plt.subplots(figsize=(7, 6))

    size = np.clip(counts, 1, None)
    size_scaled = 10 + 40 * (size / max(size.max(), 1))
    sc = ax.scatter(
        u1_fidelity, u2_fidelity,
        s=size_scaled, c=np.log1p(counts),
        cmap="YlOrRd", alpha=0.6, edgecolors="black", linewidths=0.3,
    )
    ax.set_xlabel("U1 fidelity")
    ax.set_ylabel("U2 fidelity")
    ax.set_title(
        f"UMI Fidelity: {sample_name} / {reference_name}\n"
        f"({n_u1} U1 clusters, {n_u2} U2 clusters, {int(mat.sum())} reads)",
        fontsize=11,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log1p(read count)", fontsize=8)

    # Annotate fraction of reads in dominant 1:1 pairs
    dominant_mask = (u1_fidelity == 1.0) & (u2_fidelity == 1.0)
    reads_in_dominant = int(counts[dominant_mask].sum()) if dominant_mask.any() else 0
    total_reads = int(counts.sum())
    ax.text(
        0.02, 0.02,
        f"Reads in 1:1 pairs: {reads_in_dominant}/{total_reads} "
        f"({reads_in_dominant / max(total_reads, 1):.0%})",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved UMI bipartite summary: %s", save_path)
        plt.close(fig)
    else:
        plt.show()
