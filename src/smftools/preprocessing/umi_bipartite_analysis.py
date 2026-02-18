"""Bipartite graph analysis for U1×U2 UMI cluster pairing QC."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    import anndata as ad


def analyze_umi_bipartite_graph(
    adata: "ad.AnnData",
    umi_cols: Sequence[str] = ("U1", "U2"),
    sample_col: str = "Experiment_name_and_barcode",
    reference_col: str = "Reference_strand",
    output_directory: Optional[Path] = None,
    min_edge_count_for_plot: int = 2,
    uns_flag: str = "umi_bipartite_analysis_performed",
    bypass: bool = False,
    force_redo: bool = False,
) -> "ad.AnnData":
    """Build a bipartite U1×U2 count matrix per (sample, reference) group and annotate reads.

    For each (sample_col, reference_col) group, constructs a count matrix with
    U1 clusters as rows and U2 clusters as columns.  Per-read annotations are
    added to ``adata.obs``:

    - ``RX_edge_count`` – number of reads sharing this (U1_cluster, U2_cluster) pair
    - ``RX_is_dominant_pair`` – True when this pair is the dominant partner for
      **both** its U1 and U2 cluster
    - ``RX_U1_fidelity`` – fraction of reads with this U1_cluster that share this U2_cluster
    - ``RX_U2_fidelity`` – fraction of reads with this U2_cluster that share this U1_cluster

    Parameters
    ----------
    adata : AnnData
        Must contain ``U1_cluster`` and ``U2_cluster`` columns in ``.obs``.
    umi_cols : Sequence[str]
        The two UMI column name prefixes (default ``("U1", "U2")``).
    sample_col, reference_col : str
        Columns used to define groups.
    output_directory : Path or None
        When set, saves clustermap PNGs for each group.
    min_edge_count_for_plot : int
        Only U1/U2 clusters involved in at least one edge with this many reads
        are included in the clustermap.  Filters out singleton noise that would
        otherwise produce an uninformatively large matrix (default 2).
    uns_flag : str
        Key in ``adata.uns`` to mark completion.
    bypass : bool
        Skip processing entirely.
    force_redo : bool
        Rerun even if ``uns_flag`` is already set.

    Returns
    -------
    AnnData
        Modified in-place with new ``.obs`` columns and ``adata.uns["umi_bipartite_stats"]``.
    """
    if bypass:
        logger.info("Bypassing UMI bipartite analysis (bypass=True)")
        return adata

    if adata.uns.get(uns_flag, False) and not force_redo:
        logger.info("UMI bipartite analysis already performed (set force_redo=True to rerun)")
        return adata

    u1_col = f"{umi_cols[0]}_cluster"
    u2_col = f"{umi_cols[1]}_cluster"

    if u1_col not in adata.obs.columns or u2_col not in adata.obs.columns:
        logger.warning(
            "Missing cluster columns (%s, %s) in adata.obs; skipping bipartite analysis.",
            u1_col,
            u2_col,
        )
        return adata

    # Initialise per-read annotation columns
    adata.obs["RX_edge_count"] = 0
    adata.obs["RX_is_dominant_pair"] = False
    adata.obs["RX_U1_fidelity"] = np.nan
    adata.obs["RX_U2_fidelity"] = np.nan

    # Determine grouping columns present in obs
    group_cols = [c for c in (sample_col, reference_col) if c in adata.obs.columns]
    if group_cols:
        grouped = adata.obs.groupby(group_cols, observed=True)
    else:
        grouped = [("all", adata.obs)]

    bipartite_stats: dict = {}

    for group_key, group_df in grouped:
        idx = group_df.index

        u1_vals = adata.obs.loc[idx, u1_col]
        u2_vals = adata.obs.loc[idx, u2_col]

        # Only consider reads where both clusters are non-null
        valid_mask = u1_vals.notna() & u2_vals.notna()
        valid_idx = idx[valid_mask]

        if len(valid_idx) == 0:
            continue

        u1_valid = u1_vals.loc[valid_idx].astype(str)
        u2_valid = u2_vals.loc[valid_idx].astype(str)

        # Build count matrix
        pair_counts = (
            pd.DataFrame({"u1": u1_valid.values, "u2": u2_valid.values})
            .groupby(["u1", "u2"])
            .size()
        )
        count_matrix = pair_counts.unstack(fill_value=0)

        # Row (U1) and column (U2) marginals
        u1_totals = count_matrix.sum(axis=1)
        u2_totals = count_matrix.sum(axis=0)

        # Dominant partners
        u1_dominant_u2 = count_matrix.idxmax(axis=1)  # for each U1, dominant U2
        u2_dominant_u1 = count_matrix.idxmax(axis=0)  # for each U2, dominant U1

        # Annotate each valid read
        for obs_name in valid_idx:
            u1 = str(adata.obs.at[obs_name, u1_col])
            u2 = str(adata.obs.at[obs_name, u2_col])

            edge_count = int(count_matrix.at[u1, u2]) if u1 in count_matrix.index and u2 in count_matrix.columns else 0
            adata.obs.at[obs_name, "RX_edge_count"] = edge_count

            u1_fidelity = edge_count / int(u1_totals[u1]) if u1 in u1_totals.index else np.nan
            u2_fidelity = edge_count / int(u2_totals[u2]) if u2 in u2_totals.index else np.nan
            adata.obs.at[obs_name, "RX_U1_fidelity"] = u1_fidelity
            adata.obs.at[obs_name, "RX_U2_fidelity"] = u2_fidelity

            is_dominant = (
                u1 in u1_dominant_u2.index
                and u2 in u2_dominant_u1.index
                and u1_dominant_u2[u1] == u2
                and u2_dominant_u1[u2] == u1
            )
            adata.obs.at[obs_name, "RX_is_dominant_pair"] = is_dominant

        # Store h5ad-safe stats (plain dicts)
        group_label = str(group_key) if not isinstance(group_key, tuple) else "__".join(str(g) for g in group_key)
        bipartite_stats[group_label] = {
            "count_matrix": {
                "index": count_matrix.index.tolist(),
                "columns": count_matrix.columns.tolist(),
                "data": count_matrix.values.tolist(),
            },
            "n_u1_clusters": int(count_matrix.shape[0]),
            "n_u2_clusters": int(count_matrix.shape[1]),
            "n_valid_reads": int(len(valid_idx)),
        }

        logger.info(
            "Bipartite analysis group=%s: %d U1 × %d U2 clusters, %d reads",
            group_label,
            count_matrix.shape[0],
            count_matrix.shape[1],
            len(valid_idx),
        )

        # Optional plotting
        if output_directory is not None:
            from ..plotting import plot_umi_bipartite_clustermap

            output_directory = Path(output_directory)
            if isinstance(group_key, tuple):
                sample_name = str(group_key[0]) if len(group_key) > 0 else "all"
                reference_name = str(group_key[1]) if len(group_key) > 1 else "all"
            else:
                sample_name = str(group_key)
                reference_name = "all"

            # Filter to clusters involved in at least one edge >= min_edge_count_for_plot
            plot_matrix = count_matrix.copy()
            if min_edge_count_for_plot > 1:
                row_mask = (plot_matrix >= min_edge_count_for_plot).any(axis=1)
                col_mask = (plot_matrix >= min_edge_count_for_plot).any(axis=0)
                plot_matrix = plot_matrix.loc[row_mask, col_mask]
                n_dropped_rows = int((~row_mask).sum())
                n_dropped_cols = int((~col_mask).sum())
                if n_dropped_rows or n_dropped_cols:
                    logger.info(
                        "Filtered plot matrix for group=%s: dropped %d/%d U1 and %d/%d U2 "
                        "singleton clusters (min_edge_count_for_plot=%d)",
                        group_label,
                        n_dropped_rows,
                        count_matrix.shape[0],
                        n_dropped_cols,
                        count_matrix.shape[1],
                        min_edge_count_for_plot,
                    )

            save_path = output_directory / f"{reference_name}__{sample_name}__umi_bipartite.png"
            plot_umi_bipartite_clustermap(
                count_matrix=plot_matrix,
                sample_name=sample_name,
                reference_name=reference_name,
                save_path=save_path,
            )

    adata.uns["umi_bipartite_stats"] = bipartite_stats
    adata.uns[uns_flag] = True

    n_dominant = int(adata.obs["RX_is_dominant_pair"].sum())
    n_annotated = int((adata.obs["RX_edge_count"] > 0).sum())
    logger.info(
        "UMI bipartite analysis complete: %d reads annotated, %d dominant-pair reads",
        n_annotated,
        n_dominant,
    )

    return adata
