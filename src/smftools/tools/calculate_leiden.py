from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_leiden(
    adata: "ad.AnnData",
    *,
    resolution: float = 0.1,
    key_added: str = "leiden",
    connectivities_key: str = "connectivities",
) -> "ad.AnnData":
    """Compute Leiden clusters from a connectivity graph.

    Args:
        adata: AnnData object with ``obsp[connectivities_key]`` set.
        resolution: Resolution parameter for Leiden clustering.
        key_added: Column name to store cluster assignments in ``adata.obs``.
        connectivities_key: Key in ``adata.obsp`` containing a sparse adjacency matrix.

    Returns:
        Updated AnnData object with Leiden labels in ``adata.obs``.
    """
    if connectivities_key not in adata.obsp:
        raise KeyError(f"Missing connectivities '{connectivities_key}' in adata.obsp.")

    igraph = require("igraph", extra="cluster", purpose="Leiden clustering")
    leidenalg = require("leidenalg", extra="cluster", purpose="Leiden clustering")

    connectivities = adata.obsp[connectivities_key]
    coo = connectivities.tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    graph = igraph.Graph(n=connectivities.shape[0], edges=edges, directed=False)
    graph.es["weight"] = coo.data.tolist()

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
    )

    labels = np.array(partition.membership, dtype=str)
    adata.obs[key_added] = pd.Categorical(labels)
    logger.info("Stored Leiden clusters in adata.obs['%s'].", key_added)
    return adata
