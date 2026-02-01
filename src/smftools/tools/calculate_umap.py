from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_umap(
    adata: "ad.AnnData",
    obsm: str | None = "X_pca",
    overwrite: bool = True,
    threads: int = 8,
    random_state: int | None = 0,
    output_suffix: str | None = None,
) -> "ad.AnnData":
    """Compute UMAP embedding from an `.obsm` embedding, and store connectivities."""

    import numpy as np
    import scipy.sparse as sp

    if obsm is None:
        raise ValueError("obsm must be a key in adata.obsm (e.g., 'X_pca').")

    if obsm not in adata.obsm:
        raise KeyError(f"`{obsm}` not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    umap = require("umap", extra="umap", purpose="UMAP calculation")

    output_obsm = f"X_umap_{output_suffix}" if output_suffix else "X_umap"
    conn_key = f"connectivities_{obsm}"

    # Decide n_neighbors: prefer stored KNN params, else UMAP default-ish
    n_neighbors = None
    knn_uns_key = f"knn_distances_{obsm}"
    if knn_uns_key in adata.uns:
        params = adata.uns[knn_uns_key].get("params", {})
        n_neighbors = params.get("n_neighbors_used", params.get("n_neighbors", None))
    if n_neighbors is None:
        n_neighbors = 15  # reasonable default if KNN wasn't precomputed
        logger.warning(
            "No %r found in adata.uns; defaulting n_neighbors=%d for UMAP.",
            knn_uns_key,
            n_neighbors,
        )

    # Build input matrix X and handle NaNs locally
    X = adata.obsm[obsm]
    if sp.issparse(X):
        # UMAP can accept sparse CSR; keep it sparse
        pass
    else:
        X = np.asarray(X)
        if np.isnan(X).any():
            logger.warning("NaNs detected in %s; filling NaNs with 0.5 for UMAP.", obsm)
            X = np.nan_to_num(X, nan=0.5)

    if (not overwrite) and (output_obsm in adata.obsm) and (conn_key in adata.obsp):
        logger.info("UMAP + connectivities already exist and overwrite=False; skipping.")
        return adata

    logger.info("Running UMAP (obsm=%s, n_neighbors=%d, metric=euclidean)", obsm, n_neighbors)

    # Note: umap-learn uses numba threading; n_jobs controls parallelism in UMAP
    # and is ignored when random_state is set (umap-learn behavior).
    umap_model = umap.UMAP(
        n_neighbors=int(n_neighbors),
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        n_jobs=int(threads),
    )

    embedding = umap_model.fit_transform(X)
    adata.obsm[output_obsm] = embedding

    # UMAP's computed fuzzy graph
    connectivities = getattr(umap_model, "graph_", None)
    if connectivities is not None:
        adata.obsp[conn_key] = (
            connectivities.tocsr() if sp.issparse(connectivities) else connectivities
        )
    else:
        logger.warning("UMAP model did not expose graph_; connectivities not stored.")

    adata.uns[output_obsm] = {
        "params": {
            "obsm": obsm,
            "n_neighbors": int(n_neighbors),
            "metric": "euclidean",
            "random_state": random_state,
            "n_jobs": int(threads),
        }
    }

    logger.info("Stored: adata.obsm[%s]=%s", output_obsm, embedding.shape)
    return adata
