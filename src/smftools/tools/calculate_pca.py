from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def calculate_pca(
    adata: "ad.AnnData",
    layer: str | None = "nan_half",
    var_filters: Sequence[str] | None = None,
    n_pcs: int = 15,
    overwrite: bool = True,
    output_suffix: str | None = None,
    fill_nan: float | None = 0.5,
) -> "ad.AnnData":
    """Compute PCA and store scores in `.obsm` and loadings in `.varm`."""

    import numpy as np
    import scipy.sparse as sp

    obsm_output = f"X_pca_{output_suffix}" if output_suffix else "X_pca"
    varm_output = f"PCs_{output_suffix}" if output_suffix else "PCs"

    if not overwrite and obsm_output in adata.obsm and varm_output in adata.varm:
        logger.info("PCA outputs already exist and overwrite=False; skipping (%s, %s).", obsm_output, varm_output)
        return adata

    # --- Build feature subset mask (over vars) ---
    if var_filters:
        missing = [f for f in var_filters if f not in adata.var.columns]
        if missing:
            raise KeyError(f"var_filters not found in adata.var: {missing}")

        masks = []
        for f in var_filters:
            m = np.asarray(adata.var[f].values)
            if m.dtype != bool:
                m = m.astype(bool)
            masks.append(m)

        subset_mask = np.logical_or.reduce(masks)
        n_vars_used = int(subset_mask.sum())
        if n_vars_used == 0:
            raise ValueError(f"var_filters={var_filters} retained 0 features.")
        logger.info("Subsetting vars: retained %d / %d features from filters %s", n_vars_used, adata.n_vars, var_filters)
    else:
        subset_mask = slice(None)
        n_vars_used = adata.n_vars
        logger.info("No var_filters provided; using all %d features.", adata.n_vars)

    # --- Pull matrix view ---
    if layer is None:
        matrix = adata.X
        layer_used = None
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in adata.layers. Available: {list(adata.layers.keys())}")
        matrix = adata.layers[layer]
        layer_used = layer

    matrix = matrix[:, subset_mask]  # slice view (sparse OK)

    n_obs = matrix.shape[0]
    if n_obs < 2:
        raise ValueError(f"PCA requires at least 2 observations; got n_obs={n_obs}")
    if n_vars_used < 1:
        raise ValueError("PCA requires at least 1 feature.")

    n_pcs_requested = int(n_pcs)
    n_pcs_used = min(n_pcs_requested, n_obs, n_vars_used)
    if n_pcs_used < 1:
        raise ValueError(f"n_pcs_used became {n_pcs_used}; check inputs.")

    # --- NaN handling (dense only; sparse usually won’t store NaNs) ---
    if not sp.issparse(matrix):
        X = np.asarray(matrix, dtype=np.float32)
        if fill_nan is not None and np.isnan(X).any():
            logger.warning("NaNs detected; filling NaNs with %s before PCA.", fill_nan)
            X = np.nan_to_num(X, nan=float(fill_nan))
    else:
        X = matrix  # keep sparse

    # --- PCA ---
    # Prefer sklearn's randomized PCA for speed on big matrices.
    used_sklearn = False
    try:
        sklearn = require("sklearn", extra="ml", purpose="PCA computation")
        from sklearn.decomposition import PCA, TruncatedSVD

        if sp.issparse(X):
            # TruncatedSVD works on sparse without centering; good approximation.
            # If you *need* centered PCA on sparse, you'd need different machinery.
            logger.info("Running TruncatedSVD (sparse) with n_components=%d", n_pcs_used)
            model = TruncatedSVD(n_components=n_pcs_used, random_state=0)
            scores = model.fit_transform(X)                    # (n_obs, n_pcs)
            loadings = model.components_.T                     # (n_vars_used, n_pcs)
            mean = None
            explained_variance_ratio = getattr(model, "explained_variance_ratio_", None)
        else:
            logger.info("Running sklearn PCA with n_components=%d (svd_solver=randomized)", n_pcs_used)
            model = PCA(n_components=n_pcs_used, svd_solver="randomized", random_state=0)
            scores = model.fit_transform(X)                    # (n_obs, n_pcs)
            loadings = model.components_.T                     # (n_vars_used, n_pcs)
            mean = model.mean_
            explained_variance_ratio = model.explained_variance_ratio_

        used_sklearn = True

    except Exception as e:
        # Fallback to your manual SVD (dense only)
        if sp.issparse(X):
            raise RuntimeError(
                "Sparse input PCA fallback is not implemented without sklearn. "
                "Install scikit-learn (extra 'ml') or densify upstream."
            ) from e

        import scipy.linalg as spla

        logger.warning("sklearn PCA unavailable; falling back to full SVD (can be slow). Reason: %s", e)
        Xd = np.asarray(X, dtype=np.float64)
        mean = Xd.mean(axis=0)
        centered = Xd - mean
        u, s, vt = spla.svd(centered, full_matrices=False)
        u = u[:, :n_pcs_used]
        s = s[:n_pcs_used]
        vt = vt[:n_pcs_used]
        scores = u * s
        loadings = vt.T
        explained_variance_ratio = None

    # --- Store scores (obsm) ---
    adata.obsm[obsm_output] = scores

    # --- Store loadings (varm) with original var dimension ---
    pc_matrix = np.zeros((adata.n_vars, n_pcs_used), dtype=np.float32)
    if isinstance(subset_mask, slice):
        pc_matrix[:, :] = loadings
    else:
        pc_matrix[subset_mask, :] = loadings.astype(np.float32, copy=False)

    adata.varm[varm_output] = pc_matrix

    # --- Metadata ---
    adata.uns[obsm_output] = {
        "params": {
            "layer": layer_used,
            "var_filters": list(var_filters) if var_filters else None,
            "n_pcs_requested": n_pcs_requested,
            "n_pcs_used": int(n_pcs_used),
            "used_sklearn": used_sklearn,
            "fill_nan": fill_nan,
            "note_sparse": bool(sp.issparse(matrix)),
        },
        "explained_variance_ratio": explained_variance_ratio,
        "mean": mean.tolist() if (mean is not None and isinstance(mean, np.ndarray)) else None,
    }

    logger.info("Stored PCA: adata.obsm[%s] (%s) and adata.varm[%s] (%s)", obsm_output, scores.shape, varm_output, pc_matrix.shape)
    return adata
