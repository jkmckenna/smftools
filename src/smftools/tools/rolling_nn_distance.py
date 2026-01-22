from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def _pack_bool_to_u64(values: np.ndarray) -> np.ndarray:
    """Pack a boolean matrix into uint64 blocks.

    Args:
        values: Boolean (or 0/1) matrix with shape ``(n_rows, n_cols)``.

    Returns:
        Packed uint64 array with shape ``(n_rows, ceil(n_cols / 64))``.
    """
    values = np.asarray(values, dtype=np.uint8)
    packed_u8 = np.packbits(values, axis=1)

    n_rows, n_bytes = packed_u8.shape
    pad = (-n_bytes) % 8
    if pad:
        packed_u8 = np.pad(
            packed_u8,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=0,
        )

    packed_u8 = np.ascontiguousarray(packed_u8)
    packed_u64 = packed_u8.reshape(n_rows, -1, 8).view(np.uint64).reshape(n_rows, -1)
    return packed_u64


def _popcount_u64_matrix(values: np.ndarray) -> np.ndarray:
    """Compute popcounts for a uint64 array.

    Args:
        values: Array of uint64 values.

    Returns:
        Integer array with the same shape as ``values``.
    """
    values = np.ascontiguousarray(values)
    bytes_view = values.view(np.uint8).reshape(values.shape + (8,))
    return np.unpackbits(bytes_view, axis=-1).sum(axis=-1)


def _resolve_site_mask(
    adata: "ad.AnnData",
    site_types: Optional[Sequence[str]],
    reference: Optional[str],
    site_var_suffix: str,
) -> Optional[np.ndarray]:
    if site_types is None:
        return None

    if not site_types:
        raise ValueError("site_types must contain at least one site type when provided")

    variants = _site_type_variants(site_types, site_var_suffix)
    existing = _resolve_site_columns(adata, variants, reference)

    logger.debug(
        "Resolving rolling NN site mask with site_types=%s reference=%s suffix=%s",
        site_types,
        reference,
        site_var_suffix,
    )
    logger.debug("Existing site columns in adata.var: %s", existing)

    if not existing:
        raise KeyError(f"No site columns found in adata.var for site_types={site_types}")

    mask = np.zeros(adata.n_vars, dtype=bool)
    for col in existing:
        values = adata.var[col]
        mask |= np.asarray(values, dtype=bool)

    logger.debug(
        "Rolling NN site mask selected %d/%d positions",
        int(mask.sum()),
        adata.n_vars,
    )
    if not mask.any():
        raise ValueError(f"Site mask empty for site_types={site_types}")

    return mask


def _site_type_variants(site_types: Sequence[str], site_var_suffix: str) -> list[str]:
    site_types = [str(site) for site in site_types]
    variants: list[str] = []

    for site in site_types:
        site_str = str(site)
        site_variants = {site_str}
        if not site_str.endswith(f"_{site_var_suffix}"):
            site_variants.add(f"{site_str}_{site_var_suffix}")
        else:
            site_variants.add(site_str.removesuffix(f"_{site_var_suffix}"))
        variants.extend(site_variants)

    return list(dict.fromkeys(variants))


def _resolve_site_columns(
    adata: "ad.AnnData",
    variants: Sequence[str],
    reference: Optional[str],
) -> list[str]:
    if reference is None:
        return [col for col in variants if col in adata.var]

    reference_cols = []
    for variant in variants:
        if variant.startswith(f"{reference}_"):
            reference_cols.append(variant)
        else:
            reference_cols.append(f"{reference}_{variant}")

    reference_cols = list(dict.fromkeys(reference_cols))
    reference_existing = [col for col in reference_cols if col in adata.var]
    if reference_existing:
        return reference_existing

    return [col for col in variants if col in adata.var]


def _get_layer_matrix(adata: "ad.AnnData", layer: Optional[str]) -> np.ndarray:
    X = adata.layers[layer] if layer is not None else adata.X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _validate_rolling_nn_params(window: int, step: int, min_overlap: int, n_vars: int) -> None:
    if window > n_vars:
        raise ValueError(f"window={window} is larger than n_vars={n_vars}")
    if window <= 0:
        raise ValueError("window must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if min_overlap <= 0:
        raise ValueError("min_overlap must be > 0")


def rolling_window_nn_distance(
    adata: "ad.AnnData",
    layer: Optional[str] = None,
    window: int = 15,
    step: int = 2,
    min_overlap: int = 10,
    return_fraction: bool = True,
    block_rows: int = 256,
    block_cols: int = 2048,
    store_obsm: Optional[str] = "rolling_nn_dist",
    site_types: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
    site_var_suffix: str = "site",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling-window nearest-neighbor distances per read.

    Distances are computed within each window using only overlapping observed
    positions. For each read, the nearest (minimum) distance to any other read
    is reported. If there is no neighbor with sufficient overlap, the value is
    ``NaN``.

    Args:
        adata: AnnData object containing the data.
        layer: Layer name to use; ``None`` uses ``adata.X``.
        window: Window size in ``adata.var`` coordinates.
        step: Step size between windows.
        min_overlap: Minimum overlapping observed positions.
        return_fraction: If ``True``, return mismatch/overlap; otherwise return
            mismatch counts.
        block_rows: Number of rows per block (controls memory use).
        block_cols: Number of columns per block (controls memory use).
        store_obsm: Key to store results in ``adata.obsm``. If ``None``, results
            are not stored on the AnnData object.
        site_types: Optional subset of site types to include when computing the
            rolling NN distances.
        reference: Optional reference label used to resolve site-specific columns.
        site_var_suffix: Suffix for site columns in ``adata.var``.

    Returns:
        Tuple of ``(out, starts)`` where ``out`` is ``(n_obs, n_windows)`` and
        ``starts`` is an array of window start indices.
    """
    site_mask = _resolve_site_mask(adata, site_types, reference, site_var_suffix)

    X = _get_layer_matrix(adata, layer)
    if site_mask is not None:
        X = X[:, site_mask]

    n_obs, n_vars = X.shape
    logger.debug(
        "Rolling NN distance matrix shape after site filtering: n_obs=%d n_vars=%d",
        n_obs,
        n_vars,
    )
    _validate_rolling_nn_params(window, step, min_overlap, n_vars)

    starts = np.arange(0, n_vars - window + 1, step, dtype=int)
    n_windows = len(starts)
    out = np.full((n_obs, n_windows), np.nan, dtype=float)

    for wi, start in enumerate(starts):
        window_slice = X[:, start : start + window]
        observed_mask = ~np.isnan(window_slice)
        values = np.where(observed_mask, window_slice, 0).astype(np.float32)
        values = (values > 0).astype(np.uint8)

        mask_u64 = _pack_bool_to_u64(observed_mask)
        values_u64 = _pack_bool_to_u64(values.astype(bool))

        best = np.full(n_obs, np.inf, dtype=float)

        for i0 in range(0, n_obs, block_rows):
            i1 = min(n_obs, i0 + block_rows)
            mask_i = mask_u64[i0:i1]
            values_i = values_u64[i0:i1]
            block_i = i1 - i0

            local_best = np.full(block_i, np.inf, dtype=float)

            for j0 in range(0, n_obs, block_cols):
                j1 = min(n_obs, j0 + block_cols)
                mask_j = mask_u64[j0:j1]
                values_j = values_u64[j0:j1]
                block_j = j1 - j0

                overlap_counts = np.zeros((block_i, block_j), dtype=np.uint16)
                mismatch_counts = np.zeros((block_i, block_j), dtype=np.uint16)

                for k in range(mask_i.shape[1]):
                    overlap_bits = (mask_i[:, k][:, None] & mask_j[:, k][None, :]).astype(
                        np.uint64
                    )
                    overlap_counts += _popcount_u64_matrix(overlap_bits).astype(np.uint16)

                    mismatch_bits = (
                        (values_i[:, k][:, None] ^ values_j[:, k][None, :]) & overlap_bits
                    ).astype(np.uint64)
                    mismatch_counts += _popcount_u64_matrix(mismatch_bits).astype(np.uint16)

                ok = overlap_counts >= min_overlap
                if not np.any(ok):
                    continue

                dist = np.full((block_i, block_j), np.inf, dtype=float)
                if return_fraction:
                    dist[ok] = mismatch_counts[ok] / overlap_counts[ok]
                else:
                    dist[ok] = mismatch_counts[ok].astype(float)

                if (i0 <= j1) and (j0 <= i1):
                    ii = np.arange(i0, i1)
                    jj = ii[(ii >= j0) & (ii < j1)]
                    if jj.size:
                        dist[(jj - i0), (jj - j0)] = np.inf

                local_best = np.minimum(local_best, dist.min(axis=1))

            best[i0:i1] = local_best

        best[~np.isfinite(best)] = np.nan
        out[:, wi] = best

    if store_obsm is not None:
        adata.obsm[store_obsm] = out
        adata.uns[f"{store_obsm}_starts"] = starts
        adata.uns[f"{store_obsm}_window"] = int(window)
        adata.uns[f"{store_obsm}_step"] = int(step)
        adata.uns[f"{store_obsm}_min_overlap"] = int(min_overlap)
        adata.uns[f"{store_obsm}_return_fraction"] = bool(return_fraction)
        adata.uns[f"{store_obsm}_layer"] = layer if layer is not None else "X"
        if site_mask is not None:
            adata.uns[f"{store_obsm}_var_indices"] = np.where(site_mask)[0]
            adata.uns[f"{store_obsm}_site_types"] = list(site_types or [])
            adata.uns[f"{store_obsm}_reference"] = reference

    return out, starts


def rolling_window_nn_distance_by_group(
    adata: "ad.AnnData",
    group_cols: Sequence[str],
    layer: Optional[str] = None,
    window: int = 15,
    step: int = 2,
    min_overlap: int = 10,
    return_fraction: bool = True,
    block_rows: int = 256,
    block_cols: int = 2048,
    store_obsm: Optional[str] = "rolling_nn_dist",
    site_types: Optional[Sequence[str]] = None,
    reference_col: Optional[str] = None,
    site_var_suffix: str = "site",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute rolling-window nearest-neighbor distances per group.

    Args:
        adata: AnnData object containing the data.
        group_cols: Observation columns defining the groups.
        layer: Layer name to use; ``None`` uses ``adata.X``.
        window: Window size in ``adata.var`` coordinates.
        step: Step size between windows.
        min_overlap: Minimum overlapping observed positions.
        return_fraction: If ``True``, return mismatch/overlap; otherwise return
            mismatch counts.
        block_rows: Number of rows per block (controls memory use).
        block_cols: Number of columns per block (controls memory use).
        store_obsm: Key to store results in ``adata.obsm``. If ``None``, results
            are not stored on the AnnData object.
        site_types: Optional subset of site types to include when computing the
            rolling NN distances.
        reference_col: Column containing reference labels for resolving site columns.
        site_var_suffix: Suffix for site columns in ``adata.var``.

    Returns:
        Tuple of ``(out, starts)`` where ``out`` is ``(n_obs, n_windows)`` and
        ``starts`` is an array of window start indices. Returns ``(None, None)``
        if no groups are available.
    """
    if not group_cols:
        raise ValueError("group_cols must contain at least one column name")

    missing = [col for col in group_cols if col not in adata.obs]
    if missing:
        raise KeyError(f"Missing group columns in adata.obs: {missing}")

    if site_types is not None and reference_col is not None and reference_col not in adata.obs:
        raise KeyError(f"Missing reference column in adata.obs: {reference_col}")

    group_df = adata.obs[list(group_cols)]
    grouped = group_df.groupby(list(group_cols), dropna=False)

    starts = None
    out_all = None

    for _, indices in grouped.indices.items():
        if len(indices) == 0:
            continue

        subset = adata[indices]
        reference = None
        if reference_col is not None:
            reference = str(subset.obs[reference_col].iloc[0])
        logger.debug(
            "Rolling NN group subset size=%d reference=%s site_types=%s",
            subset.n_obs,
            reference,
            site_types,
        )
        out, subset_starts = rolling_window_nn_distance(
            subset,
            layer=layer,
            window=window,
            step=step,
            min_overlap=min_overlap,
            return_fraction=return_fraction,
            block_rows=block_rows,
            block_cols=block_cols,
            store_obsm=None,
            site_types=site_types,
            reference=reference,
            site_var_suffix=site_var_suffix,
        )

        if starts is None:
            starts = subset_starts
            out_all = np.full((adata.n_obs, len(starts)), np.nan, dtype=float)
        elif not np.array_equal(starts, subset_starts):
            raise ValueError("Rolling window starts differ across groups")

        out_all[indices, :] = out

    if starts is None or out_all is None:
        return None, None

    if store_obsm is not None:
        adata.obsm[store_obsm] = out_all
        adata.uns[f"{store_obsm}_starts"] = starts
        adata.uns[f"{store_obsm}_window"] = int(window)
        adata.uns[f"{store_obsm}_step"] = int(step)
        adata.uns[f"{store_obsm}_min_overlap"] = int(min_overlap)
        adata.uns[f"{store_obsm}_return_fraction"] = bool(return_fraction)
        adata.uns[f"{store_obsm}_layer"] = layer if layer is not None else "X"
        adata.uns[f"{store_obsm}_group_cols"] = list(group_cols)
        if site_types is not None:
            adata.uns[f"{store_obsm}_site_types"] = list(site_types)
            adata.uns[f"{store_obsm}_reference_col"] = reference_col
            adata.uns[f"{store_obsm}_site_var_suffix"] = site_var_suffix

    return out_all, starts
