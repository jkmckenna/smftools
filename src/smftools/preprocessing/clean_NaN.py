from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import scipy.sparse as sp

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)

VALID_NAN_LAYERS = frozenset(
    ["fill_nans_closest", "nan0_0minus1", "nan1_12", "nan_minus_1", "nan_half"]
)
DEFAULT_NAN_LAYERS = ["nan0_0minus1", "nan_half"]


def _finalize_layer_dtype(arr: np.ndarray) -> np.ndarray:
    """Downcast a NaN-fill layer to int8 when it is safe, else keep float32.

    These layers are large and dense (n_reads x n_positions), so halving/quartering
    their dtype is a meaningful memory win. But two constraints apply:

    - int8 cannot hold NaN, so any layer that still contains NaN must stay float32.
    - float16 is NOT a safe alternative here: numpy keeps a float16 accumulator in
      sum/nansum/nanmean, so reducing a float16 layer over the read axis (which many
      downstream stats do) overflows to inf past ~2048 elements. int8 is safe because
      numpy upcasts integer reductions to int64.

    So we only downcast to int8 when the built layer is finite and exactly
    integer-valued within int8 range -- which is the case when clean_NaN ran on a
    binarized {0,1,NaN} layer (its outputs are then {-1,0,1}/{0,1,2}). When clean_NaN
    ran on a continuous X (the non-native path), "else X" keeps continuous values and
    this check correctly leaves the layer float32 rather than truncating it.
    Layers that carry a non-integer fill (e.g. nan_half's 0.5) also stay float32.
    """
    if np.isnan(arr).any():
        return np.asarray(arr, dtype=np.float32)
    if (
        arr.size > 0
        and np.array_equal(arr, np.trunc(arr))
        and arr.min() >= -128
        and arr.max() <= 127
    ):
        return arr.astype(np.int8, copy=False)
    return np.asarray(arr, dtype=np.float32)


def _ffill_bfill_rows(X: np.ndarray) -> np.ndarray:
    """Forward-then-backward fill NaN values along axis=1 (row-wise).

    Uses a mask-propagation approach to avoid pandas overhead.
    Operates in float32 and returns float32.
    """
    out = X.copy()
    n_rows, n_cols = out.shape

    # forward fill: carry last non-NaN value rightward
    for j in range(1, n_cols):
        mask = np.isnan(out[:, j])
        out[mask, j] = out[mask, j - 1]

    # backward fill: carry first non-NaN value leftward
    for j in range(n_cols - 2, -1, -1):
        mask = np.isnan(out[:, j])
        out[mask, j] = out[mask, j + 1]

    return out


def clean_NaN(
    adata: "ad.AnnData",
    layer: str | None = None,
    uns_flag: str = "clean_NaN_performed",
    bypass: bool = False,
    force_redo: bool = True,
    layers_to_build: Optional[List[str]] = None,
) -> None:
    """Append layers to ``adata`` that contain NaN-cleaning strategies.

    Uses numpy float32 operations throughout to avoid the memory overhead of
    converting to a float64 pandas DataFrame.

    Args:
        adata: AnnData object.
        layer: Layer to fill NaN values in. If ``None``, uses ``adata.X``.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        layers_to_build: Which NaN-fill strategy layers to create. Valid values:
            ``fill_nans_closest``, ``nan0_0minus1``, ``nan1_12``,
            ``nan_minus_1``, ``nan_half``. Defaults to
            ``["nan0_0minus1", "nan_half"]``.
    """

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        return

    if layers_to_build is None:
        layers_to_build = DEFAULT_NAN_LAYERS

    unknown = set(layers_to_build) - VALID_NAN_LAYERS
    if unknown:
        raise ValueError(
            f"Unknown NaN layer(s): {unknown}. Valid options: {sorted(VALID_NAN_LAYERS)}"
        )

    if not layers_to_build:
        logger.info("clean_NaN: layers_to_build is empty, nothing to do.")
        adata.uns[uns_flag] = True
        return

    # Ensure the specified source layer exists
    if layer and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Extract base matrix as float32 (avoids float64 pandas overhead)
    data = adata.layers[layer] if layer else adata.X
    if sp.issparse(data):
        X = data.toarray().astype(np.float32)
    else:
        X = np.asarray(data, dtype=np.float32)

    nan_mask = np.isnan(X)

    if "fill_nans_closest" in layers_to_build:
        logger.info("Making layer: fill_nans_closest")
        adata.layers["fill_nans_closest"] = _finalize_layer_dtype(_ffill_bfill_rows(X))

    if "nan0_0minus1" in layers_to_build:
        logger.info("Making layer: nan0_0minus1")
        adata.layers["nan0_0minus1"] = _finalize_layer_dtype(
            np.where(nan_mask, np.float32(0.0), np.where(X == 0, np.float32(-1.0), X))
        )

    if "nan1_12" in layers_to_build:
        logger.info("Making layer: nan1_12")
        adata.layers["nan1_12"] = _finalize_layer_dtype(
            np.where(nan_mask, np.float32(1.0), np.where(X == 1, np.float32(2.0), X))
        )

    if "nan_minus_1" in layers_to_build:
        logger.info("Making layer: nan_minus_1")
        adata.layers["nan_minus_1"] = _finalize_layer_dtype(
            np.where(nan_mask, np.float32(-1.0), X)
        )

    if "nan_half" in layers_to_build:
        logger.info("Making layer: nan_half")
        adata.layers["nan_half"] = _finalize_layer_dtype(np.where(nan_mask, np.float32(0.5), X))

    del X, nan_mask

    # mark as done
    adata.uns[uns_flag] = True

    return None
