from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.logging_utils import get_logger
from smftools.optional_imports import require

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def build_sequence_one_hot_and_mask(
    encoded_sequences: np.ndarray,
    *,
    bases: Sequence[str] = ("A", "C", "G", "T"),
    dtype: np.dtype | type[np.floating] = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot encoded reads and a seen/unseen mask.

    Args:
        encoded_sequences: Integer-encoded sequences shaped (n_reads, seq_len).
        bases: Bases to one-hot encode.
        dtype: Output dtype for the one-hot tensor.

    Returns:
        Tuple of (one_hot_tensor, mask) where:
            - one_hot_tensor: (n_reads, seq_len, n_bases)
            - mask: (n_reads, seq_len) boolean array indicating seen bases.
    """
    encoded = np.asarray(encoded_sequences)
    if encoded.ndim != 2:
        raise ValueError(
            f"encoded_sequences must be 2D with shape (n_reads, seq_len); got {encoded.shape}."
        )

    base_values = np.array(
        [MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[base] for base in bases],
        dtype=encoded.dtype,
    )

    if np.issubdtype(encoded.dtype, np.floating):
        encoded = encoded.copy()
        encoded[np.isnan(encoded)] = -1

    mask = np.isin(encoded, base_values)
    one_hot = np.zeros((*encoded.shape, len(base_values)), dtype=dtype)

    for idx, base_value in enumerate(base_values):
        one_hot[..., idx] = encoded == base_value

    return one_hot, mask


def calculate_sequence_cp_decomposition(
    adata: "ad.AnnData",
    *,
    layer: str,
    rank: int = 5,
    n_iter_max: int = 100,
    random_state: int = 0,
    overwrite: bool = True,
    embedding_key: str = "X_cp_sequence",
    components_key: str = "H_cp_sequence",
    uns_key: str = "cp_sequence",
    bases: Iterable[str] = ("A", "C", "G", "T"),
    backend: str = "pytorch",
    show_progress: bool = False,
    init: str = "random",
) -> "ad.AnnData":
    """Compute CP decomposition on one-hot encoded sequence data with masking.

    Args:
        adata: AnnData object to update.
        layer: Layer name containing integer-encoded sequences.
        rank: CP rank.
        n_iter_max: Maximum number of iterations for the solver.
        random_state: Random seed for initialization.
        overwrite: Whether to recompute if the embedding already exists.
        embedding_key: Key for embedding in ``adata.obsm``.
        components_key: Key for position factors in ``adata.varm``.
        uns_key: Key for metadata stored in ``adata.uns``.
        bases: Bases to one-hot encode (in order).
        backend: Tensorly backend to use (``numpy`` or ``pytorch``).
        show_progress: Whether to display progress during factorization if supported.

    Returns:
        Updated AnnData object containing the CP decomposition outputs.
    """
    if embedding_key in adata.obsm and components_key in adata.varm and not overwrite:
        logger.info("CP embedding and components already present; skipping recomputation.")
        return adata

    if backend not in {"numpy", "pytorch"}:
        raise ValueError(f"Unsupported backend '{backend}'. Use 'numpy' or 'pytorch'.")

    tensorly = require("tensorly", extra="ml-base", purpose="CP decomposition")
    from tensorly.decomposition import parafac

    tensorly.set_backend(backend)

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")

    one_hot, mask = build_sequence_one_hot_and_mask(adata.layers[layer], bases=tuple(bases))
    mask_tensor = np.repeat(mask[:, :, None], one_hot.shape[2], axis=2)

    device = "numpy"
    if backend == "pytorch":
        torch = require("torch", extra="ml-base", purpose="CP decomposition backend")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        one_hot = torch.tensor(one_hot, dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32, device=device)

    parafac_kwargs = {
        "rank": rank,
        "n_iter_max": n_iter_max,
        "init": init,
        "mask": mask_tensor,
        "random_state": random_state,
    }
    import inspect

    if "verbose" in inspect.signature(parafac).parameters:
        parafac_kwargs["verbose"] = show_progress

    cp = parafac(one_hot, **parafac_kwargs)

    if backend == "pytorch":
        weights = cp.weights.detach().cpu().numpy()
        read_factors, position_factors, base_factors = [
            factor.detach().cpu().numpy() for factor in cp.factors
        ]
    else:
        weights = np.asarray(cp.weights)
        read_factors, position_factors, base_factors = [np.asarray(f) for f in cp.factors]

    adata.obsm[embedding_key] = read_factors
    adata.varm[components_key] = position_factors
    adata.uns[uns_key] = {
        "rank": rank,
        "n_iter_max": n_iter_max,
        "random_state": random_state,
        "layer": layer,
        "components_key": components_key,
        "weights": weights,
        "base_factors": base_factors,
        "base_labels": list(bases),
        "backend": backend,
        "device": str(device),
    }

    logger.info(
        "Stored: adata.obsm['%s'], adata.varm['%s'], adata.uns['%s']",
        embedding_key,
        components_key,
        uns_key,
    )
    return adata
