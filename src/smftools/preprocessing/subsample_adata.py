from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def subsample_adata(
    adata: "ad.AnnData",
    obs_columns: Sequence[str] | None = None,
    max_samples: int = 2000,
    random_seed: int = 42,
) -> "ad.AnnData":
    """Subsample an AnnData object by observation categories.

    Each unique combination of categories in ``obs_columns`` is capped at
    ``max_samples`` observations. If ``obs_columns`` is ``None``, the function
    randomly subsamples the entire dataset.

    Args:
        adata: AnnData object to subsample.
        obs_columns: Observation column names to group by.
        max_samples: Maximum observations per category combination.
        random_seed: Random seed for reproducibility.

    Returns:
        anndata.AnnData: Subsampled AnnData object.
    """
    import numpy as np

    np.random.seed(random_seed)  # Ensure reproducibility

    if not obs_columns:  # If no obs columns are given, sample globally
        if adata.shape[0] > max_samples:
            sampled_indices = np.random.choice(adata.obs.index, max_samples, replace=False)
        else:
            sampled_indices = adata.obs.index  # Keep all if fewer than max_samples

        return adata[sampled_indices].copy()

    sampled_indices = []

    # Get unique category combinations from all specified obs columns
    unique_combinations = adata.obs[obs_columns].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        # Build filter condition dynamically for multiple columns
        condition = (adata.obs[obs_columns] == row.values).all(axis=1)

        # Get indices for the current category combination
        subset_indices = adata.obs[condition].index.to_numpy()

        # Subsample or take all
        if len(subset_indices) > max_samples:
            sampled = np.random.choice(subset_indices, max_samples, replace=False)
        else:
            sampled = subset_indices  # Keep all if fewer than max_samples

        sampled_indices.extend(sampled)

    # ⚠ Handle backed mode detection
    if adata.isbacked:
        logger.warning("Detected backed mode. Subset will be loaded fully into memory.")
        subset = adata[sampled_indices]
        subset = subset.to_memory()
    else:
        subset = adata[sampled_indices]

    # Create a new AnnData object with only the selected indices
    return subset[sampled_indices].copy()
