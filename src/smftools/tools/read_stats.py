# ------------------------- Utilities -------------------------
def random_fill_nans(X):
    import numpy as np
    nan_mask = np.isnan(X)
    X[nan_mask] = np.random.rand(*X[nan_mask].shape)
    return X

def calculate_row_entropy(
    adata,
    layer,
    output_key="entropy",
    site_config=None,
    ref_col="Reference_strand",
    encoding="signed",
    max_threads=None):
    """
    Adds an obs column to the adata that calculates entropy within each read from a given layer
    when looking at each site type passed in the site_config list.

    Parameters:
        adata (AnnData): The annotated data matrix.
        layer (str): Name of the layer to use for entropy calculation.
        method (str): Unused currently. Placeholder for potential future methods.
        output_key (str): Base name for the entropy column in adata.obs.
        site_config (dict): {ref: [site_types]} for masking relevant sites.
        ref_col (str): Column in adata.obs denoting reference strands.
        encoding (str): 'signed' (1/-1/0) or 'binary' (1/0/NaN).
        max_threads (int): Number of threads for parallel processing.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import entropy
    from joblib import Parallel, delayed
    from tqdm import tqdm

    entropy_values = []
    row_indices = []

    for ref in adata.obs[ref_col].cat.categories:
        subset = adata[adata.obs[ref_col] == ref].copy()
        if subset.shape[0] == 0:
            continue

        if site_config and ref in site_config:
            site_mask = np.zeros(subset.shape[1], dtype=bool)
            for site in site_config[ref]:
                site_mask |= subset.var[f"{ref}_{site}"]
            subset = subset[:, site_mask].copy()

        X = subset.layers[layer].copy()

        if encoding == "signed":
            X_bin = np.where(X == 1, 1, np.where(X == -1, 0, np.nan))
        else:
            X_bin = np.where(X == 1, 1, np.where(X == 0, 0, np.nan))

        def compute_entropy(row):
            counts = pd.Series(row).value_counts(dropna=True).sort_index()
            probs = counts / counts.sum()
            return entropy(probs, base=2)

        entropies = Parallel(n_jobs=max_threads)(
            delayed(compute_entropy)(X_bin[i, :]) for i in tqdm(range(X_bin.shape[0]), desc=f"Entropy: {ref}")
        )

        entropy_values.extend(entropies)
        row_indices.extend(subset.obs_names.tolist())

    entropy_key = f"{output_key}_entropy"
    adata.obs.loc[row_indices, entropy_key] = entropy_values

def binary_autocorrelation_with_spacing(row, positions, max_lag=1000, assume_sorted=True):
    """
    Fast autocorrelation over real genomic spacing.
    Uses a sliding window + bincount to aggregate per-lag products.

    Parameters
    ----------
    row : 1D array (float)
        Values per position (NaN = missing). Works for binary or real-valued.
    positions : 1D array (int)
        Genomic coordinates for each column of `row`.
    max_lag : int
        Max genomic lag (inclusive).
    assume_sorted : bool
        If True, assumes `positions` are strictly non-decreasing.

    Returns
    -------
    autocorr : 1D array, shape (max_lag+1,)
        Normalized autocorrelation; autocorr[0] = 1.0.
        Lags with no valid pairs are NaN.
    """
    import numpy as np

    # mask valid entries
    valid = ~np.isnan(row)
    if valid.sum() < 2:
        return np.full(max_lag + 1, np.nan, dtype=np.float32)

    x = row[valid].astype(np.float64, copy=False)
    pos = positions[valid].astype(np.int64, copy=False)

    # sort by position if needed
    if not assume_sorted:
        order = np.argsort(pos, kind="mergesort")
        pos = pos[order]
        x = x[order]

    n = x.size
    x_mean = x.mean()
    xc = x - x_mean
    var = np.sum(xc * xc)
    if var == 0.0:
        return np.full(max_lag + 1, np.nan, dtype=np.float32)

    lag_sums = np.zeros(max_lag + 1, dtype=np.float64)
    lag_counts = np.zeros(max_lag + 1, dtype=np.int64)

    # sliding window upper pointer
    j = 1
    for i in range(n - 1):
        # advance j to include all positions within max_lag
        while j < n and pos[j] - pos[i] <= max_lag:
            j += 1
        # consider pairs (i, i+1...j-1)
        if j - i > 1:
            diffs = pos[i+1:j] - pos[i]                 # 1..max_lag
            contrib = xc[i] * xc[i+1:j]                 # contributions for each pair
            # accumulate weighted sums and counts per lag
            lag_sums[:max_lag+1] += np.bincount(diffs, weights=contrib,
                                                minlength=max_lag+1)[:max_lag+1]
            lag_counts[:max_lag+1] += np.bincount(diffs,
                                                  minlength=max_lag+1)[:max_lag+1]

    autocorr = np.full(max_lag + 1, np.nan, dtype=np.float64)
    nz = lag_counts > 0
    autocorr[nz] = lag_sums[nz] / var
    autocorr[0] = 1.0  # by definition

    return autocorr.astype(np.float32, copy=False)

# def binary_autocorrelation_with_spacing(row, positions, max_lag=1000):
#     """
#     Compute autocorrelation within a read using real genomic spacing from `positions`.
#     Only valid (non-NaN) positions are considered.
#     Output is indexed by genomic lag (up to max_lag).
#     """
#     from collections import defaultdict
#     import numpy as np
#     # Get valid positions and values
#     valid_mask = ~np.isnan(row)
#     x = row[valid_mask]
#     pos = positions[valid_mask]
#     n = len(x)

#     if n < 2:
#         return np.full(max_lag + 1, np.nan)

#     x_mean = x.mean()
#     var = np.sum((x - x_mean)**2)
#     if var == 0:
#         return np.full(max_lag + 1, np.nan)

#     # Collect values by lag
#     lag_sums = defaultdict(float)
#     lag_counts = defaultdict(int)

#     for i in range(n):
#         for j in range(i + 1, n):
#             lag = abs(pos[j] - pos[i])
#             if lag > max_lag:
#                 continue
#             product = (x[i] - x_mean) * (x[j] - x_mean)
#             lag_sums[lag] += product
#             lag_counts[lag] += 1

#     # Normalize to get autocorrelation
#     autocorr = np.full(max_lag + 1, np.nan)
#     for lag in range(max_lag + 1):
#         if lag_counts[lag] > 0:
#             autocorr[lag] = lag_sums[lag] / var

#     return autocorr
