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