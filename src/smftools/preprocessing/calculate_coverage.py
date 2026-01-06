def calculate_coverage(
    adata,
    ref_column="Reference_strand",
    position_nan_threshold=0.01,
    smf_modality="deaminase",
    target_layer="binarized_methylation",
    uns_flag="calculate_coverage_performed",
    force_redo=False,
):
    """
    Append position-level metadata regarding whether the position is informative within the given observation category.

    Parameters:
        adata (AnnData): An AnnData object
        obs_column (str): Observation column value to subset on prior to calculating position statistics for that category.
        position_nan_threshold (float): A minimal fractional threshold of coverage within the obs_column category to call the position as valid.
        smf_modality (str): The smfmodality. For conversion/deaminase, use the adata.X. For direct, use the target_layer
        target_layer (str): The layer to use for direct smf coverage calculations

    Modifies:
        - Adds new columns to `adata.var` containing coverage statistics.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        # QC already performed; nothing to do
        return

    references = adata.obs[ref_column].cat.categories
    n_categories_with_position = np.zeros(adata.shape[1])

    # Loop over references
    for ref in references:
        print(f"Assessing positional coverage across samples for {ref} reference")

        # Subset to current category
        ref_mask = adata.obs[ref_column] == ref
        temp_ref_adata = adata[ref_mask]

        if smf_modality == "direct":
            matrix = temp_ref_adata.layers[target_layer]
        else:
            matrix = temp_ref_adata.X

        # Compute fraction of valid coverage
        ref_valid_coverage = np.sum(~np.isnan(matrix), axis=0)
        ref_valid_fraction = ref_valid_coverage / temp_ref_adata.shape[0]  # Avoid extra computation

        # Store coverage stats
        adata.var[f"{ref}_valid_count"] = pd.Series(ref_valid_coverage, index=adata.var.index)
        adata.var[f"{ref}_valid_fraction"] = pd.Series(ref_valid_fraction, index=adata.var.index)

        # Assign whether the position is covered based on threshold
        adata.var[f"position_in_{ref}"] = ref_valid_fraction >= position_nan_threshold

        # Sum the number of categories covering each position
        n_categories_with_position += adata.var[f"position_in_{ref}"].values

    # Store final category count
    adata.var[f"N_{ref_column}_with_position"] = n_categories_with_position.astype(int)

    # mark as done
    adata.uns[uns_flag] = True
