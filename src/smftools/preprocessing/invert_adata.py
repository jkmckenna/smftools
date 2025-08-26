## invert_adata

def invert_adata(adata, uns_flag='adata_positions_inverted', force_redo=False):
    """
    Inverts the AnnData object along the column (variable) axis.

    Parameters:
        adata (AnnData): An AnnData object.

    Returns:
        AnnData: A new AnnData object with inverted column ordering.
    """
    import numpy as np
    import anndata as ad

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo):
        # QC already performed; nothing to do
        return adata

    print("Inverting AnnData along the column axis...")

    # Reverse the order of columns (variables)
    inverted_adata = adata[:, ::-1].copy()

    # Reassign var_names with new order
    inverted_adata.var_names = adata.var_names

    # Optional: Store original coordinates for reference
    inverted_adata.var["Original_var_names"] = adata.var_names[::-1]

    # mark as done
    inverted_adata.uns[uns_flag] = True

    print("Inversion complete!")
    return inverted_adata
