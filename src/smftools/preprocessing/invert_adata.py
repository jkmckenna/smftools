## invert_adata

# Optional inversion of the adata

def invert_adata(adata):
    """
    Inverts the AnnData object along the column (variable) axis.

    Parameters:
        adata (AnnData): An AnnData object.

    Returns:
        AnnData: A new AnnData object with inverted column ordering.
    """
    import numpy as np
    import anndata as ad

    print("🔄 Inverting AnnData along the column axis...")

    # Reverse the order of columns (variables)
    inverted_adata = adata[:, ::-1].copy()

    # Reassign var_names with new order
    inverted_adata.var_names = adata.var_names

    # Optional: Store original coordinates for reference
    inverted_adata.var["Original_var_names"] = adata.var_names[::-1]

    print("✅ Inversion complete!")
    return inverted_adata
