## invert_adata
import numpy as np
import anndata as ad
import pandas as pd

# Optional inversion of the adata
def invert_adata(adata):
    """
    Input: An adata object
    Output: Inverts the adata object along the variable axis
    """
    # Reassign var_names with new names
    old_var_names = adata.var_names.astype(int).to_numpy()
    new_var_names = np.sort(old_var_names)[::-1].astype(str)
    adata.var['Original_positional_coordinate'] = old_var_names.astype(str)
    adata.var_names = new_var_names
    # Sort the AnnData object based on the old var_names
    adata = adata[:, old_var_names.astype(str)]