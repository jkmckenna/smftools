## reindex_references_adata

def reindex_references_adata(adata, 
                             reference_col="Reference_strand", 
                             offsets=None, 
                             new_col="reindexed"):
    if offsets is None:
        pass
    else:
        # Ensure var_names are numeric
        var_coords = adata.var_names.astype(int)

        for ref in adata.obs[reference_col].unique():
            if ref not in offsets:
                pass
            else:
                offset_value = offsets[ref]

                # Create a new var column for this reference
                colname = f"{ref}_{new_col}"

                # Add offset to all var positions
                adata.var[colname] = var_coords + offset_value

