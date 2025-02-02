## calculate_read_length_stats

# Read length QC
def calculate_read_length_stats(adata, reference_column='', sample_names_col=''):
    """
    Append first valid position in a read and last valid position in the read. From this determine and append the read length. 

    Parameters:
        adata (AnnData): An adata object
        reference_column (str): String representing the name of the Reference column to use
        sample_names_col (str): String representing the name of the sample name column to use
    
    Returns:
        upper_bound (int): last valid position in the dataset
        lower_bound (int): first valid position in the dataset
    """
    import numpy as np
    import anndata as ad
    import pandas as pd

    print('Calculating read length statistics')

    references = set(adata.obs[reference_column])
    sample_names = set(adata.obs[sample_names_col])

    ## Add basic observation-level (read-level) metadata to the object: first valid position in a read and last valid position in the read. From this determine the read length. Save two new variable which hold the first and last valid positions in the entire dataset
    print('calculating read length stats')
    # Add some basic observation-level (read-level) metadata to the anndata object
    read_first_valid_position = np.array([int(adata.var_names[i]) for i in np.argmax(~np.isnan(adata.X), axis=1)])
    read_last_valid_position = np.array([int(adata.var_names[i]) for i in (adata.X.shape[1] - 1 - np.argmax(~np.isnan(adata.X[:, ::-1]), axis=1))])
    read_length = read_last_valid_position - read_first_valid_position + np.ones(len(read_first_valid_position))

    adata.obs['first_valid_position'] = pd.Series(read_first_valid_position, index=adata.obs.index, dtype=int)
    adata.obs['last_valid_position'] = pd.Series(read_last_valid_position, index=adata.obs.index, dtype=int)
    adata.obs['read_length'] = pd.Series(read_length, index=adata.obs.index, dtype=int)

    # Define variables to hold the first and last valid position in the dataset
    upper_bound = int(np.nanmax(adata.obs['last_valid_position']))
    lower_bound = int(np.nanmin(adata.obs['first_valid_position']))

    return upper_bound, lower_bound

# # Add an unstructured element to the anndata object which points to a dictionary of read lengths keyed by reference and sample name. Points to a tuple containing (mean, median, stdev) of the read lengths of the sample for the given reference strand
#     ## Plot histogram of read length data and save the median and stdev of the read lengths for each sample.
#     adata.uns['read_length_dict'] = {}

#     for reference in references:
#         temp_reference_adata = adata[adata.obs[reference_column] == reference].copy()
#         split_reference = reference.split('_')[0][1:]
#         for sample in sample_names:
#             temp_sample_adata = temp_reference_adata[temp_reference_adata.obs[sample_names_col] == sample].copy()
#             temp_data = temp_sample_adata.obs['read_length']
#             max_length = np.max(temp_data)
#             mean = np.mean(temp_data)
#             median = np.median(temp_data)
#             stdev = np.std(temp_data)
#             adata.uns['read_length_dict'][f'{reference}_{sample}'] = [mean, median, stdev]
#             if not np.isnan(max_length):
#                 n_bins = int(max_length // 100)
#             else:
#                 n_bins = 1
#             if show_read_length_histogram or save_read_length_histogram:
#                 plt.figure(figsize=(10, 6))
#                 plt.text(median + 0.5, max(plt.hist(temp_data, bins=n_bins)[0]) / 2, f'Median: {median:.2f}', color='red')
#                 plt.hist(temp_data, bins=n_bins, alpha=0.7, color='blue', edgecolor='black')
#                 plt.xlabel('Read Length')
#                 plt.ylabel('Count')
#                 title = f'Read length distribution of {temp_sample_adata.shape[0]} total reads from {sample} sample on {split_reference} allele'
#                 plt.title(title)
#                 # Add a vertical line at the median
#                 plt.axvline(median, color='red', linestyle='dashed', linewidth=1)
#                 # Annotate the median
#                 plt.xlim(lower_bound - 100, upper_bound + 100) 
#                 if save_read_length_histogram:
#                     save_name = output_directory + f'/{readwrite.date_string()} {title}'
#                     plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
#                     plt.close()
#                 else:
#                     plt.show()