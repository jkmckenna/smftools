## preprocessing
from .. import readwrite

# Clustering and stats
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import hamming
import networkx as nx

# Signal processing
from scipy.signal import find_peaks

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# User interface
from tqdm import tqdm

output_directory =''

######################################################################################################
## General SMF

def calculate_coverage(adata, obs_column='Reference', position_nan_threshold=0.05):
    """
    Input: An adata object and an observation column of interest. Assess if the position is present in the dataset category.
    Output: Append position level metadata indicating whether the position is informative within the given observation category.
    """
    categories = adata.obs[obs_column].cat.categories
    n_categories_with_position = np.zeros(adata.shape[1])
    # Loop over reference strands
    for cat in categories:
        # Look at positional information for each reference
        temp_cat_adata = adata[adata.obs[obs_column] == cat]
        # Look at read coverage on the given category strand
        cat_valid_coverage = np.sum(~np.isnan(temp_cat_adata.X), axis=0)
        cat_invalid_coverage = np.sum(np.isnan(temp_cat_adata.X), axis=0)
        cat_valid_fraction = cat_valid_coverage / (cat_valid_coverage + cat_invalid_coverage)
        # Append metadata for category to the anndata object
        adata.var[f'{cat}_valid_fraction'] = pd.Series(cat_valid_fraction, index=adata.var.index)
        # Characterize if the position is in the given category or not
        conditions = [
            (adata.var[f'{cat}_valid_fraction'] >= position_nan_threshold),
            (adata.var[f'{cat}_valid_fraction'] < position_nan_threshold)
        ]
        choices = [True, False]
        adata.var[f'position_in_{cat}'] = np.select(conditions, choices, default=False)
        n_categories_with_position += np.array(adata.var[f'position_in_{cat}'])

    # Final array with the sum at each position of the number of categories covering that position
    adata.var[f'N_{obs_column}_with_position'] = n_categories_with_position.astype(int)

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

# Read length QC
def calculate_read_length_stats(adata):
    """
    Input: An adata object
    Output: Append first valid position in a read and last valid position in the read. From this determine and append the read length. 
    Return two new variable which hold the first and last valid positions in the entire dataset
    """
    ## Add basic observation-level (read-level) metadata to the object: first valid position in a read and last valid position in the read. From this determine the read length. Save two new variable which hold the first and last valid positions in the entire dataset

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

def plot_read_length_QC(adata, lower_bound, upper_bound, obs_column='Reference', sample_col='Sample_names', save=False):
    """
    """
    categories = adata.obs[obs_column].cat.categories
    sample_names = adata.obs[sample_col].cat.categories
    ## Plot histogram of read length data and save the median and stdev of the read lengths for each sample.
    adata.uns['read_length_dict'] = {}
    for cat in categories:
        temp_cat_adata = adata[adata.obs[obs_column] == cat].copy()
        split_cat = cat.split('_')[0][1:]
        for sample in sample_names:
            temp_sample_adata = temp_cat_adata[temp_cat_adata.obs[sample_col] == sample].copy()
            temp_data = temp_sample_adata.obs['read_length']
            max_length = np.max(temp_data)
            mean = np.mean(temp_data)
            median = np.median(temp_data)
            stdev = np.std(temp_data)
            adata.uns['read_length_dict'][f'{cat}_{sample}'] = [mean, median, stdev]
            n_bins = int(max_length // 100)
            plt.figure(figsize=(10, 6))
            plt.text(median + 0.5, max(plt.hist(temp_data, bins=n_bins)[0]) / 2, f'Median: {median:.2f}', color='red')
            plt.hist(temp_data, bins=n_bins, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Read Length')
            plt.ylabel('Count')
            title = f'Read length distribution of {temp_sample_adata.shape[0]} total reads from {sample} sample on {split_cat} allele'
            plt.title(title)
            # Add a vertical line at the median
            plt.axvline(median, color='red', linestyle='dashed', linewidth=1)
            # Annotate the median
            plt.xlim(lower_bound - 100, upper_bound + 100) 
            if save:
                date_string = date_string()
                save_name = output_directory + f'/{date_string} {title}'
                plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
                plt.close()
            else:
                plt.show()

def filter_reads_on_length(adata, filter_on_coordinates=False, min_read_length=2700):
    """
    Input: Adata object. a list of lower and upper bound (set to False or None if not wanted), and a minimum read length integer.
    Output: Susbets the adata object to keep a defined coordinate window, as well as reads that are over a minimum threshold in length
    """

    if filter_on_coordinates:
        lower_bound, upper_bound = filter_on_coordinates
        # Extract the position information from the adata object as an np array
        var_names_arr = adata.var_names.astype(int).to_numpy()
        # Find the upper bound coordinate that is closest to the specified value
        closest_end_index = np.argmin(np.abs(var_names_arr - upper_bound))
        upper_bound = int(adata.var_names[closest_end_index])
        # Find the lower bound coordinate that is closest to the specified value
        closest_start_index = np.argmin(np.abs(var_names_arr - lower_bound))
        lower_bound = int(adata.var_names[closest_start_index])
        # Get a list of positional indexes that encompass the lower and upper bounds of the dataset
        position_list = list(range(lower_bound, upper_bound + 1))
        position_list = [str(pos) for pos in position_list]
        position_set = set(position_list)
        print(f'Subsetting adata to keep data between coordinates {lower_bound} and {upper_bound}')
        adata = adata[:, adata.var_names.isin(position_set)].copy()

    if min_read_length:
        print(f'Subsetting adata to keep reads longer than {min_read_length}')
        adata = adata[adata.obs['read_length'] > min_read_length].copy()

# NaN handling
def clean_NaN(adata, layer=None):
    """
    Input: An adata object and the layer to fill Nan values of
    Output: Append layers to adata that contain NaN cleaning strategies
    """
    # Fill NaN with closest SMF value
    df = adata_to_df(adata, layer=layer)
    df = df.ffill(axis=1).bfill(axis=1)
    adata.layers['fill_nans_closest'] = df.values

    # Replace NaN values with 0, and 0 with minus 1
    old_value, new_value = [0, -1]
    df = adata_to_df(adata, layer=layer)
    df = df.replace(old_value, new_value)
    old_value, new_value = [np.nan, 0]
    df = df.replace(old_value, new_value)
    adata.layers['nan0_0minus1'] = df.values

    # Replace NaN values with 1, and 1 with 2
    old_value, new_value = [1, 2]
    df = adata_to_df(adata, layer=layer)
    df = df.replace(old_value, new_value)
    old_value, new_value = [np.nan, 1]
    df = df.replace(old_value, new_value)
    adata.layers['nan1_12'] = df.values

######################################################################################################

######################################################################################################
## Conversion SMF Specific 
##############################################

# Read methylation QC
def append_C_context(adata, obs_column='Reference', use_consensus=False):
    """
    Input: An adata object, the obs_column of interst, and whether to use the consensus sequence from the category.
    Output: Adds Cytosine context to the position within the given category. When use_consensus is True, it uses the consensus sequence, otherwise it defaults to the FASTA sequence.
    """
    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_site', 'ambiguous_CpG_site', 'other_C']
    categories = adata.obs[obs_column].cat.categories
    if use_consensus:
        sequence = adata.uns[f'{cat}_consensus_sequence']
    else:
        sequence = adata.uns[f'{cat}_FASTA_sequence']
    for cat in categories:
        boolean_dict = {}
        for site_type in site_types:
            boolean_dict[f'{cat}_{site_type}'] = np.full(len(sequence), False, dtype=bool)
        # Iterate through the sequence and apply the criteria
        for i in range(1, len(sequence) - 1):
            if sequence[i] == 'C':
                if sequence[i - 1] == 'G' and sequence[i + 1] != 'G':
                    boolean_dict[f'{cat}_GpC_site'][i] = True
                elif sequence[i - 1] == 'G' and sequence[i + 1] == 'G':
                    boolean_dict[f'{cat}_ambiguous_GpC_site'][i] = True
                elif sequence[i - 1] != 'G' and sequence[i + 1] == 'G':
                    boolean_dict[f'{cat}_CpG_site'][i] = True
                elif sequence[i - 1] == 'G' and sequence[i + 1] == 'G':
                    boolean_dict[f'{cat}_ambiguous_CpG_site'][i] = True
                elif sequence[i - 1] != 'G' and sequence[i + 1] != 'G':
                    boolean_dict[f'{cat}_other_C'][i] = True
        for site_type in site_types:
            adata.var[f'{cat}_{site_type}'] = boolean_dict[f'{cat}_{site_type}'].astype(bool)
            adata.obsm[f'{cat}_{site_type}'] = adata[:, adata.var[f'{cat}_{site_type}'] == True].copy().X

def calculate_read_methylation_stats(adata, obs_column='Reference'):
    """
    Input: adata and the observation category of interest
    Output: Adds methylation statistics for each read. Indicates whether the read GpC methylation exceeded other_C methylation (background false positives)
    """
    site_types = ['GpC_site', 'CpG_site', 'ambiguous_GpC_site', 'ambiguous_CpG_site', 'other_C']
    categories = adata.obs[obs_column].cat.categories
    for site_type in site_types:
        adata.obs[f'{site_type}_row_methylation_sums'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'{site_type}_row_methylation_means'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
        adata.obs[f'number_valid_{site_type}_in_read'] = pd.Series(0, index=adata.obs_names, dtype=int)
        adata.obs[f'fraction_valid_{site_type}_in_range'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    for cat in categories:
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        for site_type in site_types:
            print(f'Iterating over {cat}_{site_type}')
            observation_matrix = cat_subset.obsm[f'{cat}_{site_type}']
            number_valid_positions_in_read = np.nansum(~np.isnan(observation_matrix), axis=1)
            row_methylation_sums = np.nansum(observation_matrix, axis=1)
            number_valid_positions_in_read[number_valid_positions_in_read == 0] = 1
            fraction_valid_positions_in_range = number_valid_positions_in_read / np.max(number_valid_positions_in_read)
            row_methylation_means = np.divide(row_methylation_sums, number_valid_positions_in_read)
            temp_obs_data = pd.DataFrame({f'number_valid_{site_type}_in_read': number_valid_positions_in_read,
                                        f'fraction_valid_{site_type}_in_range': fraction_valid_positions_in_range,
                                        f'{site_type}_row_methylation_sums': row_methylation_sums,
                                        f'{site_type}_row_methylation_means': row_methylation_means}, index=cat_subset.obs.index)
            adata.obs.update(temp_obs_data)
    # Indicate whether the read-level GpC methylation rate exceeds the false methylation rate of the read
    pass_array = np.array(adata.obs[f'GpC_site_row_methylation_means'] > adata.obs[f'other_C_row_methylation_means'])
    adata.obs['GpC_above_other_C'] = pd.Series(pass_array, index=adata.obs.index, dtype=bool)

def filter_reads_on_methylation(adata, valid_SMF_site_threshold=0.8, min_SMF_threshold=0.025):
    """
    Input: Adata object. Minimum thresholds for valid SMF site fraction in read, as well as minimum methylation content in read
    Output: A subset of the adata object
    """
    if valid_SMF_site_threshold:
        # Keep reads that have over a given valid GpC site content
        adata = adata[adata.obs['fraction_valid_GpC_site_in_range'] > valid_SMF_site_threshold].copy()
    if min_SMF_threshold:
        # Keep reads with SMF methylation over background methylation.
        adata = adata[adata.obs['GpC_above_other_C'] == True].copy()
        # Keep reads over a defined methylation threshold
        adata = adata[adata.obs['GpC_site_row_methylation_means'] > min_SMF_threshold].copy()

# PCR duplicate detection and complexity analysis.
def binary_layers_to_ohe(adata, layers, stack='hstack'):
    """
    Input: An adata object and a list of layers containing a binary encoding.
    Output: A dictionary keyed by obs_name that points to a stacked (hstack or vstack) one-hot encoding of the binary layers
    """
    # Extract the layers
    layers = [adata.layers[layer_name] for layer_name in layers]
    n_reads = layers[0].shape[0]
    ohe_dict = {}
    for i in range(n_reads):
        read_ohe = []
        for layer in layers:
            read_ohe.append(layer[i])
        read_name = adata.obs_names[i]
        if stack == 'hstack':
            ohe_dict[read_name] = np.hstack(read_ohe)
        elif stack == 'vstack':
            ohe_dict[read_name] = np.vstack(read_ohe)
    return ohe_dict

def calculate_pairwise_hamming_distances(arrays):
    """
    Calculate the pairwise Hamming distances for a list of ndarrays.
    Input: A list of ndarrays
    Output: a 2D array containing the pairwise Hamming distances.
    """
    num_arrays = len(arrays)
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((num_arrays, num_arrays))
    # Calculate pairwise distances with progress bar
    for i in tqdm(range(num_arrays), desc="Calculating Hamming Distances"):
        for j in range(i + 1, num_arrays):
            distance = hamming(arrays[i], arrays[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

def min_non_diagonal(matrix):
    """
    Takes a matrix and returns the smallest value from each row with the diagonal masked
    Input: A data matrix
    Output: A list of minimum values from each row of the matrix
    """
    n = matrix.shape[0]
    min_values = []
    for i in range(n):
        # Mask to exclude the diagonal element
        row_mask = np.ones(n, dtype=bool)
        row_mask[i] = False
        # Extract the row excluding the diagonal element
        row = matrix[i, row_mask]
        # Find the minimum value in the row
        min_values.append(np.min(row))
    return min_values

def lander_waterman(x, C0):
    return C0 * (1 - np.exp(-x / C0))

def count_unique_reads(reads, depth):
    subsample = np.random.choice(reads, depth, replace=False)
    return len(np.unique(subsample))

def mark_duplicates(adata, layers, obs_column='Reference', sample_col='Sample_names'):
    """
    Input: adata object, list of binary layers, column names to use.
    Output: Marks duplicates in the adata object
    """
    categories = adata.obs[obs_column].cat.categories 
    sample_names = adata.obs[sample_col].cat.categories 

    # Calculate the pairwise Hamming distances within each reference/sample set. Determine distance thresholds for each reference/sample pair
    adata.obs['Nearest_neighbor_Hamming_distance'] = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    for cat in categories:
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        for sample in sample_names:
            sample_subset = cat_subset[cat_subset.obs[sample_col] == sample].copy()
            # Encode sequencing reads as a one-hot-encodings
            adata.uns[f'{cat}_{sample}_read_OHE_dict'] = binary_layers_to_ohe(sample_subset, layers, stack='hstack')
            # Unpack the read names and one hot encodings into lists
            read_names = []
            ohe_list = []
            for read_name, ohe in adata.uns[f'{cat}_{sample}_read_OHE_dict'].items():
                read_names.append(read_name)
                ohe_list.append(ohe)
            # Calculate the pairwise hamming distances
            print(f'Calculating hamming distances for {sample} on {cat} allele')
            distance_matrix = calculate_pairwise_hamming_distances(ohe_list)
            n_reads = distance_matrix.shape[0]
            # Load the hamming matrix into a dataframe with index and column names as the read_names
            distance_df = pd.DataFrame(distance_matrix, index=read_names, columns=read_names)
            # Save the distance dataframe into an unstructured component of the adata object
            adata.uns[f'Pairwise_Hamming_distance_within_{cat}_{sample}'] = distance_df
            # Calculate the minimum non-self distance for every read in the reference and sample
            min_distance_values = min_non_diagonal(distance_matrix)
            min_distance_df = pd.DataFrame({'Nearest_neighbor_Hamming_distance': min_distance_values}, index=read_names)
            adata.obs.update(min_distance_df)
            # Generate a histogram of minimum non-self distances for each read
            min_distance_bins = plt.hist(min_distance_values, bins=n_reads//4)
            # Normalize the max value in any histogram bin to 1
            normalized_min_distance_counts = min_distance_bins[0] / np.max(min_distance_bins[0])
            # Extract the bin index of peak centers in the histogram
            peak_centers, _ = find_peaks(normalized_min_distance_counts, prominence=0.2, distance=5)
            first_peak_index = peak_centers[0]
            offset_index = first_peak_index-1
            # Use the distance corresponding to the first peak as the threshold distance in graph construction
            first_peak_distance = min_distance_bins[1][first_peak_index]
            offset_distance = min_distance_bins[1][offset_index]
            adata.uns[f'Hamming_distance_threshold_for_{cat}_{sample}'] = offset_distance

    ## Detect likely duplicate reads and mark them in the adata object.
    adata.obs['Marked_duplicate'] = pd.Series(False, index=adata.obs_names, dtype=bool)
    adata.obs['Unique_in_final_read_set'] = pd.Series(False, index=adata.obs_names, dtype=bool)
    adata.obs[f'Hamming_distance_cluster_within_{obs_column}_and_sample'] = pd.Series(-1, index=adata.obs_names, dtype=int)

    for cat in categories:
        for sample in sample_names:
            distance_df = adata.uns[f'Pairwise_Hamming_distance_within_{cat}_{sample}']
            read_names = distance_df.index
            distance_matrix = distance_df.values
            n_reads = distance_matrix.shape[0]
            distance_threshold = adata.uns[f'Hamming_distance_threshold_for_{cat}_{sample}']
            # Initialize the read distance graph
            G = nx.Graph()
            # Add each read as a node to the graph
            G.add_nodes_from(range(n_reads))
            # Add edges based on the threshold
            for i in range(n_reads):
                for j in range(i + 1, n_reads):
                    if distance_matrix[i, j] <= distance_threshold:
                        G.add_edge(i, j)        
            # Determine distinct clusters using connected components
            clusters = list(nx.connected_components(G))
            clusters = [list(cluster) for cluster in clusters]
            # Get the number of clusters
            cluster_count = len(clusters)
            adata.uns[f'Hamming_distance_clusters_within_{cat}_{sample}'] = [cluster_count, n_reads, cluster_count / n_reads, clusters]
            # Update the adata object
            read_cluster_map = {}
            read_duplicate_map = {}
            read_keep_map = {}
            for i, cluster in enumerate(clusters):
                for j, read_index in enumerate(cluster):
                    read_name = read_names[read_index]
                    read_cluster_map[read_name] = i
                    if len(cluster) > 1:
                        read_duplicate_map[read_name] = True
                        if j == 0:
                            read_keep_map[read_name] = True
                        else:
                            read_keep_map[read_name] = False
                    elif len(cluster) == 1:
                        read_duplicate_map[read_name] = False
                        read_keep_map[read_name] = True
            cluster_df = pd.DataFrame.from_dict(read_cluster_map, orient='index', columns=[f'Hamming_distance_cluster_within_{obs_column}_and_sample'], dtype=int)
            duplicate_df = pd.DataFrame.from_dict(read_duplicate_map, orient='index', columns=['Marked_duplicate'], dtype=bool)
            keep_df = pd.DataFrame.from_dict(read_keep_map, orient='index', columns=['Unique_in_final_read_set'], dtype=bool)
            df_combined = pd.concat([cluster_df, duplicate_df, keep_df], axis=1)
            adata.obs.update(df_combined)
            adata.obs['Marked_duplicate'] = adata.obs['Marked_duplicate'].astype(bool)
            adata.obs['Unique_in_final_read_set'] = adata.obs['Unique_in_final_read_set'].astype(bool)
            print(f'Hamming clusters for {sample} on {cat}\nThreshold: {first_peak_distance}\nNumber clusters: {cluster_count}\nNumber reads: {n_reads}\nFraction unique: {cluster_count / n_reads}')   

def plot_complexity(adata, obs_column='Reference', sample_col='Sample_names', plot=True, save_plot=False):
    """
    Input: adata object with mark_duplicates already run.
    Output: A complexity analysis of the library
    """
    categories = adata.obs[obs_column].cat.categories 
    sample_names = adata.obs[sample_col].cat.categories 

    for cat in categories:
        for sample in sample_names:
            unique_reads, total_reads = adata.uns[f'Hamming_distance_clusters_within_{cat}_{sample}'][0:2]
            reads = np.concatenate((np.arange(unique_reads), np.random.choice(unique_reads, total_reads - unique_reads, replace=True)))
            # Subsampling depths
            subsampling_depths = [total_reads // (i+1) for i in range(10)]
            # Arrays to store results
            subsampled_total_reads = []
            subsampled_unique_reads = []
            # Perform subsampling
            for depth in subsampling_depths:
                unique_count = count_unique_reads(reads, depth)
                subsampled_total_reads.append(depth)
                subsampled_unique_reads.append(unique_count)
            # Fit the Lander-Waterman model to the data
            popt, _ = curve_fit(lander_waterman, subsampled_total_reads, subsampled_unique_reads)
            # Generate data for the complexity curve
            x_data = np.linspace(0, 5000, 100)
            y_data = lander_waterman(x_data, *popt)
            adata.uns[f'Library_complexity_{sample}_on_{cat}'] = popt[0]
            if plot:
                # Plot the complexity curve
                plt.figure(figsize=(6, 4))
                plt.plot(total_reads, unique_reads, 'o', label='Observed unique reads')
                plt.plot(x_data, y_data, '-', label=f'Lander-Waterman fit\nEstimated C0 = {popt[0]:.2f}')
                plt.xlabel('Total number of reads')
                plt.ylabel('Number of unique reads')
                title = f'Library Complexity Analysis for {sample} on {cat}'
                plt.title(title)
                plt.legend()
                plt.grid(True)
                if save_plot:
                    date_string = date_string()
                    save_name = output_directory + f'/{date_string} {title}'
                    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                else:
                    plt.show()

def remove_duplicates(adata):
    """
    Input: adata object with marked duplicates
    Output: Remove duplicates from the adata object
    """
    initial_size = adata.shape[0]
    adata = adata[adata.obs['Unique_in_final_read_set'] == True].copy()
    final_size = adata.shape[0]
    print(f'Removed {initial_size-final_size} reads from the dataset')
######################################################################################################

######################################################################################################
## Direct methylation SMF Specific 
##############################################
## Calculating and applying position level thresholds for methylation calls to binarize the SMF data
def calculate_position_Youden(adata, positive_control_sample, negative_control_sample, J_threshold=0.4, obs_column='Reference', save=False):
    """
    Input: An adata object, a plus MTase control, a minus MTase control, the minimal J-statistic threshold, and a categorical observation column to iterate over.
    Input notes: The control samples are passed as string names of the samples as they appear in the 'Sample_names' obs column
    Output: Adds new variable metadata to each position indicating whether the position provides reliable SMF methylation calls. Also outputs plots of the positional ROC curves.
    Can optionally save the output plots of the ROC curve
    """
    control_samples = [positive_control_sample, negative_control_sample]
    categories = adata.obs[obs_column].cat.categories 
    # Iterate over each category in the specified obs_column
    for cat in categories:
        # Subset to keep only reads associated with the category
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        # Iterate over positive and negative control samples
        for control in control_samples:
            # Initialize a dictionary for the given control sample. This will be keyed by dataset and position to point to a tuple of coordinate position and an array of methylation probabilities
            adata.uns[f'{cat}_position_methylation_dict_{control}'] = {}
            # get the current control subset on the given category
            filtered_obs = cat_subset.obs[cat_subset.obs['Sample_names'].str.contains(control, na=False, regex=True)]
            control_subset = cat_subset[filtered_obs.index].copy()
            # Iterate through every position in the control subset
            for position in range(control_subset.shape[1]):
                # Get the coordinate name associated with that position
                coordinate = control_subset.var_names[position]
                # Get the array of methlyation probabilities for each read in the subset at that position
                position_data = control_subset.X[:, position]
                # Get the indexes of everywhere that is not a nan value
                nan_mask = ~np.isnan(position_data)
                # Keep only the methlyation data that has real values
                position_data = position_data[nan_mask]
                # Get the position data coverage
                position_coverage = len(position_data)
                # Get fraction coverage
                fraction_coverage = position_coverage / control_subset.shape[0]
                # Save the position and the position methylation data for the control subset
                adata.uns[f'{cat}_position_methylation_dict_{control}'][f'{position}'] = (position, position_data, fraction_coverage)

    for cat in categories:
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        n_passed_positions = 0
        n_total_positions = 0
        # Initialize a list that will hold the positional thresholds for the category
        probability_thresholding_list = [(np.nan, np.nan)] * adata.shape[1]
        for i, key in enumerate(adata.uns[f'{cat}_position_methylation_dict_{positive_control_sample}'].keys()):
            position = int(adata.uns[f'{cat}_position_methylation_dict_{positive_control_sample}'][key][0])
            positive_position_array = adata.uns[f'{cat}_position_methylation_dict_{positive_control_sample}'][key][1]
            fraction_coverage = adata.uns[f'{cat}_position_methylation_dict_{positive_control_sample}'][key][2]
            if fraction_coverage > 0.2:
                try:
                    negative_position_array = adata.uns[f'{cat}_position_methylation_dict_{negative_control_sample}'][key][1] 
                    # Combine the negative and positive control data
                    data = np.concatenate([negative_position_array, positive_position_array])
                    labels = np.array([0] * len(negative_position_array) + [1] * len(positive_position_array))
                    # Calculate the ROC curve
                    fpr, tpr, thresholds = roc_curve(labels, data)
                    # Calculate Youden's J statistic
                    J = tpr - fpr
                    optimal_idx = np.argmax(J)
                    optimal_threshold = thresholds[optimal_idx]
                    max_J = np.max(J)
                    data_tuple = (optimal_threshold, max_J)
                    probability_thresholding_list[position] = data_tuple
                    n_total_positions += 1
                    if max_J > J_threshold:
                        n_passed_positions += 1
                        plt.plot(fpr, tpr, label='ROC curve')
                except:
                    probability_thresholding_list[position] = (0.8, np.nan)
        title = f'ROC Curve for {n_passed_positions} positions with J-stat greater than {J_threshold}\n out of {n_total_positions} total positions on {cat}'
        plt.title(title)
        date_string = date_string()
        save_name = output_directory + f'/{date_string} {title}'
        if save:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()    
        adata.var[f'{cat}_position_methylation_thresholding_Youden_stats'] = probability_thresholding_list
        J_max_list = [probability_thresholding_list[i][1] for i in range(adata.shape[1])]
        adata.var[f'{cat}_position_passed_QC'] = [True if i > J_threshold else False for i in J_max_list]

def binarize_on_Youden(adata, obs_column='Reference'):
    """
    Input: adata object that has had calculate_position_Youden called on it.
    Output: Add a new layer to the adata object that has binarized SMF values based on the position thresholds determined by calculate_position_Youden
    """
    temp_adata = None
    categories = adata.obs[obs_column].cat.categories 
    for cat in categories:
        # Get the category subset
        cat_subset = adata[adata.obs[obs_column] == cat].copy()
        # extract the probability matrix for the category subset
        original_matrix = cat_subset.X
        # extract the learned methylation call thresholds for each position in the category.
        thresholds = [cat_subset.var[f'{cat}_position_methylation_thresholding_Youden_stats'][i][0] for i in range(cat_subset.shape[1])]
        # In the original matrix, get all positions that are nan values
        nan_mask = np.isnan(original_matrix)
        # Binarize the matrix on the new thresholds
        binarized_matrix = (original_matrix > thresholds).astype(float)
        # At the original positions that had nan values, replace the values with nans again
        binarized_matrix[nan_mask] = np.nan
        # Make a new layer for the reference that contains the binarized methylation calls
        cat_subset.layers['binarized_methylation'] = binarized_matrix
        if temp_adata:
            # If temp_data already exists, concatenate
            temp_adata = ad.concat([temp_adata, cat_subset], join='outer', index_unique=None).copy()
        else:
            # If temp_adata is still None, initialize temp_adata with reference_subset
            temp_adata = cat_subset.copy()

    # Sort the temp adata on the index names of the primary adata
    temp_adata = temp_adata[adata.obs_names].copy()
    # Pull back the new binarized layers into the original adata object
    adata.layers['binarized_methylation'] = temp_adata.layers['binarized_methylation']

######################################################################################################