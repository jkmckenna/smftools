# recipes

def recipe_1_Kissiov_and_McKenna_2025(adata, sample_sheet_path, output_directory, mapping_key_column='Sample', reference_column = 'Reference', sample_names_col='Sample_names', invert=False):
    """
    The first part of the preprocessing workflow applied to the smf.inform.pod_to_adata() output derived from Kissiov_and_McKenna_2025.

    Performs the following tasks:
    1) Loads a sample CSV to append metadata mappings to the adata object.
    2) Appends a boolean indicating whether each position in var_names is within a given reference.
    3) Appends the cytosine context to each position from each reference.
    4) Calculate read level methylation statistics.
    5) Optionally inverts the adata to flip the position coordinate orientation.
    6) Calculates read length statistics (start position, end position, read length)
    7) Returns a dictionary to pass the variable namespace to the parent scope.

    Parameters:
        adata (AnnData): The AnnData object to use as input.
        sample_sheet_path (str): String representing the path to the sample sheet csv containing the sample metadata.
        output_directory (str): String representing the path to the output directory for plots.
        mapping_key_column (str): The column name to use as the mapping keys for applying the sample sheet metadata.
        reference_column (str): The name of the reference column to use.
        sample_names_col (str): The name of the sample name column to use.
        invert (bool): Whether to invert the positional coordinates of the adata object.

    Returns:
        variables (dict): A dictionary of variables to append to the parent scope.
    """
    import anndata as ad
    import pandas as pd
    import numpy as np
    from .load_sample_sheet import load_sample_sheet
    from .calculate_coverage import calculate_coverage
    from .append_C_context import append_C_context
    from .calculate_converted_read_methylation_stats import calculate_converted_read_methylation_stats
    from .invert_adata import invert_adata
    from .calculate_read_length_stats import calculate_read_length_stats

    # Clean up some of the Reference metadata and save variable names that point to sets of values in the column.
    adata.obs[reference_column] = adata.obs[reference_column].astype('category')
    references = adata.obs[reference_column].cat.categories
    split_references = [(reference, reference.split('_')[0][1:]) for reference in references]
    reference_mapping = {k: v for k, v in split_references}
    adata.obs[f'{reference_column}_short'] = adata.obs[reference_column].map(reference_mapping)
    short_references = set(adata.obs[f'{reference_column}_short'])
    binary_layers = adata.layers.keys()

    # load sample sheet metadata
    load_sample_sheet(adata, sample_sheet_path, mapping_key_column)

    # hold sample names set
    adata.obs[sample_names_col] = adata.obs[sample_names_col].astype('category')
    sample_names = adata.obs[sample_names_col].cat.categories

    # Add position level metadata
    calculate_coverage(adata, obs_column=reference_column)
    adata.var['SNP_position'] = (adata.var[f'N_{reference_column}_with_position'] > 0) & (adata.var[f'N_{reference_column}_with_position'] < len(references)).astype(bool)

    # Append cytosine context to the reference positions based on the conversion strand.
    append_C_context(adata, obs_column=reference_column, use_consensus=False)

    # Calculate read level methylation statistics. Assess if GpC methylation level is above other_C methylation level as a QC.
    calculate_converted_read_methylation_stats(adata, reference_column, sample_names_col, output_directory, show_methylation_histogram=False, save_methylation_histogram=False)

    # Invert the adata object (ie flip the strand orientation for visualization)
    if invert:
        invert_adata(adata)
    else:
        pass

    # Calculate read length statistics, with options to display or save the read length histograms
    upper_bound, lower_bound = calculate_read_length_stats(adata, reference_column, sample_names_col, output_directory, show_read_length_histogram=False, save_read_length_histogram=False)

    variables = {
        "short_references": short_references,
        "binary_layers": binary_layers,
        "sample_names": sample_names,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "references": references
    }
    return variables

def recipe_2_Kissiov_and_McKenna_2025(adata, output_directory, binary_layers, hamming_distance_thresholds={}, reference_column = 'Reference', sample_names_col='Sample_names'):
    """
    The second part of the preprocessing workflow applied to the adata that has already been preprocessed by recipe_1_Kissiov_and_McKenna_2025.

    Performs the following tasks:
    1) Adds new layers containing NaN replaced variants of adata.X (fill_closest, nan0_0minus1, nan1_12).
    2) Marks putative PCR duplicates using pairwise hamming distance metrics.
    3) Performs a complexity analysis of the library based on the PCR duplicate detection rate.
    4) Removes PCR duplicates from the adata.
    5) Returns two adata object: one for the filtered adata and one for the duplicate adata.

    Parameters:
        adata (AnnData): The AnnData object to use as input.
        output_directory (str): String representing the path to the output directory for plots.
        binary_layers (list): A list of layers to used for the binary encoding of read sequences. Used for duplicate detection.
        hamming_distance_thresholds (dict): A dictionary keyed by obs_column categories that points to a float corresponding to the distance threshold to apply. Default is an empty dict.
        reference_column (str): The name of the reference column to use.
        sample_names_col (str): The name of the sample name column to use.

    Returns:
        filtered_adata (AnnData): An AnnData object containing the filtered reads
        duplicates (AnnData): An AnnData object containing the duplicate reads
    """
    import anndata as ad
    import pandas as pd
    import numpy as np
    from .clean_NaN import clean_NaN
    from .mark_duplicates import mark_duplicates
    from .calculate_complexity import calculate_complexity
    from .remove_duplicates import remove_duplicates

    # NaN replacement strategies stored in additional layers. Having layer=None uses adata.X
    clean_NaN(adata, layer=None)

    # Duplicate detection using pairwise hamming distance across reads
    mark_duplicates(adata, binary_layers, obs_column=reference_column, sample_col=sample_names_col, hamming_distance_thresholds=hamming_distance_thresholds)

    # Complexity analysis using the marked duplicates and the lander-watermann algorithm
    calculate_complexity(adata, output_directory, obs_column=reference_column, sample_col=sample_names_col, plot=True, save_plot=False)

    # Remove duplicate reads and store the duplicate reads in a new AnnData object named duplicates.
    filtered_adata, duplicates = remove_duplicates(adata)
    return filtered_adata, duplicates
