## calculate_position_Youden

## Calculating and applying position level thresholds for methylation calls to binarize the SMF data
def calculate_position_Youden(adata, positive_control_sample='positive', negative_control_sample='negative', J_threshold=0.5, obs_column='Reference', infer_on_percentile=False, inference_variable='', save=False, output_directory=''):
    """
    Adds new variable metadata to each position indicating whether the position provides reliable SMF methylation calls. Also outputs plots of the positional ROC curves.

    Parameters:
        adata (AnnData): An AnnData object.
        positive_control_sample (str): string representing the sample name corresponding to the Plus MTase control sample.
        negative_control_sample (str): string representing the sample name corresponding to the Minus MTase control sample.
        J_threshold (float): A float indicating the J-statistic used to indicate whether a position passes QC for methylation calls.
        obs_column (str): The category to iterate over.
        infer_on_perdentile (bool | int): If False, use defined postive and negative control samples. If an int (0 < int < 100) is passed, this uses the top and bottom int percentile of methylated reads based on metric in inference_variable column.
        inference_variable (str): If infer_on_percentile has an integer value passed, use the AnnData observation column name passed by this string as the metric.
        save (bool): Whether to save the ROC plots.
        output_directory (str): String representing the path to the output directory to output the ROC curves.
    
    Returns: 
        None
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    control_samples = [positive_control_sample, negative_control_sample]
    categories = adata.obs[obs_column].cat.categories 
    # Iterate over each category in the specified obs_column
    for cat in categories:
        print(f"Calculating position Youden statistics for {cat}")
        # Subset to keep only reads associated with the category
        cat_subset = adata[adata.obs[obs_column] == cat]
        # Iterate over positive and negative control samples
        for control in control_samples:
            # Initialize a dictionary for the given control sample. This will be keyed by dataset and position to point to a tuple of coordinate position and an array of methylation probabilities
            adata.uns[f'{cat}_position_methylation_dict_{control}'] = {}
            if infer_on_percentile:
                sorted_column = cat_subset.obs[inference_variable].sort_values(ascending=False)
                if control == "positive":
                    threshold = np.percentile(sorted_column, 100 - infer_on_percentile)
                    control_subset = cat_subset[cat_subset.obs[inference_variable] >= threshold, :]
                else:
                    threshold = np.percentile(sorted_column, infer_on_percentile)
                    control_subset = cat_subset[cat_subset.obs[inference_variable] <= threshold, :]   
            else:
                # get the current control subset on the given category
                filtered_obs = cat_subset.obs[cat_subset.obs['Sample_names'].str.contains(control, na=False, regex=True)]
                control_subset = cat_subset[filtered_obs.index]
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
        save_name = output_directory + f'/{title}'
        if save:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()   

        adata.var[f'{cat}_position_methylation_thresholding_Youden_stats'] = probability_thresholding_list
        J_max_list = [probability_thresholding_list[i][1] for i in range(adata.shape[1])]
        adata.var[f'{cat}_position_passed_QC'] = [True if i > J_threshold else False for i in J_max_list]