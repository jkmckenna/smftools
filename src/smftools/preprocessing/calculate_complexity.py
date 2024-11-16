## calculate_complexity

def calculate_complexity(adata, output_directory='', obs_column='Reference', sample_col='Sample_names', plot=True, save_plot=False):
    """
    A complexity analysis of the library.

    Parameters:
        adata (AnnData): An adata object with mark_duplicates already run.
        output_directory (str): String representing the path to the output directory.
        obs_column (str): String of the obs column to iterate over.
        sample_col (str): String of the sample column to iterate over.
        plot (bool): Whether to plot the complexity model.
        save_plot (bool): Whether to save the complexity model.

    Returns:
        None

    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    def lander_waterman(x, C0):
        return C0 * (1 - np.exp(-x / C0))

    def count_unique_reads(reads, depth):
        subsample = np.random.choice(reads, depth, replace=False)
        return len(np.unique(subsample))

    categories = adata.obs[obs_column].cat.categories 
    sample_names = adata.obs[sample_col].cat.categories 

    for cat in categories:
        for sample in sample_names:
            unique_reads = adata.uns[f'Hamming_distance_cluster_count_within_{cat}_{sample}']
            total_reads = adata.uns[f'total_reads_within_{cat}_{sample}']
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
            adata.uns[f'Library_complexity_of_{sample}_on_{cat}'] = popt[0]
            if plot:
                import matplotlib.pyplot as plt
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
                    save_name = output_directory + f'/{date_string}_{title}'
                    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                else:
                    plt.show()