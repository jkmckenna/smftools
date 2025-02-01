# plot_read_length_and_coverage_histograms

def plot_read_length_and_coverage_histograms(bed_file, plotting_directory):
    """
    Plots read length and coverage statistics for each record.

    Parameters:
        bed_file (str): Path to the bed file to derive read lengths and coverage from.
        plot_directory (str): Path to the directory to write out historgrams.

    Returns:
        None
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    bed_basename = os.path.basename(bed_file).split('.bed')[0]
    # Load the BED file into a DataFrame
    print(f"Loading BED to plot read length and coverage histograms: {bed_file}")
    df = pd.read_csv(bed_file, sep='\t', header=None, names=['chromosome', 'start', 'end', 'length', 'read_name'])
    
    # Group by chromosome
    grouped = df.groupby('chromosome')

    for chrom, group in grouped:
        # Plot read length histogram
        plt.figure(figsize=(12, 6))
        plt.hist(group['length'], bins=50, edgecolor='k', alpha=0.7)
        plt.title(f'Read Length Histogram of reads aligned to {chrom}')
        plt.xlabel('Read Length')
        plt.ylabel('Count')
        plt.grid(True)
        save_name = os.path.join(plotting_directory, f'{bed_basename}_{chrom}_read_length_histogram.png')
        plt.savefig(save_name)
        plt.close()

        # Compute coverage
        coverage = np.zeros(group['end'].max())
        for _, row in group.iterrows():
            coverage[row['start']:row['end']] += 1
        
        # Plot coverage histogram
        plt.figure(figsize=(12, 6))
        plt.plot(coverage, color='b')
        plt.title(f'Coverage Histogram for {chrom}')
        plt.xlabel('Position')
        plt.ylabel('Coverage')
        plt.grid(True)
        save_name = os.path.join(plotting_directory, f'{bed_basename}_{chrom}_coverage_histogram.png')
        plt.savefig(save_name)
        plt.close()