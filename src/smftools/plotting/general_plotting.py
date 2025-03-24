def combined_hmm_raw_clustermap(adata, sample_col='Sample_Names', hmm_feature_layer="hmm_combined", layer_gpc="nan0_0minus1", layer_cpg="nan0_0minus1", cmap_hmm="tab10", cmap_gpc="coolwarm", cmap_cpg="viridis", min_quality=20, min_length=2700, sample_mapping=None):

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as sch
    import pandas as pd
    import anndata as ad

    # Loop through samples and references
    for ref in adata.obs["Reference_strand"].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            try:
                # Filter the data for the current combination of enzyme, reference, time, quality, and length
                subset = adata[(adata.obs['Reference_strand'] == ref) & 
                            (adata.obs[sample_col] == sample) &
                            (adata.obs['query_read_quality'] >= min_quality) &
                            (adata.obs['read_length'] >= min_length) &
                            (adata.obs['Raw_methylation_signal'] >= 20) 
                            ]

                # Skip if no reads pass the filter
                if subset.shape[0] == 0:
                    print(f"  ❌ No reads left after filtering for {sample} - {ref}")
                    continue

                # Define bins filtering
                bins = {
                    "Both Closed": (subset.obs["Enhancer_Open"] == False) & (subset.obs["Promoter_Open"] == False),
                    "Enhancer Only": (subset.obs["Enhancer_Open"] == True) & (subset.obs["Promoter_Open"] == False),
                    "Promoter Only": (subset.obs["Enhancer_Open"] == False) & (subset.obs["Promoter_Open"] == True),
                    "Both Open": (subset.obs["Enhancer_Open"] == True) & (subset.obs["Promoter_Open"] == True),
                }

                # Get valid GpC and CpG sites
                gpc_sites = subset.var.index[subset.var[f"{ref}_GpC_site"]].astype(int)
                cpg_sites = subset.var.index[subset.var[f"{ref}_CpG_site"]].astype(int)

                # Collect all data for stacked heatmap
                stacked_hmm_feature, stacked_gpc, stacked_cpg = [], [], []
                row_labels, bin_labels = [], []
                bin_boundaries = []  # To store row boundaries for horizontal lines

                total_reads = subset.shape[0]
                percentages = {}
                last_idx = 0  # Keeps track of row indices for bin boundaries

                for bin_label, bin_filter in bins.items():
                    subset_bin = subset[bin_filter].copy()
                    num_reads = subset_bin.shape[0]
                    percent_reads = (num_reads / total_reads) * 100 if total_reads > 0 else 0
                    percentages[bin_label] = percent_reads

                    if num_reads > 0:
                        # Cluster within each bin for large patch
                        linkage = sch.linkage(subset_bin.layers[layer_gpc], method="ward")
                        order = sch.leaves_list(linkage)

                        # Reorder matrices
                        stacked_hmm_feature.append(subset_bin[order].layers[hmm_feature_layer])
                        stacked_gpc.append(subset_bin[order][:, gpc_sites].layers[layer_gpc])
                        stacked_cpg.append(subset_bin[order][:, cpg_sites].layers[layer_cpg])

                        row_labels.extend([bin_label] * num_reads)
                        bin_labels.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")

                        # Store boundary row index
                        last_idx += num_reads
                        bin_boundaries.append(last_idx)

                # Stack matrices
                if stacked_hmm_feature:
                    hmm_matrix = np.vstack(stacked_hmm_feature)
                    gpc_matrix = np.vstack(stacked_gpc)
                    cpg_matrix = np.vstack(stacked_cpg)

                    # Plot heatmaps
                    fig, axes = plt.subplots(1, 3, figsize=(18, 10), gridspec_kw={'width_ratios': [8, 8, 8]})

                    # HMM Heatmap
                    sns.heatmap(hmm_matrix, cmap=cmap_hmm, ax=axes[0], xticklabels=150, yticklabels=False)
                    axes[0].set_title(f"{sample} - {ref} HMM Features")

                    # Draw horizontal black lines for bin boundaries
                    for boundary in bin_boundaries[:-1]:  # Exclude last index
                        axes[0].axhline(y=boundary, color="black", linewidth=2)

                    # GpC Methylation Heatmap
                    sns.heatmap(gpc_matrix, cmap=cmap_gpc, ax=axes[1], xticklabels=gpc_sites[::5], yticklabels=False)
                    axes[1].set_xticks(range(0, len(gpc_sites), 5))  # Set tick positions
                    axes[1].set_xticklabels(gpc_sites[::5], rotation=90, fontsize=10)
                    axes[1].set_title(f"{sample} - {ref} GpC Methylation")

                    # Draw horizontal black lines for bin boundaries
                    for boundary in bin_boundaries[:-1]:
                        axes[1].axhline(y=boundary, color="black", linewidth=2)

                    # CpG Methylation Heatmap
                    sns.heatmap(cpg_matrix, cmap=cmap_cpg, ax=axes[2], xticklabels=cpg_sites, yticklabels=False)
                    axes[2].set_xticklabels(cpg_sites, rotation=90, fontsize=10)
                    axes[2].set_title(f"{sample} - {ref} CpG Methylation")

                    # Draw horizontal black lines for bin boundaries
                    for boundary in bin_boundaries[:-1]:
                        axes[2].axhline(y=boundary, color="black", linewidth=2)

                    plt.tight_layout()
                    plt.show()

                    # Print summary statistics
                    print(f"📊 Summary for {sample} - {ref}:")
                    for bin_label, percent in percentages.items():
                        print(f"  - {bin_label}: {percent:.1f}%")

            except Exception as e:
                print(f"❌ Error processing {sample} - {ref}: {e}")
                continue
