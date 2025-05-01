import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_barplot(ax, mean_values, title):
    x = np.arange(len(mean_values))
    ax.bar(x, mean_values, color="gray", width=1.0, align='edge')
    ax.set_xlim(0, len(mean_values))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Mean")
    ax.set_title(title, fontsize=12, pad=2)

    # Hide all spines except left
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == 'left')

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


def combined_hmm_raw_clustermap(
    adata,
    sample_col='Sample_Names',
    hmm_feature_layer="hmm_combined",
    layer_gpc="nan0_0minus1",
    layer_cpg="nan0_0minus1",
    cmap_hmm="tab10",
    cmap_gpc="coolwarm",
    cmap_cpg="viridis",
    min_quality=20,
    min_length=2700,
    sample_mapping=None,
    save_path=None,
    normalize_hmm=False,
    sort_by="gpc",  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
    bins=None
):
    import scipy.cluster.hierarchy as sch
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    results = []

    for ref in adata.obs["Reference_strand"].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            try:
                subset = adata[
                    (adata.obs['Reference_strand'] == ref) &
                    (adata.obs[sample_col] == sample) &
                    (adata.obs['query_read_quality'] >= min_quality) &
                    (adata.obs['read_length'] >= min_length) &
                    (adata.obs['Raw_methylation_signal'] >= 20)
                ]
                
                subset = subset[:, subset.var[f'position_in_{ref}'] == True]

                if subset.shape[0] == 0:
                    print(f"  ❌ No reads left after filtering for {sample} - {ref}")
                    continue

                if bins:
                    pass
                else:
                    bins = {"All": (subset.obs['Reference_strand'] != None)}

                # Get column positions (not var_names!) of site masks
                gpc_sites = np.where(subset.var[f"{ref}_GpC_site"].values)[0]
                cpg_sites = np.where(subset.var[f"{ref}_CpG_site"].values)[0]

                # Use var_names for x-axis tick labels
                gpc_labels = subset.var_names[gpc_sites].astype(int)
                cpg_labels = subset.var_names[cpg_sites].astype(int)

                stacked_hmm_feature, stacked_gpc, stacked_cpg = [], [], []
                row_labels, bin_labels = [], []
                bin_boundaries = []

                total_reads = subset.shape[0]
                percentages = {}
                last_idx = 0

                for bin_label, bin_filter in bins.items():
                    subset_bin = subset[bin_filter].copy()
                    num_reads = subset_bin.shape[0]
                    percent_reads = (num_reads / total_reads) * 100 if total_reads > 0 else 0
                    percentages[bin_label] = percent_reads

                    if num_reads > 0:
                        # Determine sorting order
                        if sort_by.startswith("obs:"):
                            colname = sort_by.split("obs:")[1]
                            order = np.argsort(subset_bin.obs[colname].values)
                        elif sort_by == "gpc":
                            linkage = sch.linkage(subset_bin[:, gpc_sites].layers[layer_gpc], method="ward")
                            order = sch.leaves_list(linkage)
                        elif sort_by == "cpg":
                            linkage = sch.linkage(subset_bin[:, cpg_sites].layers[layer_cpg], method="ward")
                            order = sch.leaves_list(linkage)
                        elif sort_by == "gpc_cpg":
                            linkage = sch.linkage(subset_bin.layers[layer_gpc], method="ward")
                            order = sch.leaves_list(linkage)
                        elif sort_by == "none":
                            order = np.arange(num_reads)
                        else:
                            raise ValueError(f"Unsupported sort_by option: {sort_by}")

                        stacked_hmm_feature.append(subset_bin[order].layers[hmm_feature_layer])
                        stacked_gpc.append(subset_bin[order][:, gpc_sites].layers[layer_gpc])
                        stacked_cpg.append(subset_bin[order][:, cpg_sites].layers[layer_cpg])

                        row_labels.extend([bin_label] * num_reads)
                        bin_labels.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")
                        last_idx += num_reads
                        bin_boundaries.append(last_idx)

                if stacked_hmm_feature:
                    hmm_matrix = np.vstack(stacked_hmm_feature)
                    gpc_matrix = np.vstack(stacked_gpc)
                    cpg_matrix = np.vstack(stacked_cpg)

                    def normalized_mean(matrix):
                        mean = np.nanmean(matrix, axis=0)
                        normalized = (mean - mean.min()) / (mean.max() - mean.min() + 1e-9)
                        return normalized

                    def methylation_fraction(matrix):
                        methylated = (matrix == 1).sum(axis=0)
                        valid = (matrix != 0).sum(axis=0)
                        return np.divide(methylated, valid, out=np.zeros_like(methylated, dtype=float), where=valid != 0)

                    if normalize_hmm:
                        mean_hmm = normalized_mean(hmm_matrix)
                    else:
                        mean_hmm = np.nanmean(hmm_matrix, axis=0)
                    mean_gpc = methylation_fraction(gpc_matrix)
                    mean_cpg = methylation_fraction(cpg_matrix)

                    fig = plt.figure(figsize=(18, 12))
                    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 6], hspace=0.01)
                    fig.suptitle(f"{sample} - {ref}", fontsize=14, y=0.95)

                    axes_heat = [fig.add_subplot(gs[1, i]) for i in range(3)]
                    axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(3)]

                    clean_barplot(axes_bar[0], mean_hmm, f"{hmm_feature_layer} HMM Features")
                    clean_barplot(axes_bar[1], mean_gpc, f"GpC Methylation")
                    clean_barplot(axes_bar[2], mean_cpg, f"CpG Methylation")
                    
                    hmm_labels = subset.var_names.astype(int)
                    hmm_label_spacing = 150
                    sns.heatmap(hmm_matrix, cmap=cmap_hmm, ax=axes_heat[0], xticklabels=hmm_labels[::hmm_label_spacing], yticklabels=False, cbar=False)
                    axes_heat[0].set_xticks(range(0, len(hmm_labels), hmm_label_spacing))
                    axes_heat[0].set_xticklabels(hmm_labels[::hmm_label_spacing], rotation=90, fontsize=10)
                    for boundary in bin_boundaries[:-1]:
                        axes_heat[0].axhline(y=boundary, color="black", linewidth=2)

                    sns.heatmap(gpc_matrix, cmap=cmap_gpc, ax=axes_heat[1], xticklabels=gpc_labels[::5], yticklabels=False, cbar=False)
                    axes_heat[1].set_xticks(range(0, len(gpc_labels), 5))
                    axes_heat[1].set_xticklabels(gpc_labels[::5], rotation=90, fontsize=10)
                    for boundary in bin_boundaries[:-1]:
                        axes_heat[1].axhline(y=boundary, color="black", linewidth=2)

                    sns.heatmap(cpg_matrix, cmap=cmap_cpg, ax=axes_heat[2], xticklabels=cpg_labels, yticklabels=False, cbar=False)
                    axes_heat[2].set_xticklabels(cpg_labels, rotation=90, fontsize=10)
                    for boundary in bin_boundaries[:-1]:
                        axes_heat[2].axhline(y=boundary, color="black", linewidth=2)

                    plt.tight_layout()

                    if save_path:
                        save_name = f"{ref} — {sample}"
                        os.makedirs(save_path, exist_ok=True)
                        safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
                        out_file = os.path.join(save_path, f"{safe_name}.png")
                        plt.savefig(out_file, dpi=300)
                        print(f"📁 Saved: {out_file}")

                    plt.show()

                    print(f"📊 Summary for {sample} - {ref}:")
                    for bin_label, percent in percentages.items():
                        print(f"  - {bin_label}: {percent:.1f}%")

                    results.append({
                        "sample": sample,
                        "ref": ref,
                        "hmm_matrix": hmm_matrix,
                        "gpc_matrix": gpc_matrix,
                        "cpg_matrix": cpg_matrix,
                        "row_labels": row_labels,
                        "bin_labels": bin_labels,
                        "bin_boundaries": bin_boundaries,
                        "percentages": percentages
                    })
                    
                    adata.uns['clustermap_results'] = results

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
