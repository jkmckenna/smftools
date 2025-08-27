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
    reference_col='Reference_strand',
    hmm_feature_layer="hmm_combined",
    layer_gpc="nan0_0minus1",
    layer_cpg="nan0_0minus1",
    layer_any_c="nan0_0minus1",
    cmap_hmm="tab10",
    cmap_gpc="coolwarm",
    cmap_cpg="viridis",
    cmap_any_c='coolwarm',
    min_quality=20,
    min_length=200,
    min_mapped_length_to_reference_length_ratio=0.8,
    min_position_valid_fraction=0.5,
    sample_mapping=None,
    save_path=None,
    normalize_hmm=False,
    sort_by="gpc",  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
    bins=None,
    deaminase=False,
    min_signal=0
    ):
    import scipy.cluster.hierarchy as sch
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    results = []
    if deaminase:
        signal_type = 'deamination'
    else:
        signal_type = 'methylation'

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            try:
                subset = adata[
                    (adata.obs[reference_col] == ref) &
                    (adata.obs[sample_col] == sample) &
                    (adata.obs['read_quality'] >= min_quality) &
                    (adata.obs['read_length'] >= min_length) &
                    (adata.obs['mapped_length_to_reference_length_ratio'] > min_mapped_length_to_reference_length_ratio)
                ]
                
                mask = subset.var[f"{ref}_valid_fraction"].astype(float) > float(min_position_valid_fraction)
                subset = subset[:, mask]

                if subset.shape[0] == 0:
                    print(f"  No reads left after filtering for {sample} - {ref}")
                    continue

                if bins:
                    print(f"Using defined bins to subset clustermap for {sample} - {ref}")
                    bins_temp = bins
                else:
                    print(f"Using all reads for clustermap for {sample} - {ref}")
                    bins_temp = {"All": (subset.obs['Reference_strand'] == ref)}

                # Get column positions (not var_names!) of site masks
                gpc_sites = np.where(subset.var[f"{ref}_GpC_site"].values)[0]
                cpg_sites = np.where(subset.var[f"{ref}_CpG_site"].values)[0]
                any_c_sites = np.where(subset.var[f"{ref}_any_C_site"].values)[0]
                num_gpc = len(gpc_sites)
                num_cpg = len(cpg_sites)
                num_c = len(any_c_sites)
                print(f"Found {num_gpc} GpC sites at {gpc_sites} \nand {num_cpg} CpG sites at {cpg_sites} for {sample} - {ref}")

                # Use var_names for x-axis tick labels
                gpc_labels = subset.var_names[gpc_sites].astype(int)
                cpg_labels = subset.var_names[cpg_sites].astype(int)
                any_c_labels = subset.var_names[any_c_sites].astype(int)

                stacked_hmm_feature, stacked_gpc, stacked_cpg, stacked_any_c = [], [], [], []
                row_labels, bin_labels = [], []
                bin_boundaries = []

                total_reads = subset.shape[0]
                percentages = {}
                last_idx = 0

                for bin_label, bin_filter in bins_temp.items():
                    subset_bin = subset[bin_filter].copy()
                    num_reads = subset_bin.shape[0]
                    print(f"analyzing {num_reads} reads for {bin_label} bin in {sample} - {ref}")
                    percent_reads = (num_reads / total_reads) * 100 if total_reads > 0 else 0
                    percentages[bin_label] = percent_reads

                    if num_reads > 0 and num_cpg > 0 and num_gpc > 0:
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
                        elif sort_by == "any_c":
                            linkage = sch.linkage(subset_bin.layers[layer_any_c], method="ward")
                            order = sch.leaves_list(linkage)
                        else:
                            raise ValueError(f"Unsupported sort_by option: {sort_by}")

                        stacked_hmm_feature.append(subset_bin[order].layers[hmm_feature_layer])
                        stacked_gpc.append(subset_bin[order][:, gpc_sites].layers[layer_gpc])
                        stacked_cpg.append(subset_bin[order][:, cpg_sites].layers[layer_cpg])
                        stacked_any_c.append(subset_bin[order][:, any_c_sites].layers[layer_any_c])

                        row_labels.extend([bin_label] * num_reads)
                        bin_labels.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")
                        last_idx += num_reads
                        bin_boundaries.append(last_idx)

                if stacked_hmm_feature:
                    hmm_matrix = np.vstack(stacked_hmm_feature)
                    gpc_matrix = np.vstack(stacked_gpc)
                    cpg_matrix = np.vstack(stacked_cpg)
                    any_c_matrix = np.vstack(stacked_any_c)

                    if hmm_matrix.size > 0:
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
                        mean_any_c = methylation_fraction(any_c_matrix)

                        fig = plt.figure(figsize=(18, 12))
                        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 6], hspace=0.01)
                        fig.suptitle(f"{sample} - {ref}", fontsize=14, y=0.95)

                        axes_heat = [fig.add_subplot(gs[1, i]) for i in range(4)]
                        axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(4)]

                        clean_barplot(axes_bar[0], mean_hmm, f"{hmm_feature_layer} HMM Features")
                        clean_barplot(axes_bar[1], mean_gpc, f"GpC Accessibility Signal")
                        clean_barplot(axes_bar[2], mean_cpg, f"CpG Accessibility Signal")
                        clean_barplot(axes_bar[3], mean_any_c, f"Any C Accessibility Signal")
                        
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

                        sns.heatmap(any_c_matrix, cmap=cmap_any_c, ax=axes_heat[3], xticklabels=any_c_labels[::20], yticklabels=False, cbar=False)
                        axes_heat[3].set_xticks(range(0, len(any_c_labels), 20))
                        axes_heat[3].set_xticklabels(any_c_labels[::20], rotation=90, fontsize=10)
                        for boundary in bin_boundaries[:-1]:
                            axes_heat[3].axhline(y=boundary, color="black", linewidth=2)

                        plt.tight_layout()

                        if save_path:
                            save_name = f"{ref} — {sample}"
                            os.makedirs(save_path, exist_ok=True)
                            safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
                            out_file = os.path.join(save_path, f"{safe_name}.png")
                            plt.savefig(out_file, dpi=300)
                            print(f"Saved: {out_file}")
                            plt.close()
                        else:
                            plt.show()

                        print(f"Summary for {sample} - {ref}:")
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


def combined_raw_clustermap(
    adata,
    sample_col='Sample_Names',
    reference_col='Reference_strand',
    layer_any_c="nan0_0minus1",
    layer_gpc="nan0_0minus1",
    layer_cpg="nan0_0minus1",
    cmap_any_c="coolwarm",
    cmap_gpc="coolwarm",
    cmap_cpg="viridis",
    min_quality=20,
    min_length=200,
    min_mapped_length_to_reference_length_ratio=0.8,
    min_position_valid_fraction=0.5,
    sample_mapping=None,
    save_path=None,
    sort_by="gpc",  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
    bins=None,
    deaminase=False,
    min_signal=0
    ):
    import scipy.cluster.hierarchy as sch
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    results = []

    for ref in adata.obs[reference_col].cat.categories:
        for sample in adata.obs[sample_col].cat.categories:
            try:
                subset = adata[
                    (adata.obs[reference_col] == ref) &
                    (adata.obs[sample_col] == sample) &
                    (adata.obs['read_quality'] >= min_quality) &
                    (adata.obs['mapped_length'] >= min_length) &
                    (adata.obs['mapped_length_to_reference_length_ratio'] >= min_mapped_length_to_reference_length_ratio)
                ]

                mask = subset.var[f"{ref}_valid_fraction"].astype(float) > float(min_position_valid_fraction)
                subset = subset[:, mask]

                if subset.shape[0] == 0:
                    print(f"  No reads left after filtering for {sample} - {ref}")
                    continue

                if bins:
                    print(f"Using defined bins to subset clustermap for {sample} - {ref}")
                    bins_temp = bins
                else:
                    print(f"Using all reads for clustermap for {sample} - {ref}")
                    bins_temp = {"All": (subset.obs['Reference_strand'] == ref)}

                # Get column positions (not var_names!) of site masks
                any_c_sites = np.where(subset.var[f"{ref}_any_C_site"].values)[0]
                gpc_sites = np.where(subset.var[f"{ref}_GpC_site"].values)[0]
                cpg_sites = np.where(subset.var[f"{ref}_CpG_site"].values)[0]
                num_any_c = len(any_c_sites)
                num_gpc = len(gpc_sites)
                num_cpg = len(cpg_sites)
                print(f"Found {num_gpc} GpC sites at {gpc_sites} \nand {num_cpg} CpG sites at {cpg_sites}\n and {num_any_c} any_C sites at {any_c_sites} for {sample} - {ref}")

                # Use var_names for x-axis tick labels
                gpc_labels = subset.var_names[gpc_sites].astype(int)
                cpg_labels = subset.var_names[cpg_sites].astype(int)
                any_c_labels = subset.var_names[any_c_sites].astype(int)

                stacked_any_c, stacked_gpc, stacked_cpg = [], [], []
                row_labels, bin_labels = [], []
                bin_boundaries = []

                total_reads = subset.shape[0]
                percentages = {}
                last_idx = 0

                for bin_label, bin_filter in bins_temp.items():
                    subset_bin = subset[bin_filter].copy()
                    num_reads = subset_bin.shape[0]
                    print(f"analyzing {num_reads} reads for {bin_label} bin in {sample} - {ref}")
                    percent_reads = (num_reads / total_reads) * 100 if total_reads > 0 else 0
                    percentages[bin_label] = percent_reads

                    if num_reads > 0 and num_cpg > 0 and num_gpc > 0:
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
                        elif sort_by == "any_c":
                            linkage = sch.linkage(subset_bin[:, any_c_sites].layers[layer_any_c], method="ward")
                            order = sch.leaves_list(linkage)
                        elif sort_by == "gpc_cpg":
                            linkage = sch.linkage(subset_bin.layers[layer_gpc], method="ward")
                            order = sch.leaves_list(linkage)
                        elif sort_by == "none":
                            order = np.arange(num_reads)
                        else:
                            raise ValueError(f"Unsupported sort_by option: {sort_by}")

                        stacked_any_c.append(subset_bin[order][:, any_c_sites].layers[layer_any_c])
                        stacked_gpc.append(subset_bin[order][:, gpc_sites].layers[layer_gpc])
                        stacked_cpg.append(subset_bin[order][:, cpg_sites].layers[layer_cpg])

                        row_labels.extend([bin_label] * num_reads)
                        bin_labels.append(f"{bin_label}: {num_reads} reads ({percent_reads:.1f}%)")
                        last_idx += num_reads
                        bin_boundaries.append(last_idx)

                if stacked_any_c:
                    any_c_matrix = np.vstack(stacked_any_c)
                    gpc_matrix = np.vstack(stacked_gpc)
                    cpg_matrix = np.vstack(stacked_cpg)

                    if any_c_matrix.size > 0:
                        def normalized_mean(matrix):
                            mean = np.nanmean(matrix, axis=0)
                            normalized = (mean - mean.min()) / (mean.max() - mean.min() + 1e-9)
                            return normalized

                        def methylation_fraction(matrix):
                            methylated = (matrix == 1).sum(axis=0)
                            valid = (matrix != 0).sum(axis=0)
                            return np.divide(methylated, valid, out=np.zeros_like(methylated, dtype=float), where=valid != 0)

                        mean_gpc = methylation_fraction(gpc_matrix)
                        mean_cpg = methylation_fraction(cpg_matrix)
                        mean_any_c = methylation_fraction(any_c_matrix)

                        fig = plt.figure(figsize=(18, 12))
                        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 6], hspace=0.01)
                        fig.suptitle(f"{sample} - {ref} - {total_reads} reads", fontsize=14, y=0.95)

                        axes_heat = [fig.add_subplot(gs[1, i]) for i in range(3)]
                        axes_bar = [fig.add_subplot(gs[0, i], sharex=axes_heat[i]) for i in range(3)]

                        clean_barplot(axes_bar[0], mean_any_c, f"any C site Modification Signal")
                        clean_barplot(axes_bar[1], mean_gpc, f"GpC Modification Signal")
                        clean_barplot(axes_bar[2], mean_cpg, f"CpG Modification Signal")
                        

                        sns.heatmap(any_c_matrix, cmap=cmap_any_c, ax=axes_heat[0], xticklabels=any_c_labels[::20], yticklabels=False, cbar=False)
                        axes_heat[0].set_xticks(range(0, len(any_c_labels), 20))
                        axes_heat[0].set_xticklabels(any_c_labels[::20], rotation=90, fontsize=10)
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
                            print(f"Saved: {out_file}")
                            plt.close()
                        else:
                            plt.show()

                        print(f"Summary for {sample} - {ref}:")
                        for bin_label, percent in percentages.items():
                            print(f"  - {bin_label}: {percent:.1f}%")

                        results.append({
                            "sample": sample,
                            "ref": ref,
                            "any_c_matrix": any_c_matrix,
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
            

import os
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_hmm_layers_rolling_by_sample_ref(
    adata,
    layers: Optional[Sequence[str]] = None,
    sample_col: str = "Barcode",
    ref_col: str = "Reference_strand",
    samples: Optional[Sequence[str]] = None,
    references: Optional[Sequence[str]] = None,
    window: int = 51,
    min_periods: int = 1,
    center: bool = True,
    rows_per_page: int = 6,
    figsize_per_cell: Tuple[float, float] = (4.0, 2.5),
    dpi: int = 160,
    output_dir: Optional[str] = None,
    save: bool = True,
    show_raw: bool = False,
    cmap: str = "tab10",
    use_var_coords: bool = True,
):
    """
    For each sample (row) and reference (col) plot the rolling average of the
    positional mean (mean across reads) for each layer listed.

    Parameters
    ----------
    adata : AnnData
        Input annotated data (expects obs columns sample_col and ref_col).
    layers : list[str] | None
        Which adata.layers to plot. If None, attempts to autodetect layers whose
        matrices look like "HMM" outputs (else will error). If None and layers
        cannot be found, user must pass a list.
    sample_col, ref_col : str
        obs columns used to group rows.
    samples, references : optional lists
        explicit ordering of samples / references. If None, categories in adata.obs are used.
    window : int
        rolling window size (odd recommended). If window <= 1, no smoothing applied.
    min_periods : int
        min periods param for pd.Series.rolling.
    center : bool
        center the rolling window.
    rows_per_page : int
        paginate rows per page into multiple figures if needed.
    figsize_per_cell : (w,h)
        per-subplot size in inches.
    dpi : int
        figure dpi when saving.
    output_dir : str | None
        directory to save pages; created if necessary. If None and save=True, uses cwd.
    save : bool
        whether to save PNG files.
    show_raw : bool
        draw unsmoothed mean as faint line under smoothed curve.
    cmap : str
        matplotlib colormap for layer lines.
    use_var_coords : bool
        if True, tries to use adata.var_names (coerced to int) as x-axis coordinates; otherwise uses 0..n-1.

    Returns
    -------
    saved_files : list[str]
        list of saved filenames (may be empty if save=False).
    """

    # --- basic checks / defaults ---
    if sample_col not in adata.obs.columns or ref_col not in adata.obs.columns:
        raise ValueError(f"sample_col '{sample_col}' and ref_col '{ref_col}' must exist in adata.obs")

    # canonicalize samples / refs
    if samples is None:
        sseries = adata.obs[sample_col]
        if not pd.api.types.is_categorical_dtype(sseries):
            sseries = sseries.astype("category")
        samples_all = list(sseries.cat.categories)
    else:
        samples_all = list(samples)

    if references is None:
        rseries = adata.obs[ref_col]
        if not pd.api.types.is_categorical_dtype(rseries):
            rseries = rseries.astype("category")
        refs_all = list(rseries.cat.categories)
    else:
        refs_all = list(references)

    # choose layers: if not provided, try a sensible default: all layers
    if layers is None:
        layers = list(adata.layers.keys())
        if len(layers) == 0:
            raise ValueError("No adata.layers found. Please pass `layers=[...]` of the HMM layers to plot.")
    layers = list(layers)

    # x coordinates (positions)
    try:
        if use_var_coords:
            x_coords = np.array([int(v) for v in adata.var_names])
        else:
            raise Exception("user disabled var coords")
    except Exception:
        # fallback to 0..n_vars-1
        x_coords = np.arange(adata.shape[1], dtype=int)

    # make output dir
    if save:
        outdir = output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None

    n_samples = len(samples_all)
    n_refs = len(refs_all)
    total_pages = math.ceil(n_samples / rows_per_page)
    saved_files = []

    # color cycle for layers
    cmap_obj = plt.get_cmap(cmap)
    n_layers = max(1, len(layers))
    colors = [cmap_obj(i / max(1, n_layers - 1)) for i in range(n_layers)]

    for page in range(total_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n_samples)
        chunk = samples_all[start:end]
        nrows = len(chunk)
        ncols = n_refs

        fig_w = figsize_per_cell[0] * ncols
        fig_h = figsize_per_cell[1] * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(fig_w, fig_h), dpi=dpi,
                                 squeeze=False)

        for r_idx, sample_name in enumerate(chunk):
            for c_idx, ref_name in enumerate(refs_all):
                ax = axes[r_idx][c_idx]

                # subset adata
                mask = (adata.obs[sample_col].values == sample_name) & (adata.obs[ref_col].values == ref_name)
                sub = adata[mask]
                if sub.n_obs == 0:
                    ax.text(0.5, 0.5, "No reads", ha="center", va="center", transform=ax.transAxes, color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r_idx == 0:
                        ax.set_title(str(ref_name), fontsize=9)
                    if c_idx == 0:
                        total_reads = int((adata.obs[sample_col] == sample_name).sum())
                        ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                    continue

                # for each layer, compute positional mean across reads (ignore NaNs)
                plotted_any = False
                for li, layer in enumerate(layers):
                    if layer in sub.layers:
                        mat = sub.layers[layer]
                    else:
                        # fallback: try .X only for the first layer if layer not present
                        if layer == layers[0] and getattr(sub, "X", None) is not None:
                            mat = sub.X
                        else:
                            # layer not present for this subset
                            continue

                    # convert matrix to numpy 2D
                    if hasattr(mat, "toarray"):
                        try:
                            arr = mat.toarray()
                        except Exception:
                            arr = np.asarray(mat)
                    else:
                        arr = np.asarray(mat)

                    if arr.size == 0 or arr.shape[1] == 0:
                        continue

                    # compute column-wise mean ignoring NaNs
                    # if arr is boolean or int, convert to float to support NaN
                    arr = arr.astype(float)
                    with np.errstate(all="ignore"):
                        col_mean = np.nanmean(arr, axis=0)

                    # If all-NaN, skip
                    if np.all(np.isnan(col_mean)):
                        continue

                    # smooth via pandas rolling (centered)
                    if (window is None) or (window <= 1):
                        smoothed = col_mean
                    else:
                        ser = pd.Series(col_mean)
                        smoothed = ser.rolling(window=window, min_periods=min_periods, center=center).mean().to_numpy()

                    # x axis: x_coords (trim/pad to match length)
                    L = len(col_mean)
                    x = x_coords[:L]

                    # optionally plot raw faint line first
                    if show_raw:
                        ax.plot(x, col_mean[:L], linewidth=0.7, alpha=0.25, zorder=1)

                    ax.plot(x, smoothed[:L], label=layer, color=colors[li], linewidth=1.2, alpha=0.95, zorder=2)
                    plotted_any = True

                # labels / titles
                if r_idx == 0:
                    ax.set_title(str(ref_name), fontsize=9)
                if c_idx == 0:
                    total_reads = int((adata.obs[sample_col] == sample_name).sum())
                    ax.set_ylabel(f"{sample_name}\n(n={total_reads})", fontsize=8)
                if r_idx == nrows - 1:
                    ax.set_xlabel("position", fontsize=8)

                # legend (only show in top-left plot to reduce clutter)
                if (r_idx == 0 and c_idx == 0) and plotted_any:
                    ax.legend(fontsize=7, loc="upper right")

                ax.grid(True, alpha=0.2)

        fig.suptitle(f"Rolling mean of layer positional means (window={window}) — page {page+1}/{total_pages}", fontsize=11, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            fname = os.path.join(outdir, f"hmm_layers_rolling_page{page+1}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=dpi)
            saved_files.append(fname)
        else:
            plt.show()
        plt.close(fig)

    return saved_files
