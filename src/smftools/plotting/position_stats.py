def plot_volcano_relative_risk(results_dict, save_path=None):
    """
    Plot volcano-style log2(Relative Risk) vs Genomic Position for each group within each reference.

    Parameters:
        results_dict (dict): Output from calculate_relative_risk_by_group.
                             Format: dict[ref][group_label] = (results_df, sig_df)

    Returns:
        None. Displays plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    for ref, group_results in results_dict.items():
        for group_label, (results_df, _) in group_results.items():
            if results_df.empty:
                print(f"Skipping empty results for {ref} / {group_label}")
                continue

            # Prepare data
            log_rr = results_df['log2_Relative_Risk']
            log_pval = results_df['-log10_Adj_P']
            positions = results_df['Genomic_Position']

            # Split by site type
            gpc_df = results_df[results_df['GpC_Site']]
            cpg_df = results_df[results_df['CpG_Site']]

            plt.figure(figsize=(12, 6))

            # GpC as circles
            plt.scatter(
                gpc_df['Genomic_Position'],
                gpc_df['log2_Relative_Risk'],
                c=gpc_df['-log10_Adj_P'],
                cmap='coolwarm', edgecolor='k', s=40, marker='o', label='GpC'
            )

            # CpG as stars
            plt.scatter(
                cpg_df['Genomic_Position'],
                cpg_df['log2_Relative_Risk'],
                c=cpg_df['-log10_Adj_P'],
                cmap='coolwarm', edgecolor='k', s=60, marker='*', label='CpG'
            )

            plt.axhline(y=0, color='gray', linestyle='--')
            plt.xlabel("Genomic Position")
            plt.ylabel("log2(Relative Risk)")
            plt.title(f"{ref} / {group_label} — Relative Risk vs Genomic Position")
            cbar = plt.colorbar()
            cbar.set_label("-log10(Adjusted P-Value)")
            plt.legend()
            plt.tight_layout()

            # Save if requested
            if save_path:
                save_name = f"{ref} — {group_label}"
                os.makedirs(save_path, exist_ok=True)
                safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
                out_file = os.path.join(save_path, f"{safe_name}.png")
                plt.savefig(out_file, dpi=300)
                print(f"📁 Saved: {out_file}")

            plt.show()


def plot_bar_relative_risk(results_dict, sort_by_position=True, xlim=None, ylim=None, save_path=None):
    """
    Plot log2(Relative Risk) as a bar plot across genomic positions for each group within each reference.

    Parameters:
        results_dict (dict): Output from calculate_relative_risk_by_group.
                             Format: dict[ref][group_label] = (results_df, sig_df)
        sort_by_position (bool): Whether to sort bars left-to-right by genomic coordinate.

    Returns:
        None. Displays one plot per (ref, group).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    for ref, group_data in results_dict.items():
        for group_label, (df, _) in group_data.items():
            if df.empty:
                print(f"Skipping empty result for {ref} / {group_label}")
                continue

            df = df.copy()

            # Ensure Genomic_Position is numeric
            df['Genomic_Position'] = df['Genomic_Position'].astype(int)

            if sort_by_position:
                df = df.sort_values('Genomic_Position')

            # Setup
            x = df['Genomic_Position']
            heights = df['log2_Relative_Risk']

            # Coloring by site type (or use different plots if preferred)
            gpc_mask = df['GpC_Site'] & ~df['CpG_Site']
            cpg_mask = df['CpG_Site'] & ~df['GpC_Site']
            both_mask = df['GpC_Site'] & df['CpG_Site']

            plt.figure(figsize=(14, 6))

            # GpC bars
            plt.bar(
                df['Genomic_Position'][gpc_mask],
                heights[gpc_mask],
                width=10,
                color='steelblue',
                label='GpC Site',
                edgecolor='black'
            )

            # CpG bars
            plt.bar(
                df['Genomic_Position'][cpg_mask],
                heights[cpg_mask],
                width=10,
                color='darkorange',
                label='CpG Site',
                edgecolor='black'
            )

            # Both
            if both_mask.any():
                plt.bar(
                    df['Genomic_Position'][both_mask],
                    heights[both_mask],
                    width=10,
                    color='purple',
                    label='GpC + CpG',
                    edgecolor='black'
                )

            # Aesthetic setup
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.xlabel('Genomic Position')
            plt.ylabel('log2(Relative Risk)')
            plt.title(f"{ref} — {group_label}")
            plt.legend()
            
            # Apply axis limits if provided
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)

            plt.tight_layout()

            # Save if requested
            if save_path:
                save_name = f"{ref} — {group_label}"
                os.makedirs(save_path, exist_ok=True)
                safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
                out_file = os.path.join(save_path, f"{safe_name}.png")
                plt.savefig(out_file, dpi=300)
                print(f"📁 Saved: {out_file}")

            plt.show()

def plot_positionwise_matrix(
    adata,
    key="positionwise_result",
    log_transform=False,
    log_base="log1p",  # or 'log2', or None
    triangle="full",
    cmap="vlag",
    figsize=(10, 8),
    vmin=None,
    vmax=None,
    xtick_step=10,
    ytick_step=10,
    save_path=None
):
    """
    Plots positionwise matrices stored in adata.uns[key].

    Parameters:
        adata (AnnData): Input AnnData with computed matrix in .uns[key].
        key (str): Key in .uns containing the matrix dict.
        log_transform (bool): Whether to apply log transformation.
        log_base (str): 'log1p', 'log2', or None.
        triangle (str): 'full', 'lower', or 'upper'.
        cmap (str): Colormap.
        figsize (tuple): Figure size.
        vmin, vmax (float): Value range for color scaling.
        xtick_step, ytick_step (int): Tick step for axis labeling.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    for group, mat_df in adata.uns[key].items():
        mat = mat_df.copy()

        if log_transform:
            with np.errstate(divide='ignore', invalid='ignore'):
                if log_base == "log1p":
                    mat = np.log1p(mat)
                elif log_base == "log2":
                    mat = np.log2(mat.replace(0, np.nanmin(mat[mat > 0]) * 0.1))
                mat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Set color limits for log2 to be centered around 0
        if log_base == "log2" and log_transform and (vmin is None or vmax is None):
            abs_max = np.nanmax(np.abs(mat.values))
            vmin = -abs_max if vmin is None else vmin
            vmax = abs_max if vmax is None else vmax

        # Create mask for triangle
        mask = None
        if triangle == "lower":
            mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
        elif triangle == "upper":
            mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)

        xticks = mat.columns.astype(int)
        yticks = mat.index.astype(int)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mat,
            mask=mask,
            cmap=cmap,
            xticklabels=xticks,
            yticklabels=yticks,
            square=True,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": f"{key} ({log_base})" if log_transform else key},
            ax=ax
        )

        ax.set_title(f"{key} — {group}", pad=30)

        ax.set_xticks(np.arange(0, len(xticks), xtick_step))
        ax.set_xticklabels(xticks[::xtick_step], rotation=90)
        ax.set_yticks(np.arange(0, len(yticks), ytick_step))
        ax.set_yticklabels(yticks[::ytick_step])

        plt.tight_layout()

        # Save if requested
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            safe_name = group.replace("=", "").replace("__", "_").replace(",", "_")
            out_file = os.path.join(save_path, f"{key}_{safe_name}.png")
            plt.savefig(out_file, dpi=300)
            print(f"📁 Saved: {out_file}")

        plt.show()

def plot_positionwise_matrix_grid(
    adata,
    key,
    outer_keys=["Reference_strand", "activity_status"],
    inner_keys=["Promoter_Open", "Enhancer_Open"],
    log_transform=None,
    vmin=None,
    vmax=None,
    cmap="vlag",
    save_path=None,
    figsize=(10, 10),
    xtick_step=10,
    ytick_step=10,
    parallel=False,
    max_threads=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    from matplotlib.gridspec import GridSpec
    from joblib import Parallel, delayed

    matrices = adata.uns[key]
    group_labels = list(matrices.keys())

    parsed_inner = pd.DataFrame([dict(zip(inner_keys, g.split("_")[-len(inner_keys):])) for g in group_labels])
    parsed_outer = pd.Series(["_".join(g.split("_")[:-len(inner_keys)]) for g in group_labels], name="outer")
    parsed = pd.concat([parsed_outer, parsed_inner], axis=1)

    def plot_one_grid(outer_label):
        selected = parsed[parsed['outer'] == outer_label].copy()
        selected["group_str"] = [f"{outer_label}_{row[inner_keys[0]]}_{row[inner_keys[1]]}" for _, row in selected.iterrows()]

        row_vals = sorted(selected[inner_keys[0]].unique())
        col_vals = sorted(selected[inner_keys[1]].unique())

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(len(row_vals), len(col_vals) + 1, width_ratios=[1]*len(col_vals) + [0.05], wspace=0.3)
        axes = np.empty((len(row_vals), len(col_vals)), dtype=object)

        local_vmin, local_vmax = vmin, vmax
        if log_transform == "log2" and (vmin is None or vmax is None):
            all_data = []
            for group_str in selected["group_str"]:
                mat = matrices.get(group_str)
                if mat is not None:
                    all_data.append(np.log2(mat.replace(0, 1e-9).values))
            if all_data:
                combined = np.concatenate([arr.flatten() for arr in all_data])
                vmax_auto = np.nanmax(np.abs(combined))
                local_vmin = -vmax_auto if vmin is None else vmin
                local_vmax = vmax_auto if vmax is None else vmax

        cbar_label = {
            "log2": "log2(Value)",
            "log1p": "log1p(Value)"
        }.get(log_transform, "Value")

        cbar_ax = fig.add_subplot(gs[:, -1])

        for i, row_val in enumerate(row_vals):
            for j, col_val in enumerate(col_vals):
                group_label = f"{outer_label}_{row_val}_{col_val}"
                ax = fig.add_subplot(gs[i, j])
                axes[i, j] = ax
                mat = matrices.get(group_label)
                if mat is None:
                    ax.axis("off")
                    continue

                data = mat.copy()
                if log_transform == "log2":
                    data = np.log2(data.replace(0, 1e-9))
                elif log_transform == "log1p":
                    data = np.log1p(data)

                sns.heatmap(
                    data,
                    ax=ax,
                    cmap=cmap,
                    xticklabels=True,
                    yticklabels=True,
                    square=True,
                    vmin=local_vmin,
                    vmax=local_vmax,
                    cbar=(i == 0 and j == 0),
                    cbar_ax=cbar_ax if (i == 0 and j == 0) else None,
                    cbar_kws={"label": cbar_label if (i == 0 and j == 0) else ""}
                )
                ax.set_title(f"{inner_keys[0]}={row_val}, {inner_keys[1]}={col_val}", fontsize=9, pad=8)

                xticks = data.columns.astype(int)
                yticks = data.index.astype(int)
                ax.set_xticks(np.arange(0, len(xticks), xtick_step))
                ax.set_xticklabels(xticks[::xtick_step], rotation=90)
                ax.set_yticks(np.arange(0, len(yticks), ytick_step))
                ax.set_yticklabels(yticks[::ytick_step])

        fig.suptitle(f"{key} • {outer_label}", fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0, 0.97, 0.95])

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = outer_label.replace("_", "").replace("=", "") + ".png"
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
            print(f"✅ Saved {fname}")

        plt.close(fig)

    if parallel:
        Parallel(n_jobs=max_threads)(delayed(plot_one_grid)(outer_label) for outer_label in parsed['outer'].unique())
    else:
        for outer_label in parsed['outer'].unique():
            plot_one_grid(outer_label)

    print("✅ Finished plotting all grids.")