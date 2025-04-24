def plot_volcano_relative_risk(
    results_dict,
    save_path=None,
    highlight_regions=None,  # List of (start, end) tuples
    highlight_color="lightgray",
    highlight_alpha=0.3,
    xlim=None,
    ylim=None,
):
    """
    Plot volcano-style log2(Relative Risk) vs Genomic Position for each group within each reference.

    Parameters:
        results_dict (dict): Output from calculate_relative_risk_by_group.
                             Format: dict[ref][group_label] = (results_df, sig_df)
        save_path (str): Directory to save plots.
        highlight_regions (list): List of (start, end) tuples for shaded regions.
        highlight_color (str): Color for highlighted regions.
        highlight_alpha (float): Alpha for highlighted region.
        xlim (tuple): Optional x-axis limit.
        ylim (tuple): Optional y-axis limit.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    for ref, group_results in results_dict.items():
        for group_label, (results_df, _) in group_results.items():
            if results_df.empty:
                print(f"Skipping empty results for {ref} / {group_label}")
                continue

            # Split by site type
            gpc_df = results_df[results_df['GpC_Site']]
            cpg_df = results_df[results_df['CpG_Site']]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Highlight regions
            if highlight_regions:
                for start, end in highlight_regions:
                    ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)

            # GpC as circles
            sc1 = ax.scatter(
                gpc_df['Genomic_Position'],
                gpc_df['log2_Relative_Risk'],
                c=gpc_df['-log10_Adj_P'],
                cmap='coolwarm',
                edgecolor='k',
                s=40,
                marker='o',
                label='GpC'
            )

            # CpG as stars
            sc2 = ax.scatter(
                cpg_df['Genomic_Position'],
                cpg_df['log2_Relative_Risk'],
                c=cpg_df['-log10_Adj_P'],
                cmap='coolwarm',
                edgecolor='k',
                s=60,
                marker='*',
                label='CpG'
            )

            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_xlabel("Genomic Position")
            ax.set_ylabel("log2(Relative Risk)")
            ax.set_title(f"{ref} / {group_label} — Relative Risk vs Genomic Position")

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cbar = plt.colorbar(sc1, ax=ax)
            cbar.set_label("-log10(Adjusted P-Value)")

            ax.legend()
            plt.tight_layout()

            # Save if requested
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                safe_name = f"{ref}_{group_label}".replace("=", "").replace("__", "_").replace(",", "_").replace(" ", "_")
                out_file = os.path.join(save_path, f"{safe_name}.png")
                plt.savefig(out_file, dpi=300)
                print(f"📁 Saved: {out_file}")

            plt.show()

def plot_bar_relative_risk(
    results_dict,
    sort_by_position=True,
    xlim=None,
    ylim=None,
    save_path=None,
    highlight_regions=None,  # List of (start, end) tuples
    highlight_color="lightgray",
    highlight_alpha=0.3
):
    """
    Plot log2(Relative Risk) as a bar plot across genomic positions for each group within each reference.

    Parameters:
        results_dict (dict): Output from calculate_relative_risk_by_group.
        sort_by_position (bool): Whether to sort bars left-to-right by genomic coordinate.
        xlim, ylim (tuple): Axis limits.
        save_path (str or None): Directory to save plots.
        highlight_regions (list of tuple): List of (start, end) genomic regions to shade.
        highlight_color (str): Color of shaded region.
        highlight_alpha (float): Transparency of shaded region.
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
            df['Genomic_Position'] = df['Genomic_Position'].astype(int)

            if sort_by_position:
                df = df.sort_values('Genomic_Position')

            gpc_mask = df['GpC_Site'] & ~df['CpG_Site']
            cpg_mask = df['CpG_Site'] & ~df['GpC_Site']
            both_mask = df['GpC_Site'] & df['CpG_Site']

            fig, ax = plt.subplots(figsize=(14, 6))

            # Optional shaded regions
            if highlight_regions:
                for start, end in highlight_regions:
                    ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)

            # Bar plots
            ax.bar(
                df['Genomic_Position'][gpc_mask],
                df['log2_Relative_Risk'][gpc_mask],
                width=10,
                color='steelblue',
                label='GpC Site',
                edgecolor='black'
            )

            ax.bar(
                df['Genomic_Position'][cpg_mask],
                df['log2_Relative_Risk'][cpg_mask],
                width=10,
                color='darkorange',
                label='CpG Site',
                edgecolor='black'
            )

            if both_mask.any():
                ax.bar(
                    df['Genomic_Position'][both_mask],
                    df['log2_Relative_Risk'][both_mask],
                    width=10,
                    color='purple',
                    label='GpC + CpG',
                    edgecolor='black'
                )

            ax.axhline(y=0, color='gray', linestyle='--')
            ax.set_xlabel('Genomic Position')
            ax.set_ylabel('log2(Relative Risk)')
            ax.set_title(f"{ref} — {group_label}")
            ax.legend()

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                safe_name = f"{ref}_{group_label}".replace("=", "").replace("__", "_").replace(",", "_")
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
    figsize=(12, 10),  # Taller to accommodate line plot below
    vmin=None,
    vmax=None,
    xtick_step=10,
    ytick_step=10,
    save_path=None,
    highlight_position=None,         # Can be a single int/float or list of them
    highlight_axis="row",            # "row" or "column"
    annotate_points=False             # ✅ New option
):
    """
    Plots positionwise matrices stored in adata.uns[key], with an optional line plot
    for specified row(s) or column(s), and highlights them on the heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os

    def find_closest_index(index, target):
        index_vals = pd.to_numeric(index, errors="coerce")
        target_val = pd.to_numeric([target], errors="coerce")[0]
        diffs = pd.Series(np.abs(index_vals - target_val), index=index)
        return diffs.idxmin()

    # Ensure highlight_position is a list
    if highlight_position is not None and not isinstance(highlight_position, (list, tuple, np.ndarray)):
        highlight_position = [highlight_position]

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

        # 👉 Make taller figure: heatmap on top, line plot below
        fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1.5])
        heat_ax, line_ax = axs

        # Heatmap
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
            ax=heat_ax
        )

        heat_ax.set_title(f"{key} — {group}", pad=20)
        heat_ax.set_xticks(np.arange(0, len(xticks), xtick_step))
        heat_ax.set_xticklabels(xticks[::xtick_step], rotation=90)
        heat_ax.set_yticks(np.arange(0, len(yticks), ytick_step))
        heat_ax.set_yticklabels(yticks[::ytick_step])

        # Line plot
        if highlight_position is not None:
            colors = plt.cm.tab10.colors
            for i, pos in enumerate(highlight_position):
                try:
                    if highlight_axis == "row":
                        closest = find_closest_index(mat.index, pos)
                        series = mat.loc[closest]
                        x_vals = pd.to_numeric(series.index, errors="coerce")
                        idx = mat.index.get_loc(closest)
                        heat_ax.axhline(idx, color=colors[i % len(colors)], linestyle="--", linewidth=1)
                        label = f"Row {pos} → {closest}"
                    else:
                        closest = find_closest_index(mat.columns, pos)
                        series = mat[closest]
                        x_vals = pd.to_numeric(series.index, errors="coerce")
                        idx = mat.columns.get_loc(closest)
                        heat_ax.axvline(idx, color=colors[i % len(colors)], linestyle="--", linewidth=1)
                        label = f"Col {pos} → {closest}"

                    line = line_ax.plot(x_vals, series.values, marker='o', label=label, color=colors[i % len(colors)])

                    # Annotate each point
                    if annotate_points:
                        for x, y in zip(x_vals, series.values):
                            if not np.isnan(y):
                                line_ax.annotate(
                                    f"{y:.2f}",
                                    xy=(x, y),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    fontsize=8
                                )
                except Exception as e:
                    line_ax.text(0.5, 0.5, f"⚠️ Error plotting {highlight_axis} @ {pos}",
                                 ha='center', va='center', fontsize=10)
                    print(f"Error plotting line for {highlight_axis}={pos}: {e}")

            line_ax.set_title(f"{highlight_axis.capitalize()} Profile(s)")
            line_ax.set_xlabel(f"{'Column' if highlight_axis == 'row' else 'Row'} position")
            line_ax.set_ylabel("Value")
            line_ax.grid(True)
            line_ax.legend(fontsize=8)

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