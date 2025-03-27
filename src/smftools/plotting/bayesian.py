def plot_volcano_relative_risk(results_dict):
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
            plt.show()


def plot_bar_relative_risk(results_dict, sort_by_position=True, xlim=None, ylim=None):
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
            plt.show()
