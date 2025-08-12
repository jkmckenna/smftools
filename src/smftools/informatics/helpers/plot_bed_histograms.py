# plot_bed_histograms

def plot_bed_histograms(bed_file, plotting_directory, fasta):
    """
    Plots read length, coverage, mapq, read quality stats for each record.

    Parameters:
        bed_file (str): Path to the bed file to derive metrics from.
        plot_directory (str): Path to the directory to write out historgrams.
        fasta (str): Path to FASTA corresponding to bed

    Returns:
        None
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # plot_bed_histograms.py

def plot_bed_histograms(
    bed_file,
    plotting_directory,
    fasta,
    *,
    bins=60,
    clip_quantiles=(0.0, 0.995),
    cov_bin_size=1000,       # coverage bin size in bp
    rows_per_fig=6,          # paginate if many chromosomes
    include_mapq_quality=True,   # add MAPQ + avg read quality columns to grid
    coordinate_mode="one_based",  # "one_based" (your BED-like) or "zero_based"
):
    """
    Plot per-chromosome QC grids from a BED-like file.

    Expects columns:
      chrom, start, end, read_len, qname, mapq, avg_base_qual

    For each chromosome:
      - Column 1: Read length histogram
      - Column 2: Coverage across the chromosome (binned)
      - (optional) Column 3: MAPQ histogram
      - (optional) Column 4: Avg base quality histogram

    The figure is paginated: rows = chromosomes (up to rows_per_fig), columns depend on include_mapq_quality.
    Saves one PNG per page under `plotting_directory`.

    Parameters
    ----------
    bed_file : str
    plotting_directory : str
    fasta : str
        Reference FASTA (used to get chromosome lengths).
    bins : int
        Histogram bins for read length / MAPQ / quality.
    clip_quantiles : (float, float)
        Clip hist tails for readability (e.g., (0, 0.995)).
    cov_bin_size : int
        Bin size (bp) for coverage plot; bigger = faster/coarser.
    rows_per_fig : int
        Number of chromosomes per page.
    include_mapq_quality : bool
        If True, add MAPQ and avg base quality histograms as extra columns.
    coordinate_mode : {"one_based","zero_based"}
        One-based, inclusive (your file) vs BED-standard zero-based, half-open.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pysam

    os.makedirs(plotting_directory, exist_ok=True)

    bed_basename = os.path.basename(bed_file).rsplit(".bed", 1)[0]
    print(f"[plot_bed_histograms] Loading: {bed_file}")

    # Load BED-like table
    cols = ['chrom', 'start', 'end', 'read_len', 'qname', 'mapq', 'avg_q']
    df = pd.read_csv(bed_file, sep="\t", header=None, names=cols, dtype={
        'chrom': str, 'start': int, 'end': int, 'read_len': int, 'qname': str,
        'mapq': float, 'avg_q': float
    })

    # Drop unaligned records (chrom == '*') if present
    df = df[df['chrom'] != '*'].copy()
    if df.empty:
        print("[plot_bed_histograms] No aligned reads found; nothing to plot.")
        return

    # Ensure coordinate mode consistent; convert to 0-based half-open for bin math internally
    # Input is typically one_based inclusive (from your writer).
    if coordinate_mode not in {"one_based", "zero_based"}:
        raise ValueError("coordinate_mode must be 'one_based' or 'zero_based'")

    if coordinate_mode == "one_based":
        # convert to 0-based half-open [start0, end0)
        start0 = df['start'].to_numpy() - 1
        end0   = df['end'].to_numpy()   # inclusive in input -> +1 already handled by not subtracting
    else:
        # already 0-based half-open (assumption)
        start0 = df['start'].to_numpy()
        end0   = df['end'].to_numpy()

    # Clip helper for hist tails
    def _clip_series(s, q=(0.0, 0.995)):
        if q is None:
            return s.to_numpy()
        lo = s.quantile(q[0]) if q[0] is not None else s.min()
        hi = s.quantile(q[1]) if q[1] is not None else s.max()
        x = s.to_numpy(dtype=float)
        return np.clip(x, lo, hi)

    # Load chromosome order/lengths from FASTA
    with pysam.FastaFile(fasta) as fa:
        ref_names = list(fa.references)
        ref_lengths = dict(zip(ref_names, fa.lengths))

    # Keep only chroms present in FASTA and with at least one read
    chroms = [c for c in df['chrom'].unique() if c in ref_lengths]
    # Order chromosomes by FASTA order
    chrom_order = [c for c in ref_names if c in chroms]

    if not chrom_order:
        print("[plot_bed_histograms] No chromosomes from BED are present in FASTA; aborting.")
        return

    # Pagination
    def _sanitize(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)

    cols_per_fig = 4 if include_mapq_quality else 2

    for start_idx in range(0, len(chrom_order), rows_per_fig):
        chunk = chrom_order[start_idx:start_idx + rows_per_fig]
        nrows = len(chunk)
        ncols = cols_per_fig

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(4.0 * ncols, 2.6 * nrows),
            dpi=160,
            squeeze=False
        )

        for r, chrom in enumerate(chunk):
            chrom_len = ref_lengths[chrom]
            mask = (df['chrom'].to_numpy() == chrom)

            # Slice per-chrom arrays for speed
            s0 = start0[mask]
            e0 = end0[mask]
            len_arr = df.loc[mask, 'read_len']
            mapq_arr = df.loc[mask, 'mapq']
            q_arr = df.loc[mask, 'avg_q']

            # --- Col 1: Read length histogram (clipped) ---
            ax = axes[r, 0]
            ax.hist(_clip_series(len_arr, clip_quantiles), bins=bins, edgecolor="black", alpha=0.7)
            if r == 0:
                ax.set_title("Read length")
            ax.set_ylabel(f"{chrom}\n(n={mask.sum()})")
            ax.set_xlabel("bp")
            ax.grid(alpha=0.25)

            # --- Col 2: Coverage (binned over genome) ---
            ax = axes[r, 1]
            nb = max(1, int(np.ceil(chrom_len / cov_bin_size)))
            # Bin edges in 0-based coords
            edges = np.linspace(0, chrom_len, nb + 1, dtype=int)

            # Compute per-bin "read count coverage": number of reads overlapping each bin.
            # Approximate by incrementing all bins touched by the interval.
            # (Fast and memory-light; for exact base coverage use smaller cov_bin_size.)
            cov = np.zeros(nb, dtype=np.int32)
            # bin indices overlapped by each read (0-based half-open)
            b0 = np.minimum(np.searchsorted(edges, s0, side="right") - 1, nb - 1)
            b1 = np.maximum(np.searchsorted(edges, np.maximum(e0 - 1, 0), side="right") - 1, 0)
            # ensure valid ordering
            b_lo = np.minimum(b0, b1)
            b_hi = np.maximum(b0, b1)

            # Increment all bins in range; loop but at bin resolution (fast for reasonable cov_bin_size).
            for lo, hi in zip(b_lo, b_hi):
                cov[lo:hi + 1] += 1

            x_mid = (edges[:-1] + edges[1:]) / 2.0
            ax.plot(x_mid, cov)
            if r == 0:
                ax.set_title(f"Coverage (~{cov_bin_size} bp bins)")
            ax.set_xlim(0, chrom_len)
            ax.set_xlabel("Position (bp)")
            ax.set_ylabel("")  # already show chrom on col 1
            ax.grid(alpha=0.25)

            if include_mapq_quality:
                # --- Col 3: MAPQ ---
                ax = axes[r, 2]
                # Clip MAPQ upper tail if needed (usually 60)
                ax.hist(_clip_series(mapq_arr.fillna(0), clip_quantiles), bins=bins, edgecolor="black", alpha=0.7)
                if r == 0:
                    ax.set_title("MAPQ")
                ax.set_xlabel("MAPQ")
                ax.grid(alpha=0.25)

                # --- Col 4: Avg base quality ---
                ax = axes[r, 3]
                ax.hist(_clip_series(q_arr.fillna(np.nan), clip_quantiles), bins=bins, edgecolor="black", alpha=0.7)
                if r == 0:
                    ax.set_title("Avg base qual")
                ax.set_xlabel("Phred")
                ax.grid(alpha=0.25)

        fig.suptitle(
            f"{bed_basename} — per-chromosome QC "
            f"({'len,cov,MAPQ,qual' if include_mapq_quality else 'len,cov'})",
            y=0.995, fontsize=11
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        page = start_idx // rows_per_fig + 1
        out_png = os.path.join(plotting_directory, f"{_sanitize(bed_basename)}_qc_page{page}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    print("[plot_bed_histograms] Done.")


    # bed_basename = os.path.basename(bed_file).split('.bed')[0]
    # # Load the BED file into a DataFrame
    # print(f"Loading BED to plot read length and coverage histograms: {bed_file}")
    # df = pd.read_csv(bed_file, sep='\t', header=None, names=['chromosome', 'start', 'end', 'length', 'read_name', 'mapq', 'read_quality'])
    
    # # Group by chromosome
    # grouped = df.groupby('chromosome')

    # # for each chromosome, get the record length of that chromosome from the fasta. Use from 0 to this length for the positional coverage plot.

    # # Change below and make a plot grid instead. For each, make row for chromsome, col for read length and coverage
    # # Clip the outliers to make plots cleaner

    # for chrom, group in grouped:
    #     # Plot read length histogram
    #     plt.figure(figsize=(12, 6))
    #     plt.hist(group['length'], bins=50, edgecolor='k', alpha=0.7)
    #     plt.title(f'Read Length Histogram of reads aligned to {chrom}')
    #     plt.xlabel('Read Length')
    #     plt.ylabel('Count')
    #     plt.grid(True)
    #     save_name = os.path.join(plotting_directory, f'{bed_basename}_{chrom}_read_length_histogram.png')
    #     plt.savefig(save_name)
    #     plt.close()

    #     # Compute coverage
    #     coverage = np.zeros(group['end'].max())
    #     for _, row in group.iterrows():
    #         coverage[row['start']:row['end']] += 1
        
    #     # Plot coverage histogram
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(coverage, color='b')
    #     plt.title(f'Coverage Histogram for {chrom}')
    #     plt.xlabel('Position')
    #     plt.ylabel('Coverage')
    #     plt.grid(True)
    #     save_name = os.path.join(plotting_directory, f'{bed_basename}_{chrom}_coverage_histogram.png')
    #     plt.savefig(save_name)
    #     plt.close()