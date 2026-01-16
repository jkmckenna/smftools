import concurrent.futures
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.optional_imports import require

from ..readwrite import make_dirs

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pybedtools as pybedtools_types
    import pyBigWig as pybigwig_types
    import pysam as pysam_types

try:
    import pybedtools
except Exception:
    pybedtools = None  # type: ignore

try:
    import pyBigWig
except Exception:
    pyBigWig = None  # type: ignore

try:
    import pysam
except Exception:
    pysam = None  # type: ignore


def _require_pybedtools() -> "pybedtools_types":
    if pybedtools is not None:
        return pybedtools
    return require("pybedtools", extra="pybedtools", purpose="bedtools Python backend")


def _require_pybigwig() -> "pybigwig_types":
    if pyBigWig is not None:
        return pyBigWig
    return require("pyBigWig", extra="pybigwig", purpose="BigWig Python backend")


def _require_pysam() -> "pysam_types":
    if pysam is not None:
        return pysam
    return require("pysam", extra="pysam", purpose="FASTA indexing")


def _resolve_backend(backend: str | None, *, tool: str, python_available: bool, cli_name: str) -> str:
    choice = (backend or "auto").strip().lower()
    if choice not in {"auto", "python", "cli"}:
        raise ValueError(f"{tool}_backend must be one of: auto, python, cli")
    if choice == "python":
        if not python_available:
            raise RuntimeError(f"{tool}_backend=python requires the Python package to be installed.")
        return "python"
    if choice == "cli":
        if not shutil.which(cli_name):
            raise RuntimeError(f"{tool}_backend=cli requires {cli_name} in PATH.")
        return "cli"
    if shutil.which(cli_name):
        return "cli"
    if python_available:
        return "python"
    raise RuntimeError(f"Neither Python nor CLI backend is available for {tool}.")


def _read_chrom_sizes(chrom_sizes: Path) -> list[tuple[str, int]]:
    sizes: list[tuple[str, int]] = []
    with chrom_sizes.open() as f:
        for line in f:
            chrom, size = line.split()[:2]
            sizes.append((chrom, int(size)))
    return sizes


def _ensure_fasta_index(fasta: Path) -> Path:
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if fai.exists():
        return fai
    if shutil.which("samtools"):
        cp = subprocess.run(
            ["samtools", "faidx", str(fasta)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"samtools faidx failed (exit {cp.returncode}):\n{cp.stderr}")
        return fai
    if pysam is not None:
        pysam_mod = _require_pysam()
        pysam_mod.faidx(str(fasta))
        return fai
    raise RuntimeError("FASTA indexing requires pysam or samtools in PATH.")


def _ensure_chrom_sizes(fasta: Path) -> Path:
    fai = _ensure_fasta_index(fasta)
    chrom_sizes = fasta.with_suffix(".chrom.sizes")
    if chrom_sizes.exists():
        return chrom_sizes
    with fai.open() as f_in, chrom_sizes.open("w") as out:
        for line in f_in:
            chrom, size = line.split()[:2]
            out.write(f"{chrom}\t{size}\n")
    return chrom_sizes


def _bed_to_bigwig(
    fasta: str,
    bed: str,
    *,
    bedtools_backend: str | None = "auto",
    bigwig_backend: str | None = "auto",
) -> str:
    """
    BED → bedGraph → bigWig
    Requires:
      - FASTA must have .fai index present
    """

    bed = Path(bed)
    fa = Path(fasta)  # path to .fa
    parent = bed.parent
    stem = bed.stem
    chrom_sizes = _ensure_chrom_sizes(fa)

    bedgraph = parent / f"{stem}.bedgraph"
    bigwig = parent / f"{stem}.bw"

    # 1) Compute coverage → bedGraph
    bedtools_choice = _resolve_backend(
        bedtools_backend, tool="bedtools", python_available=pybedtools is not None, cli_name="bedtools"
    )
    if bedtools_choice == "python":
        logger.debug(f"[pybedtools] generating coverage bedgraph from {bed}")
        pybedtools_mod = _require_pybedtools()
        bt = pybedtools_mod.BedTool(str(bed))
        # bedtools genomecov -bg
        coverage = bt.genome_coverage(bg=True, genome=str(chrom_sizes))
        coverage.saveas(str(bedgraph))
    else:
        if not shutil.which("bedtools"):
            raise RuntimeError("bedtools is required but not available in PATH.")
        cmd = [
            "bedtools",
            "genomecov",
            "-i",
            str(bed),
            "-g",
            str(chrom_sizes),
            "-bg",
        ]
        logger.debug("[bedtools] generating coverage bedgraph: %s", " ".join(cmd))
        with bedgraph.open("w") as out:
            cp = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"bedtools genomecov failed (exit {cp.returncode}):\n{cp.stderr}")

    # 2) Convert bedGraph → BigWig via pyBigWig
    bigwig_choice = _resolve_backend(
        bigwig_backend, tool="bigwig", python_available=pyBigWig is not None, cli_name="bedGraphToBigWig"
    )
    if bigwig_choice == "python":
        logger.debug(f"[pyBigWig] converting bedgraph → bigwig: {bigwig}")
        pybigwig_mod = _require_pybigwig()
        bw = pybigwig_mod.open(str(bigwig), "w")
        bw.addHeader(_read_chrom_sizes(chrom_sizes))

        with bedgraph.open() as f:
            for line in f:
                chrom, start, end, coverage = line.strip().split()
                bw.addEntries(chrom, int(start), ends=int(end), values=float(coverage))

        bw.close()
    else:
        if not shutil.which("bedGraphToBigWig"):
            raise RuntimeError("bedGraphToBigWig is required but not available in PATH.")
        cmd = ["bedGraphToBigWig", str(bedgraph), str(chrom_sizes), str(bigwig)]
        logger.debug("[bedGraphToBigWig] converting bedgraph → bigwig: %s", " ".join(cmd))
        cp = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"bedGraphToBigWig failed (exit {cp.returncode}):\n{cp.stderr}")

    logger.debug(f"BigWig written: {bigwig}")
    return str(bigwig)


def _plot_bed_histograms(
    bed_file,
    plotting_directory,
    fasta,
    *,
    bins=60,
    clip_quantiles=(0.0, 0.995),
    cov_bin_size=1000,  # coverage bin size in bp
    rows_per_fig=6,  # paginate if many chromosomes
    include_mapq_quality=True,  # add MAPQ + avg read quality columns to grid
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
    os.makedirs(plotting_directory, exist_ok=True)

    bed_basename = os.path.basename(bed_file).rsplit(".bed", 1)[0]
    logger.debug(f"[plot_bed_histograms] Loading: {bed_file}")

    # Load BED-like table
    cols = ["chrom", "start", "end", "read_len", "qname", "mapq", "avg_q"]
    df = pd.read_csv(
        bed_file,
        sep="\t",
        header=None,
        names=cols,
        dtype={
            "chrom": str,
            "start": int,
            "end": int,
            "read_len": int,
            "qname": str,
            "mapq": float,
            "avg_q": float,
        },
    )

    # Drop unaligned records (chrom == '*') if present
    df = df[df["chrom"] != "*"].copy()
    if df.empty:
        logger.debug("[plot_bed_histograms] No aligned reads found; nothing to plot.")
        return

    # Ensure coordinate mode consistent; convert to 0-based half-open for bin math internally
    # Input is typically one_based inclusive (from your writer).
    if coordinate_mode not in {"one_based", "zero_based"}:
        raise ValueError("coordinate_mode must be 'one_based' or 'zero_based'")

    if coordinate_mode == "one_based":
        # convert to 0-based half-open [start0, end0)
        start0 = df["start"].to_numpy() - 1
        end0 = df["end"].to_numpy()  # inclusive in input -> +1 already handled by not subtracting
    else:
        # already 0-based half-open (assumption)
        start0 = df["start"].to_numpy()
        end0 = df["end"].to_numpy()

    # Clip helper for hist tails
    def _clip_series(s, q=(0.0, 0.995)):
        """Clip a Series to quantile bounds for plotting."""
        if q is None:
            return s.to_numpy()
        lo = s.quantile(q[0]) if q[0] is not None else s.min()
        hi = s.quantile(q[1]) if q[1] is not None else s.max()
        x = s.to_numpy(dtype=float)
        return np.clip(x, lo, hi)

    # Load chromosome order/lengths from FASTA
    pysam_mod = _require_pysam()
    with pysam_mod.FastaFile(fasta) as fa:
        ref_names = list(fa.references)
        ref_lengths = dict(zip(ref_names, fa.lengths))

    # Keep only chroms present in FASTA and with at least one read
    chroms = [c for c in df["chrom"].unique() if c in ref_lengths]
    # Order chromosomes by FASTA order
    chrom_order = [c for c in ref_names if c in chroms]

    if not chrom_order:
        logger.debug(
            "[plot_bed_histograms] No chromosomes from BED are present in FASTA; aborting."
        )
        return

    # Pagination
    def _sanitize(name: str) -> str:
        """Sanitize a string for use in filenames."""
        return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)

    cols_per_fig = 4 if include_mapq_quality else 2

    for start_idx in range(0, len(chrom_order), rows_per_fig):
        chunk = chrom_order[start_idx : start_idx + rows_per_fig]
        nrows = len(chunk)
        ncols = cols_per_fig

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 2.6 * nrows), dpi=160, squeeze=False
        )

        for r, chrom in enumerate(chunk):
            chrom_len = ref_lengths[chrom]
            mask = df["chrom"].to_numpy() == chrom

            # Slice per-chrom arrays for speed
            s0 = start0[mask]
            e0 = end0[mask]
            len_arr = df.loc[mask, "read_len"]
            mapq_arr = df.loc[mask, "mapq"]
            q_arr = df.loc[mask, "avg_q"]

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
                cov[lo : hi + 1] += 1

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
                ax.hist(
                    _clip_series(mapq_arr.fillna(0), clip_quantiles),
                    bins=bins,
                    edgecolor="black",
                    alpha=0.7,
                )
                if r == 0:
                    ax.set_title("MAPQ")
                ax.set_xlabel("MAPQ")
                ax.grid(alpha=0.25)

                # --- Col 4: Avg base quality ---
                ax = axes[r, 3]
                ax.hist(
                    _clip_series(q_arr.fillna(np.nan), clip_quantiles),
                    bins=bins,
                    edgecolor="black",
                    alpha=0.7,
                )
                if r == 0:
                    ax.set_title("Avg base qual")
                ax.set_xlabel("Phred")
                ax.grid(alpha=0.25)

        fig.suptitle(
            f"{bed_basename} — per-chromosome QC "
            f"({'len,cov,MAPQ,qual' if include_mapq_quality else 'len,cov'})",
            y=0.995,
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        page = start_idx // rows_per_fig + 1
        out_png = os.path.join(plotting_directory, f"{_sanitize(bed_basename)}_qc_page{page}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    logger.debug("[plot_bed_histograms] Done.")


def aligned_BAM_to_bed(
    aligned_BAM,
    out_dir,
    fasta,
    make_bigwigs,
    threads=None,
    *,
    bedtools_backend: str | None = "auto",
    bigwig_backend: str | None = "auto",
):
    """
    Takes an aligned BAM as input and writes a BED file of reads as output.
    Bed columns are: Record name, start position, end position, read length, read name, mapping quality, read quality.

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract to a BED file.
        out_dir (str): Directory to output files.
        fasta (str): File path to the reference genome.
        make_bigwigs (bool): Whether to generate bigwig files.
        threads (int): Number of threads to use.

    Returns:
        None
    """
    threads = threads or os.cpu_count()  # Use max available cores if not specified

    # Create necessary directories
    plotting_dir = out_dir / "bed_cov_histograms"
    bed_dir = out_dir / "beds"
    make_dirs([plotting_dir, bed_dir])

    bed_output = bed_dir / str(aligned_BAM.name).replace(".bam", "_bed.bed")

    logger.debug(f"Creating BED-like file from BAM (with MAPQ and avg base quality): {aligned_BAM}")

    pysam_mod = _require_pysam()
    with pysam_mod.AlignmentFile(aligned_BAM, "rb") as bam, open(bed_output, "w") as out:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                chrom = "*"
                start1 = 1
                rl = read.query_length or 0
                mapq = 0
            else:
                chrom = bam.get_reference_name(read.reference_id)
                # pysam reference_start is 0-based → +1 for 1-based SAM-like start
                start1 = int(read.reference_start) + 1
                rl = read.query_length or 0
                mapq = int(read.mapping_quality)

            # End position in 1-based inclusive coords
            end1 = start1 + (rl or 0) - 1

            qname = read.query_name
            quals = read.query_qualities
            if quals is None or rl == 0:
                avg_q = float("nan")
            else:
                avg_q = float(np.mean(quals))

            out.write(f"{chrom}\t{start1}\t{end1}\t{rl}\t{qname}\t{mapq}\t{avg_q:.3f}\n")

    logger.debug(f"BED-like file created: {bed_output}")

    def split_bed(bed):
        """Splits into aligned and unaligned reads (chrom == '*')."""
        bed = str(bed)
        aligned = bed.replace(".bed", "_aligned.bed")
        unaligned = bed.replace(".bed", "_unaligned.bed")
        with (
            open(bed, "r") as infile,
            open(aligned, "w") as aligned_out,
            open(unaligned, "w") as unaligned_out,
        ):
            for line in infile:
                (unaligned_out if line.startswith("*\t") else aligned_out).write(line)
        os.remove(bed)
        return aligned

    logger.debug(f"Splitting: {bed_output}")
    aligned_bed = split_bed(bed_output)

    with ProcessPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(_plot_bed_histograms, aligned_bed, plotting_dir, fasta))
        if make_bigwigs:
            futures.append(
                executor.submit(
                    _bed_to_bigwig,
                    fasta,
                    aligned_bed,
                    bedtools_backend=bedtools_backend,
                    bigwig_backend=bigwig_backend,
                )
            )
        concurrent.futures.wait(futures)

    logger.debug("Processing completed successfully.")


def extract_read_lengths_from_bed(file_path):
    """
    Load a dict of read names that points to the read length

    Params:
        file_path (str): file path to a bed file
    Returns:
        read_dict (dict)
    """
    import pandas as pd

    columns = ["chrom", "start", "end", "length", "name"]
    df = pd.read_csv(file_path, sep="\t", header=None, names=columns, comment="#")
    read_dict = {}
    for _, row in df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        name = row["name"]
        length = row["length"]
        read_dict[name] = length

    return read_dict
